"""VK: Long Poll, отправка сообщений, клавиатура и обработка входящих."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Set

import aiohttp

from ai_adapter import AIAdapter, GigaChatError
from kb_search import KnowledgeBase
from lead_collector import LeadCollector, LeadProfile
from security import (
    INJECTION_REPLY,
    PROFANITY_REPLY,
    contains_profanity,
    contains_prompt_injection,
    pseudonym,
)
from utils import Settings

logger = logging.getLogger(__name__)

VK_API_BASE = "https://api.vk.com/method"
_RETRYABLE_CODES = frozenset({1, 6, 9, 10})

GREETING = (
    "Здравствуйте! Я Стас, консультант AI-Мастерской — студии контента и автоматизации.\n"
    "Помогу разобраться с услугами: карточки для маркетплейсов, тексты, лендинги, "
    "чат-боты и автоматизация.\n"
    "Напишите, что Вас интересует. Если захотите обсудить задачу со специалистом, "
    "напишите «оператор»."
)
TEXT_ONLY_REPLY = (
    "Я пока понимаю только текстовые сообщения. "
    "Пожалуйста, напишите ваш вопрос текстом."
)
QUESTION_PROMPT = "Напишите, что Вас интересует, и я постараюсь помочь."
RATE_LIMIT_REPLY = (
    "Слишком много сообщений. Пожалуйста, подождите {seconds} сек. "
    "и повторите вопрос."
)
NO_KB_ANSWER = (
    "Не хочу вводить Вас в заблуждение — этот вопрос лучше уточнить у специалиста. "
    "Напишите «оператор», чтобы оставить заявку на консультацию."
)
AI_UNAVAILABLE = (
    "Сервис временно недоступен. Пожалуйста, попробуйте через 5 минут "
    "или оставьте заявку оператору."
)

_BUTTON_QUESTION = "задать вопрос"
_BUTTON_OPERATOR = "связаться с оператором"
_START_WORDS = frozenset({"/start", "start", "начать"})
_GREETING_WORDS = frozenset(
    {"привет", "здравствуйте", "здравствуй", "добрый день", "добрый вечер", "доброе утро"}
)
_OPERATOR_RE = re.compile(r"\b(оператор\w*|менеджер\w*)\b")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))


class VKError(RuntimeError):
    """Ошибка VK API."""


@dataclass(frozen=True)
class VKUser:
    user_id: int
    name: str
    profile_url: str


@dataclass(frozen=True)
class LongPollServer:
    url: str
    key: str
    ts: str


TYPING_MIN_SECONDS = 1.0
TYPING_PING_SECONDS = 4.0


class VKClient:
    """Минимальный асинхронный клиент VK API (без лишних зависимостей)."""

    def __init__(
        self,
        token: str,
        api_version: str,
        session: aiohttp.ClientSession,
    ) -> None:
        self._token = token
        self._version = api_version
        self._session = session

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._session

    async def call(self, method: str, **params: Any) -> Any:
        payload: Dict[str, Any] = {
            "access_token": self._token,
            "v": self._version,
            **params,
        }
        url = f"{VK_API_BASE}/{method}"

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with self._session.post(url, data=payload) as response:
                    data = await response.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                await asyncio.sleep(1.5 * (attempt + 1))
                continue

            if isinstance(data, dict) and "error" in data:
                error = data["error"]
                code = int(error.get("error_code", 0))
                if code in _RETRYABLE_CODES and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise VKError(
                    f"{method}: [{code}] {error.get('error_msg', 'unknown error')}"
                )
            if not isinstance(data, dict) or "response" not in data:
                raise VKError(f"{method}: неожиданный ответ VK API")
            return data["response"]

        raise VKError(f"{method}: сетевая ошибка ({last_error})")

    async def get_long_poll_server(self, group_id: int) -> LongPollServer:
        data = await self.call("groups.getLongPollServer", group_id=group_id)
        try:
            return LongPollServer(
                url=str(data["server"]),
                key=str(data["key"]),
                ts=str(data["ts"]),
            )
        except (KeyError, TypeError) as exc:
            raise VKError("groups.getLongPollServer: некорректный ответ") from exc

    async def send_message(self, peer_id: int, text: str) -> None:
        params: Dict[str, Any] = {
            "peer_id": peer_id,
            "message": text,
            "random_id": secrets.randbits(31),
            "dont_parse_links": 1,
        }

        try:
            await self.call("messages.send", **params)
        except VKError as exc:
            # 901 — пользователь запретил сообщения сообщества: это не ошибка бота.
            logger.warning("Не удалось отправить сообщение: %s", exc)

    async def set_typing(self, user_id: int, group_id: int) -> None:
        """Показать статус «печатает». Сбой индикатора не должен ломать ответ."""
        try:
            await self.call(
                "messages.setActivity",
                user_id=user_id,
                type="typing",
                group_id=group_id,
            )
        except Exception as exc:  # индикатор не критичен для отправки ответа
            logger.debug("Не удалось показать «печатает»: %s", exc)

    async def get_user(self, user_id: int) -> VKUser:
        response = await self.call(
            "users.get",
            user_ids=user_id,
            fields="screen_name",
        )
        if not isinstance(response, list) or not response:
            return VKUser(
                user_id=user_id,
                name="Клиент",
                profile_url=f"https://vk.com/id{user_id}",
            )

        user = response[0]
        first_name = str(user.get("first_name", "")).strip()
        last_name = str(user.get("last_name", "")).strip()
        screen_name = str(user.get("screen_name", "")).strip()
        name = " ".join(part for part in (first_name, last_name) if part) or "Клиент"
        profile_url = (
            f"https://vk.com/{screen_name}" if screen_name else f"https://vk.com/id{user_id}"
        )
        return VKUser(user_id=user_id, name=name, profile_url=profile_url)


class LongPollListener:
    """Bots Long Poll: работает без белого IP и входящих портов."""

    def __init__(self, vk: VKClient, group_id: int, wait: int = 25) -> None:
        self._vk = vk
        self._group_id = group_id
        self._wait = wait

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        # Стартовое подключение: VK может быть временно недоступен.
        while True:
            try:
                server = await self._vk.get_long_poll_server(self._group_id)
                break
            except VKError as exc:
                logger.error(
                    "Long Poll: не удалось подключиться (%s), повтор через 10 сек.", exc
                )
                await asyncio.sleep(10)

        logger.info(
            "Long Poll подключён: сообщество %s, ожидаю сообщения.", self._group_id
        )

        while True:
            try:
                data = await self._poll(server)
            except (aiohttp.ClientError, asyncio.TimeoutError, VKError) as exc:
                logger.warning("Long Poll: ошибка соединения (%s), переподключаюсь", exc)
                await asyncio.sleep(5)
                server = await self._refresh(server)
                continue

            if "failed" in data:
                failed = data.get("failed")
                logger.warning("Long Poll: код ошибки %s, обновляю сервер", failed)
                if failed == 1 and "ts" in data:
                    server = LongPollServer(server.url, server.key, str(data["ts"]))
                else:
                    server = await self._refresh(server)
                    if failed not in (2, 3):
                        await asyncio.sleep(3)
                continue

            server = LongPollServer(
                server.url, server.key, str(data.get("ts", server.ts))
            )
            for update in data.get("updates", []):
                if isinstance(update, dict):
                    yield update

    async def _refresh(self, server: LongPollServer) -> LongPollServer:
        try:
            return await self._vk.get_long_poll_server(self._group_id)
        except VKError as exc:
            logger.error("Long Poll: не удалось получить сервер: %s", exc)
            return server

    async def _poll(self, server: LongPollServer) -> Dict[str, Any]:
        params = {
            "act": "a_check",
            "key": server.key,
            "ts": server.ts,
            "wait": self._wait,
            "mode": 2,
            "version": 3,
        }
        timeout = aiohttp.ClientTimeout(total=self._wait + 15)
        async with self._vk.session.get(
            server.url, params=params, timeout=timeout
        ) as response:
            data = await response.json(content_type=None)
        if not isinstance(data, dict):
            raise VKError("Long Poll: некорректный ответ")
        return data


class MessageProcessor:
    """Оркестратор: команды → заявки → лимиты → безопасность → БЗ → GigaChat."""

    def __init__(
        self,
        settings: Settings,
        vk: VKClient,
        knowledge_base: KnowledgeBase,
        ai: AIAdapter,
        leads: LeadCollector,
    ) -> None:
        self._settings = settings
        self._vk = vk
        self._kb = knowledge_base
        self._ai = ai
        self._leads = leads
        self._greeted: Set[int] = set()
        self._last_request: Dict[int, float] = {}

    async def handle_event(self, event: Dict[str, Any]) -> None:
        try:
            await self._handle(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Необработанная ошибка при обработке события")

    def purge_stale(self) -> None:
        now = time.monotonic()
        stale = [
            user_id
            for user_id, last in self._last_request.items()
            if now - last > 3600
        ]
        for user_id in stale:
            self._last_request.pop(user_id, None)

    # ------------------------------------------------------------------ #
    # Маршрутизация
    # ------------------------------------------------------------------ #
    async def _handle(self, event: Dict[str, Any]) -> None:
        if event.get("type") != "message_new":
            return

        payload = event.get("object") or {}
        message = payload.get("message") or {}
        if message.get("out"):
            return

        from_id = message.get("from_id")
        peer_id = message.get("peer_id", from_id)
        if not isinstance(from_id, int) or from_id <= 0:
            return
        if peer_id != from_id:
            logger.debug("Пропущено сообщение из беседы")
            await self._send(peer_id, "Я отвечаю только в личных сообщениях сообщества.")
            return

        text = str(message.get("text") or "").strip()
        attachments = message.get("attachments") or []
        normalized = _normalize(text)

        greeted_now = False
        if from_id not in self._greeted:
            self._greeted.add(from_id)
            greeted_now = True
            await self._send(peer_id, GREETING)

        if normalized in _START_WORDS:
            self._ai.reset(from_id)
            self._leads.reset(from_id)
            if not greeted_now:
                await self._send(peer_id, GREETING)
            return

        if normalized in _GREETING_WORDS:
            if not greeted_now:
                await self._send(peer_id, GREETING)
            return

        # Нет текста: вложения и пустые сообщения обрабатываем мягко.
        if not text:
            if attachments:
                await self._send(peer_id, TEXT_ONLY_REPLY)
            else:
                await self._send(peer_id, QUESTION_PROMPT)
            return

        # Диалог заявки идёт мимо лимитов и ИИ (там персональные данные).
        if self._leads.is_active(from_id):
            profile = await self._profile(from_id)
            step = await self._leads.handle(from_id, text, profile)
            await self._send(peer_id, step.text)
            return

        if normalized == _BUTTON_QUESTION:
            await self._send(peer_id, QUESTION_PROMPT)
            return

        if normalized == _BUTTON_OPERATOR or _OPERATOR_RE.search(normalized):
            logger.info("Запрос оператора: пользователь=%s", pseudonym(from_id))
            await self._send(peer_id, self._leads.start(from_id))
            return

        remaining = self._consume_rate_limit(from_id)
        if remaining:
            await self._send(
                peer_id, RATE_LIMIT_REPLY.format(seconds=remaining)
            )
            return

        if contains_profanity(text):
            logger.info("Антимат: пользователь=%s", pseudonym(from_id))
            await self._send(peer_id, PROFANITY_REPLY)
            return

        if contains_prompt_injection(text):
            logger.warning("Prompt injection: пользователь=%s", pseudonym(from_id))
            await self._send(peer_id, INJECTION_REPLY)
            return

        await self._answer_from_knowledge_base(from_id, peer_id, text)

    async def _answer_from_knowledge_base(
        self, user_id: int, peer_id: int, text: str
    ) -> None:
        hits = self._kb.search(text)
        if not hits:
            logger.info(
                "БЗ: нет релевантных фрагментов (пользователь=%s, длина=%d)",
                pseudonym(user_id), len(text),
            )
            await self._send(peer_id, NO_KB_ANSWER)
            return

        logger.info(
            "БЗ: найдено фрагментов=%d, лучший источник=%s, пользователь=%s",
            len(hits), hits[0].chunk.source, pseudonym(user_id),
        )
        context = self._kb.format_context(hits)
        # Показываем «печатает» сразу, не дожидаясь первого тика фоновой задачи:
        # при быстром ответе задача может не успеть запуститься до отмены.
        await self._vk.set_typing(user_id, self._settings.vk_group_id)
        typing = asyncio.create_task(self._keep_typing(user_id))
        started = time.monotonic()
        try:
            answer = await self._ai.answer(user_id, text, context)
        except GigaChatError:
            logger.warning("GigaChat недоступен, fallback-ответ выдан")
            await self._send(peer_id, AI_UNAVAILABLE)
            return
        finally:
            typing.cancel()
            try:
                await typing
            except asyncio.CancelledError:
                pass

        # Даём индикатору «печатает» побыть видимым хотя бы секунду.
        remaining = TYPING_MIN_SECONDS - (time.monotonic() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)

        await self._send(peer_id, answer)

    async def _keep_typing(self, user_id: int) -> None:
        """Обновляет статус «печатает», пока GigaChat готовит ответ."""
        while True:
            await asyncio.sleep(TYPING_PING_SECONDS)
            await self._vk.set_typing(user_id, self._settings.vk_group_id)

    # ------------------------------------------------------------------ #
    # Вспомогательное
    # ------------------------------------------------------------------ #
    def _consume_rate_limit(self, user_id: int) -> int:
        """Возвращает 0, если запрос разрешён, иначе — сколько секунд ждать."""
        now = time.monotonic()
        limit = self._settings.rate_limit_seconds
        last = self._last_request.get(user_id)
        if last is not None and now - last < limit:
            return int(limit - (now - last)) + 1
        self._last_request[user_id] = now
        return 0

    async def _profile(self, user_id: int) -> LeadProfile:
        try:
            user = await self._vk.get_user(user_id)
            return LeadProfile(
                user_id=user_id,
                name=user.name,
                profile_url=user.profile_url,
            )
        except Exception:
            logger.warning(
                "Не удалось получить профиль VK: пользователь=%s", pseudonym(user_id)
            )
            return LeadProfile(
                user_id=user_id,
                name="Клиент",
                profile_url=f"https://vk.com/id{user_id}",
            )

    async def _send(self, peer_id: int, text: str) -> None:
        if not text:
            return
        await self._vk.send_message(peer_id, text)


__all__ = [
    "LongPollListener",
    "MessageProcessor",
    "VKClient",
    "VKError",
    "VKUser",
]
