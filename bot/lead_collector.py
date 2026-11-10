"""Сбор заявок: контакт → согласие на обработку ПД → сохранение.

Поля заявки (VK-профиль и контакт) не добавляются в запросы GigaChat.
Пользовательские сообщения обрабатываются отдельно AIAdapter и могут включать
имя или ссылку, введённые самим пользователем. В логах и уведомлениях
администратору показываются только маски.

Куда писать заявки:
1. локальный бэкап (JSONL, права 0600) — всегда, чтобы заявка не потерялась;
2. Яндекс Таблицы через REST API — опционально и «спяще»: канал включается
   только при заданных YANDEX_TABLES_TOKEN/ID, без них код не вызывается.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, Final, Optional
from urllib.parse import quote

import aiohttp

from security import mask_name, mask_phone, mask_profile, pseudonym, redact_pii
from utils import PROJECT_DIR, Settings

logger = logging.getLogger(__name__)

PHONE_PATTERN: Final = re.compile(
    r"^(\+7|8)?[\s\-]?\(?[489][0-9]{2}\)?[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}$"
)
VK_CONTACT: Final = "ВКонтакте"

CONTACT_QUESTION: Final = (
    "Как вам удобнее, чтобы оператор с вами связался? "
    "Напишите ваш номер телефона или ответьте «ВКонтакте» — тогда оператор напишет вам здесь."
)
CONSENT_QUESTION: Final = (
    "Для передачи заявки оператору мне нужно обработать ваши персональные данные: "
    "имя профиля ВКонтакте и контакт для связи. "
    "Вы даёте согласие на обработку ПД? (Да/Нет)"
)
CONSENT_RETRY: Final = "Пожалуйста, ответьте однозначно: «Да» или «Нет»."
CONSENT_REFUSED: Final = (
    "Хорошо, я не буду сохранять ваши данные. "
    "Если передумаете — напишите «оператор»."
)
CONSENT_FAILED: Final = (
    "Без явного согласия я не могу сохранить заявку. "
    "Вы можете вернуться к этому позже, написав «оператор»."
)
PHONE_RETRY: Final = (
    "Не удалось распознать номер. Введите телефон в формате "
    "+7 (999) 123-45-67. Осталось попыток: {left}."
)
CONTACT_RETRY: Final = (
    "Пожалуйста, напишите номер телефона (например, +7 999 123-45-67) "
    "или слово «ВКонтакте»."
)
CONTACT_FAILED: Final = (
    "Не получилось распознать контакт за 3 попытки, заявка не создана. "
    "Вы можете вернуться к этому позже, написав «оператор»."
)
CANCEL_REPLY: Final = "Хорошо, отменил. Если понадобится — напишите «оператор»."
SESSION_EXPIRED: Final = (
    "Похоже, диалог заявки устарел. Напишите «оператор», чтобы начать заново."
)
SUCCESS: Final = "Заявка принята! Оператор свяжется с вами в ближайшее время."
SAVE_FAILED: Final = (
    "Не удалось сохранить заявку. Пожалуйста, попробуйте позже "
    "или напишите нам ещё раз."
)

MAX_ATTEMPTS: Final = 3
_CANCEL_WORDS: Final = frozenset({"отмена", "cancel", "стоп", "/cancel"})
_VK_HINTS: Final = (
    "вк",
    "вконтакте",
    "в контакте",
    "в лс",
    "в личку",
    "здесь",
    "тут",
    "напишите мне",
)


@dataclass(frozen=True)
class LeadProfile:
    """Данные профиля, необходимые для заявки (в логи не попадают)."""

    user_id: int
    name: str
    profile_url: str


@dataclass(frozen=True)
class Lead:
    created_at: str
    user_id: int
    name: str
    profile_url: str
    contact: str

    def as_row(self) -> list[str]:
        return [
            self.created_at,
            str(self.user_id),
            self.name,
            self.profile_url,
            self.contact,
            "да",
        ]

    def as_json(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "user_id": self.user_id,
            "name": self.name,
            "profile_url": self.profile_url,
            "contact": self.contact,
            "consent": True,
        }


@dataclass(frozen=True)
class SaveOutcome:
    backup_ok: bool
    remote_ok: Optional[bool]  # None — Яндекс Таблицы не настроены (не ошибка)

    @property
    def any_ok(self) -> bool:
        return self.backup_ok or bool(self.remote_ok)


@dataclass(frozen=True)
class LeadStep:
    text: str
    finished: bool


class _Stage(Enum):
    CONTACT = auto()
    CONSENT = auto()


@dataclass
class _Session:
    stage: _Stage
    phone_attempts: int = 0
    consent_attempts: int = 0
    contact: Optional[str] = None
    last_seen: float = field(default_factory=monotonic)


# --------------------------------------------------------------------------- #
# Хранилище заявок
# --------------------------------------------------------------------------- #
class LeadStorage:
    """Локальный бэкап + Яндекс Таблицы."""

    def __init__(self, settings: Settings, session: aiohttp.ClientSession) -> None:
        self._settings = settings
        self._session = session
        self._backup_path: Path = settings.leads_backup_path
        if settings.yandex_enabled:
            logger.info("Яндекс Таблицы включены: заявки дублируются в таблицу.")
        else:
            logger.info(
                "Яндекс Таблицы отключены (YANDEX_TABLES_TOKEN/ID пусты): "
                "заявки сохраняются только в локальный бэкап %s",
                self._backup_path,
            )

    @property
    def backup_path(self) -> Path:
        """Путь к локальному бэкапу заявок (полные ПД, права 0600)."""
        return self._backup_path

    async def save(self, lead: Lead) -> SaveOutcome:
        backup_ok = await self._write_backup(lead)
        remote_ok: Optional[bool] = None
        if self._settings.yandex_enabled:
            remote_ok = await self._append_remote(lead)

        logger.info(
            "Заявка: пользователь=%s, контакт=%s, бэкап=%s, таблицы=%s",
            pseudonym(lead.user_id),
            "ВКонтакте" if lead.contact == VK_CONTACT else mask_phone(lead.contact),
            "ok" if backup_ok else "fail",
            "off" if remote_ok is None else ("ok" if remote_ok else "fail"),
        )
        return SaveOutcome(backup_ok=backup_ok, remote_ok=remote_ok)

    async def _write_backup(self, lead: Lead) -> bool:
        try:
            await asyncio.to_thread(self._append_backup_sync, lead)
            return True
        except OSError as exc:
            logger.error("Не удалось записать локальный бэкап заявки: %s", exc)
            return False

    def _append_backup_sync(self, lead: Lead) -> None:
        self._backup_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._backup_path.exists()
        with self._backup_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(lead.as_json(), ensure_ascii=False) + "\n")
        if is_new:
            # Полные ПД: доступ только владельцу файла.
            os.chmod(self._backup_path, 0o600)

    async def _append_remote(self, lead: Lead) -> bool:
        try:
            url = self._settings.yandex_api_url.format(
                spreadsheet_id=quote(self._settings.yandex_spreadsheet_id, safe=""),
                sheet=quote(self._settings.yandex_sheet, safe=""),
            )
        except (KeyError, IndexError) as exc:
            logger.error("Некорректный шаблон YANDEX_TABLES_API_URL: %s", exc)
            return False

        headers = {
            "Authorization": f"OAuth {self._settings.yandex_token}",
            "Content-Type": "application/json",
        }
        params = {
            "valueInputOption": "USER_ENTERED",
            "insertDataOption": "INSERT_ROWS",
        }
        body = {"values": [lead.as_row()]}

        for attempt in (1, 2):
            try:
                async with self._session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status < 400:
                        return True
                    error_text = redact_pii((await response.text())[:300])
                    logger.warning(
                        "Яндекс Таблицы: HTTP %s (попытка %d): %s",
                        response.status, attempt, error_text,
                    )
                    if response.status < 500:
                        return False
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Яндекс Таблицы: сетевая ошибка (попытка %d): %s", attempt, exc
                )

            if attempt == 1:
                await asyncio.sleep(2)

        return False


# --------------------------------------------------------------------------- #
# Диалог сбора заявки
# --------------------------------------------------------------------------- #
class LeadCollector:
    """Конечный автомат: контакт → согласие → сохранение."""

    def __init__(
        self,
        settings: Settings,
        storage: LeadStorage,
        notifier: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        self._ttl = settings.context_ttl_seconds
        self._storage = storage
        self._notifier = notifier
        self._sessions: Dict[int, _Session] = {}

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #
    def is_active(self, user_id: int) -> bool:
        return self._session(user_id) is not None

    def start(self, user_id: int) -> str:
        self._sessions[user_id] = _Session(stage=_Stage.CONTACT)
        logger.info("Заявка: старт сбора, пользователь=%s", pseudonym(user_id))
        return CONTACT_QUESTION

    def reset(self, user_id: int) -> None:
        self._sessions.pop(user_id, None)

    async def handle(self, user_id: int, text: str, profile: LeadProfile) -> LeadStep:
        session = self._session(user_id)
        if session is None:
            return LeadStep(SESSION_EXPIRED, finished=True)

        session.last_seen = monotonic()
        normalized = _normalize(text)

        if normalized in _CANCEL_WORDS:
            self.reset(user_id)
            return LeadStep(CANCEL_REPLY, finished=True)

        if session.stage is _Stage.CONTACT:
            return self._handle_contact(user_id, session, normalized)
        return await self._handle_consent(user_id, session, text, profile)

    def purge_stale(self) -> int:
        now = monotonic()
        stale = [
            user_id
            for user_id, session in self._sessions.items()
            if now - session.last_seen > self._ttl
        ]
        for user_id in stale:
            self._sessions.pop(user_id, None)
        if stale:
            logger.info("Очищены устаревшие сессии заявок: %d", len(stale))
        return len(stale)

    # ------------------------------------------------------------------ #
    # Шаги диалога
    # ------------------------------------------------------------------ #
    def _handle_contact(
        self, user_id: int, session: _Session, normalized: str
    ) -> LeadStep:
        phone = _extract_phone(normalized)
        if phone is not None:
            session.contact = phone
            session.stage = _Stage.CONSENT
            logger.info(
                "Заявка: контакт получен (телефон), пользователь=%s",
                pseudonym(user_id),
            )
            return LeadStep(CONSENT_QUESTION, finished=False)

        if _is_vk_contact(normalized):
            session.contact = VK_CONTACT
            session.stage = _Stage.CONSENT
            logger.info(
                "Заявка: контакт получен (ВКонтакте), пользователь=%s",
                pseudonym(user_id),
            )
            return LeadStep(CONSENT_QUESTION, finished=False)

        session.phone_attempts += 1
        attempts_left = MAX_ATTEMPTS - session.phone_attempts
        if attempts_left <= 0:
            self.reset(user_id)
            logger.info(
                "Заявка: контакт не распознан за %d попытки, пользователь=%s",
                MAX_ATTEMPTS, pseudonym(user_id),
            )
            return LeadStep(CONTACT_FAILED, finished=True)

        if _looks_like_phone(normalized):
            return LeadStep(PHONE_RETRY.format(left=attempts_left), finished=False)
        return LeadStep(CONTACT_RETRY, finished=False)

    async def _handle_consent(
        self, user_id: int, session: _Session, text: str, profile: LeadProfile
    ) -> LeadStep:
        consent = _parse_consent(text)
        if consent is None:
            session.consent_attempts += 1
            if session.consent_attempts >= MAX_ATTEMPTS:
                self.reset(user_id)
                return LeadStep(CONSENT_FAILED, finished=True)
            return LeadStep(CONSENT_RETRY, finished=False)

        if not consent:
            self.reset(user_id)
            logger.info("Заявка: согласие не дано, пользователь=%s", pseudonym(user_id))
            return LeadStep(CONSENT_REFUSED, finished=True)

        assert session.contact is not None  # гарантируется предыдущим шагом
        lead = Lead(
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            user_id=user_id,
            name=profile.name,
            profile_url=profile.profile_url,
            contact=session.contact,
        )
        self.reset(user_id)

        outcome = await self._storage.save(lead)
        await self._notify_admin(lead, outcome)

        if not outcome.any_ok:
            return LeadStep(SAVE_FAILED, finished=True)
        return LeadStep(SUCCESS, finished=True)

    # ------------------------------------------------------------------ #
    # Уведомление администратора (только маски)
    # ------------------------------------------------------------------ #
    async def _notify_admin(self, lead: Lead, outcome: SaveOutcome) -> None:
        if self._notifier is None:
            return

        contact = (
            "ВКонтакте (личные сообщения)"
            if lead.contact == VK_CONTACT
            else mask_phone(lead.contact)
        )
        lines = [
            "🔔 Новая заявка на консультацию",
            f"Имя: {mask_name(lead.name)}",
            f"Контакт: {contact}",
            f"Профиль: {mask_profile(lead.profile_url)}",
            f"Время: {lead.created_at}",
        ]
        if not outcome.any_ok:
            lines.append("⚠️ Заявку не удалось сохранить — свяжитесь с клиентом вручную.")
        elif outcome.remote_ok is True:
            lines.append("Полные данные: Яндекс Таблицы.")
            if not outcome.backup_ok:
                lines.append("⚠️ Локальный бэкап не записан.")
        elif outcome.remote_ok is False:
            lines.append("⚠️ Яндекс Таблицы недоступны, заявка в локальном бэкапе.")
        else:
            lines.append(
                f"Полные данные: локальный бэкап ({self._backup_path_label()})."
            )

        try:
            await self._notifier("\n".join(lines))
        except Exception:
            logger.warning("Не удалось уведомить администратора", exc_info=True)

    def _backup_path_label(self) -> str:
        """Короткий путь для уведомления: относительно папки bot/, если возможно."""
        path = self._storage.backup_path
        try:
            return str(path.relative_to(PROJECT_DIR))
        except ValueError:
            return str(path)

    def _session(self, user_id: int) -> Optional[_Session]:
        session = self._sessions.get(user_id)
        if session is None:
            return None
        if monotonic() - session.last_seen > self._ttl:
            self._sessions.pop(user_id, None)
            return None
        return session


# --------------------------------------------------------------------------- #
# Разбор ввода
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))


def _looks_like_phone(text: str) -> bool:
    """Похоже ли сообщение на попытку ввести телефон (а не «мне 25 лет»)."""
    return sum(character.isdigit() for character in text) >= 5


def _extract_phone(text: str) -> Optional[str]:
    cleaned = text.strip().strip(" .,;:!?\"'")
    if PHONE_PATTERN.fullmatch(cleaned):
        return cleaned
    compact = re.sub(r"[\s\-()]", "", cleaned)
    if PHONE_PATTERN.fullmatch(compact):
        return compact
    return None


def _is_vk_contact(text: str) -> bool:
    if text in {"вк", "вконтакте"}:
        return True
    return any(hint in text for hint in _VK_HINTS if len(hint) > 2)


def _parse_consent(text: str) -> Optional[bool]:
    normalized = re.sub(r"[^а-яa-z ]", " ", _normalize(text))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    if re.search(r"\b(нет|no|отказ\w*)\b", normalized):
        return False
    # "не согласен", "не хочу", "не разрешаю" — это отказ, а не согласие.
    if re.search(r"\bне\s+(соглас\w*|хочу|разреша\w*|подтвержда\w*|даю)\b", normalized):
        return False
    if re.search(r"\b(да|yes|соглас\w*|подтвержда\w*|разреша\w*)\b", normalized):
        return True
    return None
