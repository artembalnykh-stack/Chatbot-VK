"""Фильтры безопасности: антимат, защита от prompt injection, маскировка ПД.

Модуль намеренно не зависит от остальных модулей проекта, чтобы его можно
было переиспользовать где угодно (в том числе в logging-фильтрах).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Final

try:  # better-profanity необязателен: без него работает встроенный фильтр
    from better_profanity import profanity
except ImportError:  # pragma: no cover - зависит от окружения
    profanity = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Ответы-заглушки (используются обработчиком VK)
# --------------------------------------------------------------------------- #
PROFANITY_REPLY: Final = (
    "Пожалуйста, используйте корректные выражения. "
    "Я готов помочь с вопросами об услугах AI-Мастерской."
)
INJECTION_REPLY: Final = (
    "Я — Стас, консультант AI-Мастерской. "
    "Могу помочь с вопросами об услугах студии."
)

# --------------------------------------------------------------------------- #
# Антимат
# --------------------------------------------------------------------------- #
# Храним именно КОРНИ, чтобы ловить словоформы ("хуйня", "пиздец", "заебал").
# Список можно и нужно дополнять под свою аудиторию.
_MAT_STEMS: Final[tuple[str, ...]] = (
    "хуй",
    "хуе",
    "хуи",
    "пизд",
    "бляд",
    "блят",
    "ебат",
    "ебал",
    "ебан",
    "ебет",
    "ебуч",
    "ебок",
    "заеб",
    "наеб",
    "уеб",
    "отъеб",
    "въеб",
    "долбоеб",
    "мудак",
    "мудил",
    "гондон",
    "гандон",
    "пидор",
    "пидар",
    "педер",
    "сука",
    "суки",
    "суке",
    "суку",
    "сукой",
    "охуе",
    "ахуе",
    "нехуй",
    "жоп",
    "дерьм",
    "говн",
)

# Латиница/цифры -> кириллица: ловим "xyй", "пuздец", "xуй" и прочие обходы.
_LEET_TRANSLATION = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
        "0": "о",
        "3": "з",
        "4": "ч",
        "6": "б",
    }
)

_WORD_RE = re.compile(r"[0-9a-zа-я]+")
_REPEAT_RE = re.compile(r"(.)\1+")


def _init_profanity() -> None:
    """Догружаем русские корни в better-profanity (английский список остаётся)."""
    if profanity is None:  # pragma: no cover
        logger.warning(
            "better-profanity не установлен: работает только встроенный фильтр. "
            "Установите: pip install better-profanity"
        )
        return
    profanity.load_censor_words()
    profanity.add_censor_words(list(_MAT_STEMS))


_init_profanity()


def _normalized_words(text: str) -> list[str]:
    """Нижний регистр, ё -> е, только буквы/цифры."""
    return _WORD_RE.findall(text.lower().replace("ё", "е"))


def contains_profanity(text: str) -> bool:
    """True, если в тексте есть мат (русский по корням, английский — better-profanity)."""
    if not text:
        return False

    words = _normalized_words(text)
    for word in words:
        # Приводим leet-замены и схлопываем повторы: "хуууй" -> "хуй".
        candidate = _REPEAT_RE.sub(r"\1", word.translate(_LEET_TRANSLATION))
        if any(candidate.startswith(stem) for stem in _MAT_STEMS):
            return True

    if profanity is not None and profanity.contains_profanity(" ".join(words)):
        return True
    return False


# --------------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------------- #
# Паттерны ищем в нормализованном тексте (нижний регистр, без пунктуации).
# Список легко расширять; см. README, раздел "Безопасность".
_INJECTION_PATTERNS: Final[tuple[str, ...]] = (
    # RU
    "забудь инструкц",
    "забудь все инструкц",
    "забудь предыдущ",
    "игнорируй инструкц",
    "игнорируй предыдущ",
    "игнорируй все",
    "не обращай вниман",
    "новые инструкц",
    "теперь ты",
    "ты теперь",
    "представь что ты",
    "притворись",
    "смени роль",
    "отмени инструкц",
    "покажи инструкц",
    "раскрой инструкц",
    "покажи промпт",
    "систем промпт",
    "системный промпт",
    "режим разработчика",
    "режим бога",
    # EN
    "ignore previous",
    "ignore all previous",
    "ignore instructions",
    "forget instructions",
    "forget all",
    "system prompt",
    "developer mode",
    "jailbreak",
    "jail break",
    "do anything now",
    "dan mode",
    "act as",
    "pretend to be",
    "you are now",
    "new instructions",
    "reveal your prompt",
)


def contains_prompt_injection(text: str) -> bool:
    """True, если сообщение похоже на попытку инъекции в системный промпт."""
    if not text:
        return False
    normalized = " ".join(_normalized_words(text))
    return any(pattern in normalized for pattern in _INJECTION_PATTERNS)


# --------------------------------------------------------------------------- #
# Маскировка и редактирование ПД
# --------------------------------------------------------------------------- #
_CARD_RE = re.compile(r"\b\d{16}\b")
_PHONE_RE = re.compile(
    r"(?:\+7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def redact_pii(text: str) -> str:
    """Убирает телефоны/e-mail/номера карт — для логов и для промпта GigaChat."""
    if not text:
        return ""
    text = _CARD_RE.sub("[номер скрыт]", text)
    text = _EMAIL_RE.sub("[e-mail скрыт]", text)
    return _PHONE_RE.sub("[телефон скрыт]", text)


def mask_phone(phone: str) -> str:
    """Маска телефона для логов/уведомлений: +7 (XXX) XXX-XX-XX.

    Формат зафиксирован требованиями 152-ФЗ в ТЗ. Если оператору нужно
    отличать заявки в уведомлении, можно оставить последние 2 цифры.
    """
    if not phone:
        return "+7 (XXX) XXX-XX-XX"
    return "+7 (XXX) XXX-XX-XX"


def mask_name(name: str) -> str:
    """Маска имени: «Иван» -> «Ив***»."""
    cleaned = (name or "").strip()
    if not cleaned:
        return "***"
    return f"{cleaned[:2]}***"


def mask_profile(url: str) -> str:
    """Маска ссылки на профиль: «https://vk.com/ivan» -> «vk.com/id***»."""
    cleaned = (url or "").replace("https://", "").replace("http://", "").rstrip("/")
    domain = cleaned.split("/")[0] or "vk.com"
    return f"{domain}/id***"


def pseudonym(user_id: int) -> str:
    """Псевдоним для логов: позволяет сопоставлять события, не раскрывая user_id."""
    digest = hashlib.sha256(f"vk:{user_id}".encode("utf-8")).hexdigest()
    return f"u_{digest[:10]}"


class PIIScrubFilter(logging.Filter):
    """Вычищает телефоны/e-mail/карты из текста любого лог-сообщения."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - защита от битых аргументов
            return True
        scrubbed = redact_pii(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        return True
