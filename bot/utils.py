"""Конфигурация приложения (.env) и настройка логирования."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from security import PIIScrubFilter

PROJECT_DIR = Path(__file__).resolve().parent
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ConfigError(RuntimeError):
    """Некорректная или неполная конфигурация — бот не должен стартовать."""


REQUIRED_ENV = ("VK_TOKEN", "VK_GROUP_ID", "GIGACHAT_CREDENTIALS")


def missing_required_env() -> list[str]:
    """Обязательные переменные, которых не хватает (для --check и диагностики)."""
    load_dotenv(PROJECT_DIR / ".env")
    return [name for name in REQUIRED_ENV if not _raw(name)]


@dataclass(frozen=True)
class Settings:
    """Все настройки в одном неизменяемом объекте."""

    # VK
    vk_token: str
    vk_group_id: int
    vk_api_version: str

    # GigaChat
    gigachat_credentials: str
    gigachat_scope: str
    gigachat_model: str
    gigachat_timeout: float
    gigachat_verify_ssl: bool

    # База знаний
    data_dir: Path

    # Яндекс Таблицы (опционально: без них заявки пишутся только в локальный бэкап)
    yandex_token: str
    yandex_spreadsheet_id: str
    yandex_sheet: str
    yandex_api_url: str

    # Прочее
    admin_vk_id: int | None
    leads_backup_path: Path
    log_file: Path
    log_level: str
    rate_limit_seconds: float
    context_ttl_seconds: float
    history_messages: int
    max_concurrent_tasks: int

    @property
    def yandex_enabled(self) -> bool:
        return bool(self.yandex_token and self.yandex_spreadsheet_id)

    @classmethod
    def from_env(cls, require_credentials: bool = True) -> "Settings":
        """Читает .env и переменные окружения. Бросает ConfigError с понятным текстом.

        require_credentials=False используется для локальных команд --check/--ask:
        они должны работать до заполнения токенов.
        """
        load_dotenv(PROJECT_DIR / ".env")

        missing = missing_required_env()
        if missing and require_credentials:
            raise ConfigError(
                "Не заданы обязательные переменные окружения: "
                + ", ".join(missing)
                + ". Скопируйте .env.example в .env и заполните значения."
            )

        data_dir = _path("DATA_DIR", PROJECT_DIR / "knowledge_base")
        if not data_dir.is_dir():
            raise ConfigError(
                f"Папка базы знаний не найдена: {data_dir} "
                "(проверьте переменную DATA_DIR)."
            )

        group_id = 0
        raw_group_id = _raw("VK_GROUP_ID")
        if raw_group_id:
            try:
                group_id = int(raw_group_id)
            except ValueError as exc:
                if require_credentials:
                    raise ConfigError(
                        "VK_GROUP_ID должен быть числом — id сообщества без минуса."
                    ) from exc

        return cls(
            vk_token=_raw("VK_TOKEN"),
            vk_group_id=group_id,
            vk_api_version=_raw("VK_API_VERSION", "5.199"),
            gigachat_credentials=_raw("GIGACHAT_CREDENTIALS"),
            gigachat_scope=_raw("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            gigachat_model=_raw("GIGACHAT_MODEL", "GigaChat-2"),
            gigachat_timeout=_float("GIGACHAT_TIMEOUT", 30.0),
            gigachat_verify_ssl=_bool("GIGACHAT_VERIFY_SSL", False),
            data_dir=data_dir,
            yandex_token=_raw("YANDEX_TABLES_TOKEN"),
            yandex_spreadsheet_id=_raw("YANDEX_TABLES_ID"),
            yandex_sheet=_raw("YANDEX_TABLES_SHEET", "Leads"),
            yandex_api_url=_raw(
                "YANDEX_TABLES_API_URL",
                "https://api.sheets.yandex.net/v1/spreadsheets/"
                "{spreadsheet_id}/values/{sheet}:append",
            ),
            admin_vk_id=_int_or_none("ADMIN_VK_ID"),
            leads_backup_path=_path(
                "LEADS_BACKUP_PATH", PROJECT_DIR / "var" / "leads_backup.jsonl"
            ),
            log_file=_path("LOG_FILE", PROJECT_DIR / "logs" / "bot.log"),
            log_level=_raw("LOG_LEVEL", "INFO").upper(),
            rate_limit_seconds=_float("RATE_LIMIT_SECONDS", 60.0),
            context_ttl_seconds=_float("CONTEXT_TTL_SECONDS", 900.0),
            history_messages=_int("HISTORY_MESSAGES", 5),
            max_concurrent_tasks=_int("MAX_CONCURRENT_TASKS", 20),
        )


def _raw(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = _raw(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть целым числом, получено: {raw!r}") from exc


def _int_or_none(name: str) -> int | None:
    raw = _raw(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть целым числом, получено: {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = _raw(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должен быть числом, получено: {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "да", "on"}


def _path(name: str, default: Path) -> Path:
    raw = _raw(name)
    path = Path(raw).expanduser() if raw else default
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def setup_logging(settings: Settings) -> None:
    """Консоль + ротация файла. Все сообщения проходят через фильтр ПД."""
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(PIIScrubFilter())

    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(PIIScrubFilter())

    for handler in (console, file_handler):
        root.addHandler(handler)

    # Сторонние библиотеки не должны засорять лог
    for noisy in ("aiohttp.access", "aiohttp.client", "gigachat", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
