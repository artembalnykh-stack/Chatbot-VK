"""Точка входа: настройки, Long Poll, главный цикл, graceful shutdown.

Запуск:
    python main.py              # бот
    python main.py --check      # проверка конфигурации и базы знаний без сети
    python main.py --ask "..."  # локальный тест поиска/ответа без VK
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict, Optional, Set

import aiohttp

from ai_adapter import AIAdapter, GigaChatError
from kb_search import KnowledgeBase
from lead_collector import LeadCollector, LeadStorage
from utils import ConfigError, Settings, missing_required_env, setup_logging
from vk_handler import LongPollListener, MessageProcessor, VKClient

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 60  # секунд


async def _cleanup_loop(
    ai: AIAdapter,
    leads: LeadCollector,
    processor: MessageProcessor,
) -> None:
    """Периодически чистит контекст (15 минут неактивности) и лимиты."""
    while True:
        try:
            await asyncio.sleep(_CLEANUP_INTERVAL)
            ai.purge_stale()
            leads.purge_stale()
            processor.purge_stale()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Ошибка в задаче очистки контекста")


async def run_bot(settings: Settings) -> None:
    timeout = aiohttp.ClientTimeout(total=60, connect=15)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        vk = VKClient(settings.vk_token, settings.vk_api_version, session)
        knowledge_base = KnowledgeBase.load(settings.data_dir)
        ai = AIAdapter(settings)

        async def notify_admin(text: str) -> None:
            if settings.admin_vk_id is None:
                return
            await vk.send_message(settings.admin_vk_id, text)

        storage = LeadStorage(settings, session)
        leads = LeadCollector(settings, storage, notifier=notify_admin)
        processor = MessageProcessor(settings, vk, knowledge_base, ai, leads)

        logger.info(
            "Старт: сообщество=%s, модель=%s, БЗ=%d файлов/%d фрагментов, "
            "Яндекс Таблицы=%s, админ=%s",
            settings.vk_group_id,
            settings.gigachat_model,
            knowledge_base.stats.files,
            knowledge_base.stats.chunks,
            "вкл" if settings.yandex_enabled else "выкл",
            "задан" if settings.admin_vk_id else "нет",
        )

        listener = LongPollListener(vk, settings.vk_group_id)
        semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        tasks: Set[asyncio.Task[None]] = set()
        cleanup = asyncio.create_task(_cleanup_loop(ai, leads, processor))

        try:
            async for event in listener.events():
                task = asyncio.create_task(
                    _process_event(processor, event, semaphore)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        finally:
            cleanup.cancel()
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(cleanup, *tasks, return_exceptions=True)
                await ai.close()
            except asyncio.CancelledError:
                # Остановка по Ctrl+C: гасим задачи и выходим без трейсбека.
                pass
            logger.info("Бот остановлен")


async def _process_event(
    processor: MessageProcessor,
    event: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        await processor.handle_event(event)


async def run_ask(settings: Settings, knowledge_base: KnowledgeBase, question: str) -> None:
    """Локальная проверка поиска и GigaChat без запуска VK."""
    hits = knowledge_base.search(question)
    if not hits:
        print("Поиск: релевантных фрагментов не найдено (бот ответил бы дисклеймером).")
        return

    print(f"Найдено фрагментов: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        preview = hit.chunk.text.replace("\n", " ")[:160]
        print(f"  {index}. [{hit.score:.2f}] {hit.chunk.source}: {preview}…")

    if not settings.gigachat_credentials:
        print("\nGIGACHAT_CREDENTIALS не задан — запрос к ИИ пропущен.")
        return

    ai = AIAdapter(settings)
    try:
        answer = await ai.answer(0, question, knowledge_base.format_context(hits))
        print("\nОтвет ИИ:\n" + answer)
    except GigaChatError as exc:
        print(f"\nGigaChat недоступен: {exc}")
    finally:
        await ai.close()


def _print_check(settings: Settings, knowledge_base: KnowledgeBase) -> None:
    missing = missing_required_env()
    print("Проверка конфигурации")
    print("-" * 60)
    print(f"VK token:             {'задан' if settings.vk_token else 'НЕ ЗАДАН'}")
    print(f"VK group id:          {settings.vk_group_id or 'НЕ ЗАДАН'}")
    print(f"GigaChat credentials: {'задан' if settings.gigachat_credentials else 'НЕ ЗАДАН'}")
    print(f"GigaChat модель:      {settings.gigachat_model} (scope={settings.gigachat_scope})")
    print(f"База знаний:          {settings.data_dir}")
    print(f"  файлов:             {knowledge_base.stats.files}")
    print(f"  фрагментов:         {knowledge_base.stats.chunks}")
    print(f"  темы:               {', '.join(knowledge_base.stats.topics) or '—'}")
    print(f"Яндекс Таблицы:       {'вкл' if settings.yandex_enabled else 'выкл'}")
    print(f"Админ VK id:          {settings.admin_vk_id or 'не задан'}")
    print(f"Лог-файл:             {settings.log_file}")
    print(f"Бэкап заявок:         {settings.leads_backup_path}")
    print("-" * 60)

    if missing:
        print("Не хватает переменных: " + ", ".join(missing))
    if not knowledge_base.stats.chunks:
        print("ВНИМАНИЕ: база знаний пуста — бот будет отвечать дисклеймером.")
    if not knowledge_base.stats.chunks or missing:
        print("Итог: есть предупреждения.")
    else:
        print("Итог: конфигурация выглядит корректно.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="VK-бот-консультант по базе знаний с GigaChat",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="проверить .env и базу знаний, не подключаясь к VK",
    )
    parser.add_argument(
        "--ask",
        metavar="ВОПРОС",
        help="локально прогнать вопрос через поиск и GigaChat (без VK)",
    )
    args = parser.parse_args(argv)

    local_only = args.check or bool(args.ask)

    try:
        settings = Settings.from_env(require_credentials=not local_only)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    setup_logging(settings)

    if local_only:
        try:
            knowledge_base = KnowledgeBase.load(settings.data_dir)
        except (OSError, ValueError) as exc:
            print(f"Ошибка загрузки базы знаний: {exc}", file=sys.stderr)
            return 1

        if args.check:
            _print_check(settings, knowledge_base)

        if args.ask:
            try:
                asyncio.run(run_ask(settings, knowledge_base, args.ask))
            except KeyboardInterrupt:
                pass
        return 0

    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
