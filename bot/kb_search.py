"""Поиск по локальной базе знаний.

Принципы:
* никаких внешних индексов (Elasticsearch, векторные БД) — только файлы на диске;
* БЗ читается один раз при старте (перезапуск = переиндексация);
* эвристика: совпадение ключевых слов (с учётом словоформ) + длина фрагмента
  + близость к началу документа;
* CSV обязан иметь строку заголовков (первая строка).

Где улучшать в будущем:
* `KnowledgeBase._match` — точка подключения синонимов/эмбеддингов;
* `KnowledgeBase.load` — сюда можно добавить горячую перезагрузку по mtime,
  не меняя остальной код.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: Final = frozenset({".md", ".txt", ".csv"})

_MAX_CHUNK_CHARS = 900
_MIN_CHUNK_CHARS = 25
_MAX_CONTEXT_CHARS = 3500
# Сумма idf, после которой релевантность считается максимальной.
_SATURATION_IDF = 2.5
# Минимальная сумма idf совпавших слов: отсекает случайные совпадения.
_MIN_MATCHED_IDF = 1.0
# Бонус за точное вхождение фразы запроса (важно для строк FAQ).
_PHRASE_BONUS = 0.12

_TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Служебные слова, которые не несут смысла при поиске.
_STOPWORDS: Final = frozenset(
    """
    а без более бы был была были было быть в вам вас ведь весь во вот все всего
    всех вы где да даже для до его ее если есть еще же за здесь и из или им их к
    как какая какие какой когда кто ли либо мне может можно мой мы на надо над
    нам нас наш не него нее нет ни них но ну о об однако он она они оно от очень
    по под при про с сам свой себе себя так также такой там те тем теперь то того
    тоже тот тут ты у уж уже хотя чего чей чем через что чтоб чтобы чуть эта эти
    это этого этой этом этот я
    """.split()
)

# Окончания для очень простого стеммера (длинные — первыми).
_ENDINGS: Final = tuple(
    sorted(
        {
            "иями", "ями", "ами", "ов", "ев", "ий", "ый", "ой", "ая", "яя",
            "ое", "ее", "ые", "ие", "ую", "юю", "ах", "ях", "ом", "ем", "ам",
            "ям", "тся", "ться", "ют", "ут", "ат", "ят", "ил", "ила", "или",
            "ло", "ла", "ли", "ет", "ешь", "ете", "ть", "ти", "у", "ю", "е",
            "о", "ы", "и", "а", "я", "ь", "й",
        },
        key=len,
        reverse=True,
    )
)


def _stem(token: str) -> str:
    """Очень простой стеммер: срезает наиболее частые русские окончания."""
    if len(token) <= 4:
        return token
    for ending in _ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            return token[: -len(ending)]
    return token


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def _stems(text: str) -> list[str]:
    return [_stem(token) for token in _tokens(text)]


def _query_stems(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in _tokens(text):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        stem = _stem(token)
        if stem in _STOPWORDS or stem in seen:
            continue
        seen.add(stem)
        result.append(stem)
    return result


def _phrase_key(text: str) -> str:
    """Ключ для проверки точной фразы: слова без служебных, с границами."""
    tokens = [
        token
        for token in _tokens(text)
        if len(token) >= 3 and token not in _STOPWORDS
    ]
    if not tokens:
        return ""
    return f" {' '.join(tokens)} "


@dataclass(frozen=True)
class KnowledgeChunk:
    """Один смысловой фрагмент БЗ."""

    source: str      # путь относительно DATA_DIR, например "Услуги/чат-боты.md"
    topic: str       # иерархия папок, например "Услуги"
    title: str       # имя файла без расширения
    text: str
    position: float  # 0.0 — начало документа, 1.0 — конец


@dataclass(frozen=True)
class SearchHit:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class KnowledgeStats:
    files: int
    chunks: int
    topics: tuple[str, ...]


class KnowledgeBase:
    """In-memory индекс, который строится при запуске бота."""

    def __init__(self, chunks: Sequence[KnowledgeChunk], files: Sequence[str]) -> None:
        self._chunks: tuple[KnowledgeChunk, ...] = tuple(chunks)
        self._files: tuple[str, ...] = tuple(files)
        # stem -> {chunk_id: вес}, где 1.0 — совпадение в тексте, 0.5 — только в теме/заголовке
        self._postings: dict[str, dict[int, float]] = defaultdict(dict)
        self._sequences: tuple[str, ...] = tuple(
            _phrase_key(chunk.text) for chunk in self._chunks
        )
        self._build_index()

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, data_dir: Path) -> "KnowledgeBase":
        """Читает все поддерживаемые файлы из папки (рекурсивно)."""
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Папка базы знаний не найдена: {data_dir}")

        chunks: list[KnowledgeChunk] = []
        files: list[str] = []

        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(data_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue

            source = relative.as_posix()
            topic = " / ".join(relative.parts[:-1]) or "Общее"
            try:
                raw = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as exc:
                logger.error("Не удалось прочитать файл БЗ %s: %s", source, exc)
                continue

            texts = _chunk_csv(raw) if suffix == ".csv" else _chunk_text(raw)
            if not texts:
                logger.warning("Файл БЗ пуст или слишком короткий: %s", source)
                continue

            files.append(source)
            total = len(texts)
            for index, text in enumerate(texts):
                # Строки CSV — самостоятельные записи, а не части документа:
                # позиция внутри файла для них не имеет смысла.
                position = 0.0 if suffix == ".csv" else (index + 0.5) / total
                chunks.append(
                    KnowledgeChunk(
                        source=source,
                        topic=topic,
                        title=path.stem,
                        text=text,
                        position=position,
                    )
                )

        knowledge_base = cls(chunks, files)
        logger.info(
            "БЗ загружена: файлов=%d, фрагментов=%d, тем=%d",
            len(knowledge_base._files),
            len(knowledge_base._chunks),
            len(knowledge_base.topics()),
        )
        if not chunks:
            logger.warning(
                "База знаний пуста. Бот будет отвечать дисклеймером. "
                "Проверьте DATA_DIR=%s", data_dir,
            )
        return knowledge_base

    def _build_index(self) -> None:
        for chunk_id, chunk in enumerate(self._chunks):
            body_stems = set(_stems(chunk.text))
            meta_stems = set(_stems(f"{chunk.topic} {chunk.title}")) - body_stems

            for stem in body_stems:
                self._postings[stem][chunk_id] = 1.0
            for stem in meta_stems:
                self._postings[stem][chunk_id] = 0.5

    # ------------------------------------------------------------------ #
    # Поиск
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        min_score: float = 0.30,
    ) -> list[SearchHit]:
        """Возвращает наиболее релевантные фрагменты (возможно, пустой список)."""
        query_stems = _query_stems(query)
        if not query_stems or not self._chunks:
            return []

        # Точная фраза из 2+ значимых слов: строки FAQ должны обыгрывать
        # общие документы, где слова встречаются по отдельности.
        query_phrase = _phrase_key(query)
        use_phrase = len(query_phrase.split()) >= 2

        matched_idf: dict[int, float] = defaultdict(float)
        matched_terms: dict[int, int] = defaultdict(int)

        for query_stem in query_stems:
            for chunk_id, weight, idf in self._match(query_stem):
                matched_idf[chunk_id] += idf * weight
                matched_terms[chunk_id] += 1

        total = len(self._chunks)
        scored: list[SearchHit] = []
        for chunk_id, idf_sum in matched_idf.items():
            if matched_terms[chunk_id] < 1 or idf_sum < _MIN_MATCHED_IDF:
                continue

            chunk = self._chunks[chunk_id]
            saturation = min(1.0, idf_sum / _SATURATION_IDF)
            length_score = max(
                0.0,
                1.0 - abs(len(chunk.text) - 450) / 600,
            )
            position_score = 1.0 - min(1.0, max(0.0, chunk.position))
            # Сколько значимых слов запроса покрыто фрагментом: защищает от
            # ответа по одному общему слову на многословный вопрос.
            coverage = min(1.0, matched_terms[chunk_id] / len(query_stems))

            # Веса подобраны эмпирически.
            score = (
                0.55 * saturation
                + 0.20 * length_score
                + 0.15 * position_score
                + 0.10 * coverage
            )
            if use_phrase and query_phrase in self._sequences[chunk_id]:
                score += _PHRASE_BONUS
            score = min(1.0, score)
            if score >= min_score:
                scored.append(SearchHit(chunk=chunk, score=round(score, 4)))

        if not scored:
            return []

        scored.sort(key=lambda hit: hit.score, reverse=True)
        best = scored[0].score
        threshold = max(min_score, best * 0.70)

        # Разнообразие: не более двух фрагментов из одного файла.
        hits: list[SearchHit] = []
        per_source: dict[str, int] = defaultdict(int)
        for hit in scored:
            if hit.score < threshold:
                break
            if per_source[hit.chunk.source] >= 2:
                continue
            per_source[hit.chunk.source] += 1
            hits.append(hit)
            if len(hits) >= limit:
                break

        logger.debug(
            "Поиск: запрос=%r, кандидатов=%d, отобрано=%d, лучший=%.3f",
            query[:80], len(scored), len(hits), best,
        )
        return hits

    def _match(self, query_stem: str) -> Iterable[tuple[int, float, float]]:
        """Ищет чанки по стему с учётом простых словоформ (префиксное совпадение)."""
        for index_stem, postings in self._postings.items():
            if index_stem == query_stem:
                factor = 1.0
            elif (
                len(index_stem) >= 4
                and len(query_stem) >= 4
                and (
                    index_stem.startswith(query_stem)
                    or query_stem.startswith(index_stem)
                )
            ):
                factor = 0.8
            else:
                continue

            idf = 1.0 + math.log(
                1.0 + (len(self._chunks) - len(postings) + 0.5)
                / (len(postings) + 0.5)
            )
            for chunk_id, stored_weight in postings.items():
                yield chunk_id, factor * stored_weight, idf

    # ------------------------------------------------------------------ #
    # Вспомогательное
    # ------------------------------------------------------------------ #
    def topics(self) -> tuple[str, ...]:
        return tuple(sorted({chunk.topic for chunk in self._chunks}))

    @property
    def stats(self) -> KnowledgeStats:
        return KnowledgeStats(
            files=len(self._files),
            chunks=len(self._chunks),
            topics=self.topics(),
        )

    def format_context(self, hits: Sequence[SearchHit]) -> str:
        """Собирает фрагменты для промпта, укладываясь в лимит символов."""
        blocks: list[str] = []
        used = 0
        for index, hit in enumerate(hits, start=1):
            block = (
                f"[Фрагмент {index}] Источник: {hit.chunk.source}\n"
                f"{hit.chunk.text.strip()}"
            )
            if used + len(block) > _MAX_CONTEXT_CHARS:
                if not blocks:
                    blocks.append(block[:_MAX_CONTEXT_CHARS])
                break
            blocks.append(block)
            used += len(block) + len("\n\n---\n\n")
        return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Разбиение файлов на фрагменты
# --------------------------------------------------------------------------- #
def _split_long(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Режет слишком длинный абзац по предложениям, при необходимости — жёстко."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_RE.split(text):
        while len(sentence) > max_chars:
            if buffer:
                parts.append(buffer)
                buffer = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]

        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                parts.append(buffer)
            buffer = sentence

    if buffer:
        parts.append(buffer)
    return parts


def _chunk_text(raw: str) -> list[str]:
    """Текст/markdown: абзацы, склеенные до ~900 символов."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in raw.splitlines() if line.strip()]

    merged: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        for piece in _split_long(paragraph):
            if not buffer:
                buffer = piece
            elif len(buffer) + len(piece) + 2 <= _MAX_CHUNK_CHARS:
                buffer = f"{buffer}\n{piece}"
            else:
                merged.append(buffer)
                buffer = piece
    if buffer:
        merged.append(buffer)

    return [chunk for chunk in merged if len(chunk) >= _MIN_CHUNK_CHARS]


def _chunk_csv(raw: str) -> list[str]:
    """CSV: первая строка — заголовок, каждая строка — отдельный фрагмент."""
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw), dialect)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return []

    header = [
        (cell.strip() or f"col{index + 1}") for index, cell in enumerate(rows[0])
    ]

    chunks: list[str] = []
    for row in rows[1:]:
        parts = [
            f"{header[index]}: {value.strip()}"
            for index, value in enumerate(row)
            if value.strip() and index < len(header)
        ]
        if parts:
            chunks.append("; ".join(parts))
    return chunks
