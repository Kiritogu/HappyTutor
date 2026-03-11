# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Iterable


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunk_id(kb_name: str, provider: str, source_doc: str, text: str, idx: int) -> str:
    digest = hashlib.sha1(f"{kb_name}|{provider}|{source_doc}|{idx}|{text}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:20]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_entities(text: str, max_entities: int = 15) -> list[str]:
    # Lightweight, deterministic fallback extractor for first release.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{3,}", text or "")
    freq: dict[str, int] = {}
    for token in tokens:
        t = token.lower()
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _count in ranked[:max_entities]]


def pairwise_relations(entities: Iterable[str]) -> list[tuple[str, str]]:
    arr = list(dict.fromkeys(entities))
    pairs: list[tuple[str, str]] = []
    for i in range(len(arr) - 1):
        pairs.append((arr[i], arr[i + 1]))
    return pairs

