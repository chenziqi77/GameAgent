"""Retrieval for the evolving memory.

Default: a dependency-free BM25 retriever over the union of experiences, skills,
reflections and rule chunks, filtered to the current (game_id, player, phase)
perspective — essential for self-play where one agent plays both sides.

Pattern: Generative Agents (Park et al. 2023) memory-stream retrieval combining
relevance x recency x importance.

Score = BM25_relevance + alpha * recency + beta * importance + gamma * phase_match.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


@dataclass(slots=True)
class ScoredItem:
    source: str
    id: str
    text: str
    score: float
    recency: float = 0.0
    importance: float = 0.0
    relevance: float = 0.0
    game_id: str = ""
    player: int | None = None
    tags: list[str] = field(default_factory=list)


class Retriever(Protocol):
    def retrieve(self, *, query: str, game_id: str, player: int | None = None,
                 phase: str | None = None, top_k: int = 8) -> list[ScoredItem]:
        ...


class BM25Retriever:
    """In-memory BM25 over a small in-house corpus. Re-indexes when source files change."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75,
                 alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.5,
                 indexer: "CorpusIndexer | None" = None) -> None:
        self.k1 = k1
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.indexer = indexer or CorpusIndexer()

    def retrieve(self, *, query: str, game_id: str, player: int | None = None,
                 phase: str | None = None, top_k: int = 8) -> list[ScoredItem]:
        docs = self.indexer.documents(game_id=game_id)
        if not docs:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            # No query signal: fall back to importance + recency only.
            scored = [(d, self.alpha * d.recency + self.beta * d.importance) for d in docs]
        else:
            scored = [(d, self._score(d, q_terms, player, phase)) for d in docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[ScoredItem] = []
        for doc, score in scored:
            if score <= 0 and not results:
                continue
            doc.score = float(score)
            results.append(doc)
            if len(results) >= top_k:
                break
        return results

    def _score(self, doc: ScoredItem, q_terms: list[str], player: int | None, phase: str | None) -> float:
        rel = self.indexer.bm25(doc, q_terms, k1=self.k1, b=self.b)
        # perspective hard-filter: if a doc is player-tagged and we have a player, require match;
        # allow untagged (game-general) docs through.
        perspective = 1.0
        if player is not None and doc.player is not None and doc.player != player:
            perspective = 0.0
        if doc.game_id and doc.game_id != "general" and perspective == 0.0:
            perspective = 0.0
        phase_match = self.gamma if (phase and any(phase in t for t in doc.tags)) else 0.0
        return (rel + self.alpha * doc.recency + self.beta * doc.importance + phase_match) * perspective


@dataclass(slots=True)
class _IndexedDoc:
    item: ScoredItem
    tokens: list[str]
    tf: dict[str, int]
    length: int


class CorpusIndexer:
    """Builds and caches an inverted index from the file-backed memory corpus."""

    def __init__(self) -> None:
        self._cache_key: tuple = ()
        self._docs: list[_IndexedDoc] = []
        self._idf: dict[str, float] = {}
        self._avg_len: float = 0.0

    def documents(self, *, game_id: str) -> list[ScoredItem]:
        return [d.item for d in self._docs if (not d.item.game_id or d.item.game_id == game_id or d.item.game_id == "general")]

    def bm25(self, doc: ScoredItem, q_terms: list[str], *, k1: float, b: float) -> float:
        idx = next((d for d in self._docs if d.item is doc), None)
        if idx is None:
            return 0.0
        score = 0.0
        denom_norm = (1 - b + b * (idx.length / self._avg_len)) if self._avg_len > 0 else 1.0
        for term in q_terms:
            f = idx.tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += idf * (f * (k1 + 1)) / (f + k1 * denom_norm)
        return score / max(1, len(q_terms))

    def index(self, sources: Iterable[ScoredItem]) -> None:
        docs: list[_IndexedDoc] = []
        for item in sources:
            toks = tokenize(item.text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            docs.append(_IndexedDoc(item=item, tokens=toks, tf=tf, length=len(toks)))
        self._docs = docs
        n = max(1, len(docs))
        self._avg_len = sum(d.length for d in docs) / n
        df: dict[str, int] = {}
        for d in docs:
            for term in d.tf:
                df[term] = df.get(term, 0) + 1
        self._idf = {term: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5)) for term, cnt in df.items()}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _recency_from(ts: str | None, *, half_life_days: float = 30.0) -> float:
    if not ts:
        return 0.0
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return math.exp(-age_days / half_life_days)
    except Exception:
        return 0.0


class EmbeddingRetriever:
    """Pluggable embedding retriever (default off). A caller supplies embed_fn."""

    def __init__(self, *, embed_fn: Callable[[str], "np.ndarray"], dim: int,
                 alpha: float = 0.3, beta: float = 0.4) -> None:
        self.embed_fn = embed_fn
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self._items: list[ScoredItem] = []
        self._vecs: list[np.ndarray] = []

    def add(self, item: ScoredItem, vec: "np.ndarray | None" = None) -> None:
        self._items.append(item)
        self._vecs.append(vec if vec is not None else self.embed_fn(item.text))

    def retrieve(self, *, query: str, game_id: str, player: int | None = None,
                 phase: str | None = None, top_k: int = 8) -> list[ScoredItem]:
        if not self._items:
            return []
        qv = self.embed_fn(query)
        mats = np.stack(self._vecs) if self._vecs else np.zeros((0, self.dim))
        rel = mats @ qv if mats.size else np.zeros(len(self._items))
        scored = []
        for item, r in zip(self._items, rel):
            if player is not None and item.player is not None and item.player != player:
                continue
            if item.game_id and item.game_id not in (game_id, "general"):
                continue
            s = float(r) + self.alpha * item.recency + self.beta * item.importance
            scored.append((item, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:top_k]]


def build_corpus_from_memory(memory_root: Path, *, game_id: str) -> list[ScoredItem]:
    """Read the file-backed memory corpus into a list of ScoredItems, limited to a game."""
    items: list[ScoredItem] = []
    experiences = memory_root / "experiences.jsonl"
    if experiences.exists():
        for line in experiences.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = __import__("json").loads(line)
            except Exception:
                continue
            gid = str(row.get("game_id") or "")
            if gid and gid != game_id:
                continue
            text = f"experience {row.get('id','')}: {row.get('action_text','')} | outcome={row.get('outcome')} reward={row.get('reward')} lesson={row.get('lesson','')} critique={row.get('critique','')}"
            items.append(ScoredItem(
                source="experiences", id=str(row.get("id") or ""), text=text[:1200],
                score=0.0, recency=_recency_from(row.get("created_at")),
                importance=_importance(row),
                game_id=gid, player=_int(row.get("player")),
                tags=[str(row.get("outcome") or ""), str(gid)],
            ))
    skills = memory_root / "skills.jsonl"
    if skills.exists():
        for line in skills.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = __import__("json").loads(line)
            except Exception:
                continue
            guid = (row.get("trigger") or "")
            if game_id not in guid and game_id not in (row.get("tags") or []):
                continue
            uses = int(row.get("uses") or 0)
            wins = int(row.get("wins") or 0)
            text = f"skill {row.get('name','')}: trigger={row.get('trigger','')} guidance={row.get('guidance','')} uses={uses} wins={wins}"
            items.append(ScoredItem(
                source="skills", id=str(row.get("id") or ""), text=text[:1200],
                score=0.0, recency=_recency_from(row.get("updated_at") or row.get("created_at")),
                importance=_skill_importance(row),
                game_id=str(row.get("trigger","").split(":")[0].replace("game","").strip() or game_id),
                player=None, tags=list(row.get("tags") or []),
            ))
    reflections = memory_root / "reflections.jsonl"
    if reflections.exists():
        for line in reflections.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = __import__("json").loads(line)
            except Exception:
                continue
            gid = str(row.get("game_id") or "")
            if gid and gid != game_id:
                continue
            text = f"reflection {row.get('id','')}: {row.get('text','')} lesson={row.get('actionable_lesson','')}"
            items.append(ScoredItem(
                source="reflections", id=str(row.get("id") or ""), text=text[:1200],
                score=0.0, recency=_recency_from(row.get("created_at")),
                importance=2.5, game_id=gid, player=None,
                tags=list(row.get("state_keys") or []),
            ))
    rules = memory_root / "rules.md"
    if rules.exists():
        for chunk in _split_markdown(rules.read_text(encoding="utf-8", errors="replace")):
            items.append(ScoredItem(
                source="rules", id="rules", text=chunk[:1200], score=0.0,
                recency=1.0, importance=2.0, game_id="general", player=None, tags=["general"],
            ))
    return items


def _int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _importance(row: dict) -> float:
    r = row.get("reward")
    if r is not None:
        try:
            return 2.0 + abs(float(r)) * 2.0
        except Exception:
            pass
    return 1.5


def _skill_importance(row: dict) -> float:
    uses = int(row.get("uses") or 0)
    return min(5.0, 1.5 + uses * 0.15)


def _split_markdown(text: str) -> list[str]:
    chunks = [c.strip() for c in re.split(r"\n(?=- |\#\#?)", text or "") if c.strip()]
    return chunks or ([text.strip()] if text.strip() else [])
