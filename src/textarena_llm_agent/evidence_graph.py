"""Evidence Graph — append-only SQLite store wiring decisions → critic reports → skills → tools.

Design goals (per 改进需求.pdf §2 "证据图"):
  * Every piece of derived knowledge (skill, tool, prompt patch) must point back
    via SUPPORTED_BY / DERIVED_FROM edges to immutable raw evidence
    (DecisionFrame, Memory, Phenomenon, CriticReport).
  * Edges carry a predicate from a closed vocabulary so downstream queries
    ("what supports skill X?", "which decisions contradict skill Y?") are
    cheap and uniform.
  * Read-side queries used by Critic Agent (`nodes_supporting`, `replay_targets`,
    `count_supporting`) avoid loading the entire graph.

The 13 node tables are intentionally thin — they store identifiers, status,
and a JSON ``attrs`` column. The authoritative content lives in JSONL files
(experiences.jsonl, decision_frames.jsonl, critic_reports.jsonl, ...) and is
referenced by id. SQLite is the index, not the source of truth.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --- Closed vocabularies ----------------------------------------------------

NODE_TABLES: tuple[str, ...] = (
    "episode",
    "decision_frame",
    "memory",
    "phenomenon",
    "critic_report",
    "skill",
    "skill_version",
    "tool",
    "tool_version",
    "synthesized_tool",
    "experiment",
    "evaluation_run",
    "policy_version",
    "prompt_patch",
)

EDGE_PREDICATES: frozenset[str] = frozenset({
    "CONTAINS",       # episode CONTAINS decision_frame
    "PRODUCED",       # decision_frame PRODUCED memory
    "SUPPORTS",       # decision_frame SUPPORTS skill_version
    "CONTRADICTS",    # decision_frame CONTRADICTS skill_version
    "SUMMARIZED_AS",  # episode SUMMARIZED_AS critic_report
    "DERIVED_FROM",   # skill_version DERIVED_FROM phenomenon
    "SUPPORTED_BY",   # skill_version SUPPORTED_BY memory
    "VALIDATED_BY",   # tool_version VALIDATED_BY experiment
    "REQUESTS",       # critic_report REQUESTS synthesized_tool
})


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episode (
    id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    env_id TEXT,
    seed INTEGER,
    outcome TEXT,
    policy_version TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_episode_game ON episode(game_id);
CREATE INDEX IF NOT EXISTS idx_episode_policy ON episode(policy_version);

CREATE TABLE IF NOT EXISTS decision_frame (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    turn INTEGER,
    step INTEGER,
    candidate_id TEXT,
    action_text TEXT,
    policy_version TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_df_episode ON decision_frame(episode_id);
CREATE INDEX IF NOT EXISTS idx_df_state ON decision_frame(state_hash);
CREATE INDEX IF NOT EXISTS idx_df_game ON decision_frame(game_id);

CREATE TABLE IF NOT EXISTS memory (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,         -- experience | reflection | insight | do_not_learn
    game_id TEXT,
    player INTEGER,
    do_not_learn INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_game ON memory(game_id);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);

CREATE TABLE IF NOT EXISTS phenomenon (
    id TEXT PRIMARY KEY,
    title TEXT,
    game_id TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS critic_report (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    game_id TEXT,
    policy_version TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_critic_episode ON critic_report(episode_id);

CREATE TABLE IF NOT EXISTS skill (
    id TEXT PRIMARY KEY,
    name TEXT,
    game_id TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_skill_game ON skill(game_id);

CREATE TABLE IF NOT EXISTS skill_version (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,       -- proposed|candidate|validated|active|deprecated|rejected
    policy_version TEXT,
    replay_score REAL,
    ab_score REAL,
    created_by TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT,
    UNIQUE(skill_id, version)
);
CREATE INDEX IF NOT EXISTS idx_sv_skill ON skill_version(skill_id);
CREATE INDEX IF NOT EXISTS idx_sv_status ON skill_version(status);

CREATE TABLE IF NOT EXISTS tool (
    id TEXT PRIMARY KEY,
    name TEXT,
    game_id TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS tool_version (
    id TEXT PRIMARY KEY,
    tool_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,       -- tool_need|tool_spec|candidate_tool|validated_tool|active_tool|demoted|disabled
    policy_version TEXT,
    replay_score REAL,
    ab_score REAL,
    unit_tests_passed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    attrs_json TEXT,
    UNIQUE(tool_id, version)
);
CREATE INDEX IF NOT EXISTS idx_tv_tool ON tool_version(tool_id);
CREATE INDEX IF NOT EXISTS idx_tv_status ON tool_version(status);

CREATE TABLE IF NOT EXISTS synthesized_tool (
    id TEXT PRIMARY KEY,
    tool_id TEXT,
    name TEXT,
    status TEXT NOT NULL,        -- tool_need|tool_spec|candidate_tool|validated_tool|active_tool|demoted|disabled
    version INTEGER DEFAULT 0,
    game_id TEXT,
    policy_version TEXT,
    replay_score REAL DEFAULT 0.0,
    ab_score REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_st_game ON synthesized_tool(game_id);
CREATE INDEX IF NOT EXISTS idx_st_status ON synthesized_tool(status);

CREATE TABLE IF NOT EXISTS experiment (
    id TEXT PRIMARY KEY,
    hypothesis TEXT,
    control TEXT,
    treatment TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_run (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    policy_version TEXT,
    metric_name TEXT,
    metric_value REAL,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS policy_version (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent TEXT,
    skill_set_hash TEXT,
    tool_set_hash TEXT,
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS prompt_patch (
    id TEXT PRIMARY KEY,
    game_id TEXT,
    layer TEXT,                  -- static_prefix|game_static|policy_static|user_dynamic
    created_at TEXT NOT NULL,
    attrs_json TEXT
);

CREATE TABLE IF NOT EXISTS evidence_edges (
    src_type TEXT NOT NULL,
    src_id TEXT NOT NULL,
    edge TEXT NOT NULL,
    dst_type TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    attrs_json TEXT,
    PRIMARY KEY (src_type, src_id, edge, dst_type, dst_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON evidence_edges(dst_type, dst_id, edge);
CREATE INDEX IF NOT EXISTS idx_edge_src ON evidence_edges(src_type, src_id, edge);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EdgeRow:
    src_type: str
    src_id: str
    edge: str
    dst_type: str
    dst_id: str
    created_at: str = ""
    attrs: dict[str, Any] | None = None


class EvidenceGraph:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # --- node API -----------------------------------------------------------

    def add_node(self, node_type: str, node_id: str, **fields: Any) -> str:
        if node_type not in NODE_TABLES:
            raise ValueError(f"Unknown node type: {node_type}")
        attrs = fields.pop("attrs", None)
        created_at = fields.pop("created_at", None) or _now()
        columns = ["id", "created_at"]
        values: list[Any] = [node_id, created_at]
        for k, v in fields.items():
            columns.append(k)
            values.append(v)
        columns.append("attrs_json")
        values.append(json.dumps(attrs or {}, ensure_ascii=False, default=str))
        placeholders = ",".join("?" for _ in columns)
        col_list = ",".join(columns)
        # INSERT OR REPLACE to make ingestion idempotent.
        sql = f"INSERT OR REPLACE INTO {node_type} ({col_list}) VALUES ({placeholders})"
        self._conn.execute(sql, values)
        return node_id

    def get_node(self, node_type: str, node_id: str) -> dict[str, Any] | None:
        if node_type not in NODE_TABLES:
            raise ValueError(f"Unknown node type: {node_type}")
        cur = self._conn.execute(f"SELECT * FROM {node_type} WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        out = dict(zip(cols, row))
        if "attrs_json" in out and out["attrs_json"]:
            try:
                out["attrs"] = json.loads(out["attrs_json"])
            except Exception:
                out["attrs"] = {}
        return out

    def count_nodes(self, node_type: str) -> int:
        if node_type not in NODE_TABLES:
            raise ValueError(f"Unknown node type: {node_type}")
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {node_type}")
        return int(cur.fetchone()[0])

    # --- edge API -----------------------------------------------------------

    def add_edge(self, src_type: str, src_id: str, edge: str, dst_type: str, dst_id: str,
                 *, attrs: dict[str, Any] | None = None) -> None:
        if edge not in EDGE_PREDICATES:
            raise ValueError(f"Unknown edge predicate: {edge}")
        self._conn.execute(
            "INSERT OR IGNORE INTO evidence_edges (src_type, src_id, edge, dst_type, dst_id, created_at, attrs_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (src_type, src_id, edge, dst_type, dst_id, _now(), json.dumps(attrs or {}, ensure_ascii=False, default=str)),
        )

    def add_edges(self, rows: Iterable[EdgeRow]) -> int:
        n = 0
        for r in rows:
            self.add_edge(r.src_type, r.src_id, r.edge, r.dst_type, r.dst_id, attrs=r.attrs)
            n += 1
        return n

    # --- read queries -------------------------------------------------------

    def nodes_supporting(self, dst_type: str, dst_id: str, *, edge: str = "SUPPORTS") -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT src_type, src_id, attrs_json FROM evidence_edges WHERE dst_type = ? AND dst_id = ? AND edge = ?",
            (dst_type, dst_id, edge),
        )
        return [{"src_type": r[0], "src_id": r[1], "attrs": json.loads(r[2] or "{}")} for r in cur.fetchall()]

    def nodes_produced_by(self, src_type: str, src_id: str, *, edge: str = "PRODUCED") -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT dst_type, dst_id, attrs_json FROM evidence_edges WHERE src_type = ? AND src_id = ? AND edge = ?",
            (src_type, src_id, edge),
        )
        return [{"dst_type": r[0], "dst_id": r[1], "attrs": json.loads(r[2] or "{}")} for r in cur.fetchall()]

    def count_supporting(self, dst_type: str, dst_id: str, *, edge: str = "SUPPORTS") -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM evidence_edges WHERE dst_type = ? AND dst_id = ? AND edge = ?",
            (dst_type, dst_id, edge),
        )
        return int(cur.fetchone()[0])

    def replay_targets(self, game_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent DecisionFrame rows useful for replay-eval."""
        cur = self._conn.execute(
            "SELECT id, episode_id, state_hash, candidate_id, action_text, policy_version, attrs_json, created_at "
            "FROM decision_frame WHERE game_id = ? ORDER BY created_at DESC LIMIT ?",
            (game_id, int(limit)),
        )
        cols = [d[0] for d in cur.description]
        rows: list[dict[str, Any]] = []
        for r in cur.fetchall():
            obj = dict(zip(cols, r))
            try:
                obj["attrs"] = json.loads(obj.get("attrs_json") or "{}")
            except Exception:
                obj["attrs"] = {}
            rows.append(obj)
        return rows

    def episodes_for_policy(self, policy_version: str, *, limit: int = 100) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, game_id, outcome, created_at FROM episode WHERE policy_version = ? ORDER BY created_at DESC LIMIT ?",
            (policy_version, int(limit)),
        )
        return [{"id": r[0], "game_id": r[1], "outcome": r[2], "created_at": r[3]} for r in cur.fetchall()]

    def skill_versions(self, *, game_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT sv.id, sv.skill_id, sv.version, sv.status, sv.policy_version, sv.replay_score, sv.ab_score, sv.created_by, sv.created_at, sv.attrs_json, s.name, s.game_id " \
            "FROM skill_version sv JOIN skill s ON s.id = sv.skill_id"
        clauses: list[str] = []
        params: list[Any] = []
        if game_id is not None:
            clauses.append("s.game_id = ?")
            params.append(game_id)
        if status is not None:
            clauses.append("sv.status = ?")
            params.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY sv.created_at DESC"
        cur = self._conn.execute(q, params)
        cols = [d[0] for d in cur.description]
        rows: list[dict[str, Any]] = []
        for r in cur.fetchall():
            obj = dict(zip(cols, r))
            try:
                obj["attrs"] = json.loads(obj.get("attrs_json") or "{}")
            except Exception:
                obj["attrs"] = {}
            rows.append(obj)
        return rows

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Escape hatch for the Critic Agent's `query_evidence_graph` tool.

        Only SELECT statements are allowed; everything else raises ValueError.
        """
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries allowed via query()")
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # --- ingestion helpers --------------------------------------------------

    def ingest_decision_frame(self, frame: dict[str, Any]) -> str:
        fid = str(frame.get("id"))
        episode_id = str(frame.get("episode_id") or "")
        game_id = str(frame.get("game_id") or "")
        self.add_node(
            "decision_frame", fid,
            episode_id=episode_id, game_id=game_id, state_hash=str(frame.get("state_hash") or ""),
            turn=int(frame.get("turn") or 0), step=int(frame.get("step") or 0),
            candidate_id=str(frame.get("candidate_id") or ""), action_text=str(frame.get("action_text") or ""),
            policy_version=str(frame.get("policy_version") or ""),
            attrs=frame,
        )
        if episode_id:
            # ensure episode row exists so the FK-style link from decision_frame.episode_id resolves
            if self.get_node("episode", episode_id) is None:
                self.add_node("episode", episode_id, game_id=game_id, env_id="", seed=None, outcome="", policy_version=str(frame.get("policy_version") or ""), attrs={})
            self.add_edge("episode", episode_id, "CONTAINS", "decision_frame", fid)
        return fid

    def ingest_episode_trace(self, ep: dict[str, Any]) -> str:
        eid = str(ep.get("episode_id") or ep.get("id"))
        self.add_node(
            "episode", eid,
            game_id=str(ep.get("game_id") or ""), env_id=str(ep.get("env_id") or ""),
            seed=ep.get("seed"), outcome=str(ep.get("outcome") or ""),
            policy_version=str(ep.get("policy_version") or ""),
            attrs=ep,
        )
        return eid

    def ingest_critic_report(self, report: dict[str, Any]) -> str:
        rid = str(report.get("id"))
        episode_id = str(report.get("episode_id") or "")
        self.add_node(
            "critic_report", rid,
            episode_id=episode_id, game_id=str(report.get("game_id") or ""),
            policy_version=str(report.get("policy_version") or ""),
            attrs=report,
        )
        if episode_id and self.get_node("episode", episode_id) is not None:
            self.add_edge("episode", episode_id, "SUMMARIZED_AS", "critic_report", rid)
        return rid

    # --- bootstrap from legacy JSONL ---------------------------------------

    def bootstrap_from_jsonl(self, memory_dir: str | Path) -> dict[str, int]:
        """One-shot import of existing experiences.jsonl / reflections.jsonl / skills.jsonl.

        Idempotent: re-running on the same directory only re-inserts existing
        nodes (INSERT OR REPLACE). Counts of newly seen ids are returned.
        """
        root = Path(memory_dir)
        counts = {"experiences": 0, "reflections": 0, "skills": 0}
        for path, kind in [
            (root / "experiences.jsonl", "experience"),
            (root / "reflections.jsonl", "reflection"),
        ]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                mid = str(obj.get("id") or "")
                if not mid:
                    continue
                self.add_node(
                    "memory", mid,
                    kind=kind, game_id=str(obj.get("game_id") or ""),
                    player=int(obj.get("player") or 0) if obj.get("player") is not None else None,
                    do_not_learn=0, attrs=obj,
                )
                counts[f"{kind}s"] = counts.get(f"{kind}s", 0) + 1
        # legacy skills: status defaults to "active" with created_by="legacy".
        skills_path = root / "skills.jsonl"
        if skills_path.exists():
            for line in skills_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = str(obj.get("id") or "")
                if not sid:
                    continue
                self.add_node(
                    "skill", sid,
                    name=str(obj.get("name") or sid)[:120], game_id=str(obj.get("game_id") or ""),
                    attrs=obj,
                )
                self.add_node(
                    "skill_version", f"{sid}@v1",
                    skill_id=sid, version=1, status="active", policy_version="v0",
                    replay_score=None, ab_score=None, created_by="legacy",
                    attrs={"evidence_count_low": True, "source": "legacy_jsonl"},
                )
                counts["skills"] += 1
        return counts


__all__ = ["EvidenceGraph", "EdgeRow", "NODE_TABLES", "EDGE_PREDICATES"]
