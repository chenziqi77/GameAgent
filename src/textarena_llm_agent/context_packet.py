"""Per-decision context packet builder with principled budget allocation.

Replaces the ad-hoc truncation in the old ``_build_user_prompt``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DecisionContextPacket:
    """Per-decision prompt payload + optional four-layer cache metadata.

    The ``layer_hashes`` / ``layer_tokens`` fields are populated when the packet
    was built via ``PromptCompiler.compile``; legacy callsites that go through
    ``ContextBudgeter.build`` directly leave them empty. Cache observation in
    the tracer / evidence graph relies only on these dicts being optional.
    """
    system: str
    user: str
    sections: dict[str, str] = field(default_factory=dict)
    budget_used: int = 0
    layer_hashes: dict[str, str] = field(default_factory=dict)
    layer_tokens: dict[str, int] = field(default_factory=dict)
    policy_version: str = ""
    stable_prefix_tokens: int = 0


class ContextBudgeter:
    """Allocate a fixed character budget across decision-context sections."""

    def __init__(self, *, total_chars: int = 24000) -> None:
        self.total_chars = total_chars
        # budget allocation (sums to total minus headroom)
        self.budget = {
            "memory": 5000,
            "state": 6000,
            "candidates": 6000,
            "reflection": 1200,
            "patches": 1200,
        }

    def build(self, *, spec_system: str, state_text: str,
              candidates: list[dict[str, Any]], memory_excerpt: str,
              reflection: str, patches: list[str]) -> DecisionContextPacket:
        sections: dict[str, str] = {}
        sections["reflection"] = (reflection or "").strip()[: self.budget["reflection"]]
        sections["patches"] = "\n".join(f"- {p}" for p in (patches or []) if p)[: self.budget["patches"]]
        sections["memory"] = (memory_excerpt or "No relevant memory.").strip()[: self.budget["memory"]]

        sections["candidates"] = self._fit_candidates_json(candidates)

        # state: prioritize visible_state + recent log; trim if over budget
        state_section = state_text
        if len(state_section) > self.budget["state"]:
            # try to drop the verbose recent_log tail first
            try:
                obj = json.loads(state_text)
                if "recent_log" in obj:
                    obj["recent_log"] = obj["recent_log"][-6:]
                    state_section = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
            if len(state_section) > self.budget["state"]:
                state_section = state_section[: self.budget["state"]] + "\n...STATE_TRUNCATED_FOR_BUDGET..."
        sections["state"] = state_section

        user = (
            "TACTICAL_MEMORY:\n" + sections["memory"]
            + ("\n\nPROMPT_PATCHES:\n" + sections["patches"] if sections["patches"] else "")
            + ("\n\nREFLECTION:\n" + sections["reflection"] if sections["reflection"] else "")
            + "\n\nVISIBLE_STATE_JSON:\n" + sections["state"]
            + "\n\nCANDIDATE_ACTIONS_JSON:\n" + sections["candidates"]
            + "\n\nChoose exactly one candidate_id from CANDIDATE_ACTIONS_JSON. Return JSON only."
        )
        # hard cap
        if len(user) > self.total_chars:
            user = user[: self.total_chars] + "\n...PACKET_TRUNCATED..."
        return DecisionContextPacket(system=spec_system, user=user, sections=sections, budget_used=len(user))

    def _fit_candidates_json(self, candidates: list[dict[str, Any]]) -> str:
        """Return a valid JSON array for the strongest candidates within budget."""
        budget = self.budget["candidates"]
        payload = list(candidates)
        while payload:
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            if len(text) <= budget:
                return text
            payload = payload[:-1]
        compact = [_compact_candidate(c) for c in candidates]
        while compact:
            text = json.dumps(compact, ensure_ascii=False, indent=2, default=str)
            if len(text) <= budget:
                return text
            compact = compact[:-1]
        return "[]"


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "candidate_id": candidate.get("candidate_id"),
        "action_text": candidate.get("action_text"),
        "action_type": candidate.get("action_type"),
        "score": candidate.get("score"),
    }
    reasons = candidate.get("reasons")
    if isinstance(reasons, list) and reasons:
        keep["reasons"] = [str(x)[:120] for x in reasons[:2]]
    risks = candidate.get("risks")
    if isinstance(risks, list) and risks:
        keep["risks"] = [str(x)[:120] for x in risks[:2]]
    return {k: v for k, v in keep.items() if v not in (None, "", [], {})}
