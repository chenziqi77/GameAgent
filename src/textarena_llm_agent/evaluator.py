from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .action_analyzer import CandidateAnalysis
from .llm import DecisionLLM, HeuristicLLM
from .memory import EvolvingMemory, Experience, state_key_from_text
from .prompts import EVALUATOR_SYSTEM_PROMPT


@dataclass(slots=True)
class EvaluationResult:
    score: float
    accept: bool
    suggested_candidate_id: str | None
    critique: str
    lesson: str = ""
    prompt_patch: str = ""
    raw: dict[str, Any] | None = None


class DecisionEvaluator:
    def __init__(self, *, llm: DecisionLLM | None = None, memory: EvolvingMemory | None = None, min_accept_score: float = 0.45) -> None:
        self.llm = llm or HeuristicLLM()
        self.memory = memory or EvolvingMemory()
        self.min_accept_score = min_accept_score

    def evaluate(self, *, state_text: str, candidates: list[CandidateAnalysis], decision: dict[str, Any], temperature: float = 0.0, max_tokens: int = 800) -> EvaluationResult:
        selected_id = str(decision.get("candidate_id") or "")
        by_id = {c.candidate_id: c for c in candidates}
        selected = by_id.get(selected_id)
        best = candidates[0] if candidates else None
        if isinstance(self.llm, HeuristicLLM):
            return self._heuristic_evaluate(selected, best)
        user = (
            "VISIBLE_STATE_JSON:\n" + state_text[:12000]
            + "\n\nCANDIDATE_ACTIONS_JSON:\n" + json.dumps([c.to_prompt_dict() for c in candidates], ensure_ascii=False, indent=2, default=str)
            + "\n\nPROPOSED_DECISION_JSON:\n" + json.dumps(decision, ensure_ascii=False, indent=2, default=str)
        )
        raw = self.llm.complete_json(system=EVALUATOR_SYSTEM_PROMPT, user=user, temperature=temperature, max_tokens=max_tokens)
        score = _float(raw.get("score"), 0.0)
        suggested = raw.get("suggested_candidate_id")
        suggested_id = str(suggested) if suggested not in (None, "", "null") else None
        if suggested_id and suggested_id not in by_id:
            suggested_id = best.candidate_id if best is not None else None
        accept = bool(raw.get("accept", score >= self.min_accept_score))
        if score < self.min_accept_score and suggested_id:
            accept = False
        result = EvaluationResult(
            score=score,
            accept=accept,
            suggested_candidate_id=suggested_id,
            critique=str(raw.get("critique") or ""),
            lesson=str(raw.get("lesson") or ""),
            prompt_patch=str(raw.get("prompt_patch") or ""),
            raw=raw,
        )
        if result.lesson:
            self.memory.add_rule(result.lesson, evidence="pre_action_evaluator", tags=["evaluator"])
        if result.prompt_patch:
            self.memory.add_prompt_patch(result.prompt_patch, evidence="pre_action_evaluator")
        return result

    def learn_from_transition(
        self,
        *,
        before_state_text: str,
        game_id: str,
        player: int,
        decision: dict[str, Any],
        evaluation: EvaluationResult | None = None,
        reward: float | None = None,
        outcome: str | None = None,
    ) -> str:
        exp = Experience(
            state_key=state_key_from_text(before_state_text),
            game_id=game_id,
            player=int(player),
            action_text=str(decision.get("action_text") or decision.get("action") or ""),
            evaluator_score=evaluation.score if evaluation else None,
            reward=reward,
            outcome=outcome,
            critique=evaluation.critique if evaluation else "",
            lesson=evaluation.lesson if evaluation else "",
            tags=[game_id, f"player_{player}"],
        )
        exp_id = self.memory.record_experience(exp)
        self.memory.record_retrieval(query=state_key_from_text(before_state_text), items=[{"source": "experience", "score": 1, "id": exp_id, "text": exp.action_text}])
        return exp_id

    def _heuristic_evaluate(self, selected: CandidateAnalysis | None, best: CandidateAnalysis | None) -> EvaluationResult:
        if selected is None:
            return EvaluationResult(0.0, False, best.candidate_id if best else None, "Selected candidate is not in the legal candidate set.", "Always select candidate_id from the current legal candidate list.")
        if best is None:
            return EvaluationResult(0.5, True, None, "No alternative candidates available.")
        gap = float(best.score) - float(selected.score)
        if gap > 4.0:
            return EvaluationResult(
                0.35,
                False,
                best.candidate_id,
                f"Chosen action is weaker than {best.candidate_id} by heuristic gap {gap:.1f}.",
                "When one legal candidate has a large tactical or payoff advantage, prefer it unless there is a concrete counter-reason.",
            )
        return EvaluationResult(max(0.45, min(0.95, 0.8 - gap * 0.05)), True, None, "Chosen action is consistent with ranked candidate analysis.")


def fallback_terminal_lesson(*, game_id: str, outcome: str | None, reward: float | None, critique: str = "") -> str:
    """Deterministic fallback lesson when an external critic is unavailable or silent.

    The rule is intentionally conservative: it only derives lessons from visible
    outcome signals and existing evaluator critique, so it can create auditable
    skill events without pretending to have a richer LLM analysis.
    """
    outcome_l = str(outcome or "").lower()
    critique_l = critique.lower()
    if "illegal" in critique_l or "not in the legal" in critique_l:
        return "Always choose an exact candidate_id from the current legal action list; never emit invented or stale actions."
    if game_id == "TicTacToe":
        if "loss" in outcome_l or (reward is not None and reward < 0):
            return "In TicTacToe, before creating a threat, first block any immediate opponent three-in-a-row threat and prefer center or fork-blocking moves."
        if "win" in outcome_l or (reward is not None and reward > 0):
            return "In TicTacToe, preserve winning patterns by taking center/corners early and converting immediate two-in-a-row threats."
    if game_id == "KuhnPoker":
        if "loss" in outcome_l or (reward is not None and reward < 0):
            return "In KuhnPoker, condition bets and calls on card strength and betting history; avoid predictable aggression with weak private cards."
        if "win" in outcome_l or (reward is not None and reward > 0):
            return "In KuhnPoker, mix value bets with selective bluffs so the opponent cannot exploit a fixed betting pattern."
    if game_id == "SimpleNegotiation":
        if "loss" in outcome_l or (reward is not None and reward < 0):
            return "In negotiation, avoid conceding high-value resources without reciprocal gain; compare each offer against own item values before accepting."
        if "win" in outcome_l or (reward is not None and reward > 0):
            return "In negotiation, trade away lower-value resources for higher-value returns while keeping offers acceptable enough to close a deal."
    if "loss" in outcome_l or (reward is not None and reward < 0):
        return "After a terminal loss, identify the opponent threat or payoff swing before choosing the next action in similar states."
    if "win" in outcome_l or (reward is not None and reward > 0):
        return "Preserve strategies that improved terminal reward, but re-check legality and opponent counterplay before reusing them."
    if "draw" in outcome_l:
        return "When the position is drawish, prefer moves that keep legal flexibility and avoid giving the opponent a forced win."
    return ""


def _float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default
