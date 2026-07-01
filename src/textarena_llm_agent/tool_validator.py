"""Orchestrator for the 5-stage Voyager-style tool pipeline.

Owns the closed lifecycle vocabulary:

    tool_need -> tool_spec -> candidate_tool -> validated_tool -> active_tool

with failure sinks ``demoted`` (recoverable) and ``disabled`` (terminal).

The Critic Agent's ``propose_tool`` call hands off here. Each stage:
  * mutates ``ToolLibrary`` (status, scores, version)
  * writes an evidence-graph edge so the audit chain is preserved

This module never decides *whether* to synthesize a tool — that's the
``ToolNeedDetector`` / Critic's job. It only drives a given tool through
its remaining lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tool_library import SynthesizedToolRecord, ToolLibrary, ToolStatus
from .tool_synthesis import ToolSynthesizer


@dataclass(slots=True)
class StageResult:
    ok: bool
    stage: str
    record_id: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineResult:
    """Final outcome of driving a tool from tool_need through to a terminal stage."""
    record_id: str
    final_status: str
    activated_name: str | None
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.final_status == ToolStatus.ACTIVE_TOOL.value


class ToolValidator:
    """End-to-end driver. The Critic calls ``run_pipeline`` once per proposal."""

    def __init__(self, *, synthesizer: ToolSynthesizer, library: ToolLibrary,
                 graph: Any = None, replay_threshold: float = 0.6,
                 ab_min_delta: float = 0.0) -> None:
        self.synthesizer = synthesizer
        self.library = library
        self.graph = graph                  # EvidenceGraph or None
        self.replay_threshold = replay_threshold
        self.ab_min_delta = ab_min_delta

    # ------------------------------------------------------------------ entrypoints
    def create_need(self, *, task_description: str, game_id: str,
                    policy_version: str = "v0",
                    source_node: tuple[str, str] | None = None) -> SynthesizedToolRecord:
        """Stage 0 entry: register a recurring sub-problem.

        If ``source_node`` is given (e.g. ``("critic_report", report_id)``), a
        REQUESTS edge is written from that node to the new tool record.
        """
        rec = self.library.create_need(
            task_description=task_description, game_id=game_id,
            policy_version=policy_version,
        )
        self._ensure_graph_node(rec)
        if source_node and self.graph is not None:
            src_type, src_id = source_node
            try:
                self.graph.add_edge(src_type, src_id, "REQUESTS",
                                    "synthesized_tool", rec.id)
            except Exception:
                pass
        return rec

    def run_pipeline(self, *, record_id: str, context_summary: str,
                     game_state_snapshot: dict[str, Any],
                     replay_frames: list[dict[str, Any]],
                     active_scores: dict[str, float] | None = None,
                     visible_text: str = "",
                     policy_version: str = "v0") -> PipelineResult:
        """Drive a record from its current stage to terminal.

        Stops at the first failure (record is demoted/disabled by the stage).
        """
        rec = self.library.get(record_id)
        if rec is None:
            return PipelineResult(record_id=record_id, final_status="missing",
                                  activated_name=None)
        stages: list[StageResult] = []
        # Stage 1: spec (only if currently at tool_need)
        if rec.status == ToolStatus.TOOL_NEED.value:
            r1 = self.synthesizer.synthesize_spec(
                record_id=record_id, task_description=rec.task_description,
                game_id=rec.game_id, context_summary=context_summary,
                game_state_snapshot=game_state_snapshot,
            )
            stages.append(StageResult(ok=bool(r1.get("ok")), stage="synthesize_spec",
                                      record_id=record_id,
                                      status=str(r1.get("status") or rec.status),
                                      detail=r1))
            if not r1.get("ok"):
                return self._finalize(record_id, stages)
        # Stage 2: compile candidate
        rec = self.library.get(record_id)
        if rec and rec.status == ToolStatus.TOOL_SPEC.value:
            r2 = self.synthesizer.compile_candidate(
                record_id=record_id, game_state_snapshot=game_state_snapshot,
                visible_text=visible_text,
            )
            stages.append(StageResult(ok=bool(r2.get("ok")), stage="compile_candidate",
                                      record_id=record_id,
                                      status=str(r2.get("status") or "demoted"),
                                      detail=r2))
            if not r2.get("ok"):
                return self._finalize(record_id, stages)
        # Stage 3: replay eval
        rec = self.library.get(record_id)
        if rec and rec.status == ToolStatus.CANDIDATE_TOOL.value:
            r3 = self.synthesizer.replay_eval(
                record_id=record_id, replay_frames=replay_frames,
                threshold=self.replay_threshold, visible_text=visible_text,
            )
            stages.append(StageResult(ok=bool(r3.get("ok")), stage="replay_eval",
                                      record_id=record_id,
                                      status=str(r3.get("status") or "demoted"),
                                      detail=r3))
            if not r3.get("ok"):
                return self._finalize(record_id, stages)
        # Stage 4: A/B
        rec = self.library.get(record_id)
        if rec and rec.status == ToolStatus.VALIDATED_TOOL.value:
            r4 = self.synthesizer.ab_test(
                record_id=record_id, active_scores=active_scores,
                candidate_score=rec.replay_score, min_delta=self.ab_min_delta,
            )
            stages.append(StageResult(ok=bool(r4.get("ok")), stage="ab_test",
                                      record_id=record_id,
                                      status=str(r4.get("status") or rec.status),
                                      detail=r4))
            if not r4.get("ok"):
                return self._finalize(record_id, stages)
        # Stage 5: activate
        rec = self.library.get(record_id)
        if rec and rec.status == ToolStatus.VALIDATED_TOOL.value:
            r5 = self.synthesizer.activate(record_id=record_id,
                                            policy_version=policy_version)
            stages.append(StageResult(ok=bool(r5.get("ok")), stage="activate",
                                      record_id=record_id,
                                      status=str(r5.get("status") or rec.status),
                                      detail=r5))
        return self._finalize(record_id, stages)

    # ------------------------------------------------------------------ helpers
    def _finalize(self, record_id: str, stages: list[StageResult]) -> PipelineResult:
        rec = self.library.get(record_id)
        final_status = rec.status if rec else "missing"
        activated_name = rec.name if rec and rec.status == ToolStatus.ACTIVE_TOOL.value else None
        # Reflect the final stage transitions in the evidence graph.
        if rec is not None:
            self._ensure_graph_node(rec)
        return PipelineResult(record_id=record_id, final_status=final_status,
                              activated_name=activated_name, stages=stages)

    def _ensure_graph_node(self, rec: SynthesizedToolRecord) -> None:
        if self.graph is None:
            return
        try:
            self.graph.add_node(
                "synthesized_tool", rec.id,
                tool_id=rec.tool_id, name=rec.name, status=rec.status,
                version=rec.version, game_id=rec.game_id,
                policy_version=rec.policy_version,
                replay_score=rec.replay_score, ab_score=rec.ab_score,
                attrs={"task_description": rec.task_description},
            )
        except Exception:
            pass
