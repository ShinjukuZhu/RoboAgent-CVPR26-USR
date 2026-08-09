#!/usr/bin/env python3
"""Skill Memory stubs (Phase 3) — log-first, cross-ep adapt OFF by default.

GPU-free. No agent.py wiring. Aligns with proposal.md §5.4 and
experiments/phase3_skill_memory/DESIGN.md.

Modes (when later wired via env):
  ROBOAGENT_SKILL_MEMORY=off|log|adapt   (design default: log)
  Cross-episode adapt_params / hints only when enable_cross_episode_adapt=True.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Canonical failure taxonomy (DESIGN.md §2)
FAILURE_TYPES = frozenset(
    {
        "detection_miss",
        "navigation_fail",
        "grasp_fail",
        "exploration_exhausted",
    }
)

# Optional soft buckets — log in meta, not promoted without paper note
SOFT_FAILURE_TYPES = frozenset(
    {"multi_object", "precondition", "scheduler_parse", "unknown"}
)

# Detector thr clamp for optional adapt (DESIGN.md §3.1)
DET_THR_MIN = 0.20
DET_THR_MAX = 0.50
DET_THR_STEP = 0.02
DET_THR_DEFAULT = 0.35


@dataclass
class CallRecord:
    skill: str
    inputs: Dict[str, Any]
    outputs: Any
    success: bool
    failure_type: Optional[str] = None
    episode_id: str = ""
    step: int = -1
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


class SkillMemory:
    """Minimal viable Skill Memory: record + histograms; adapt/hints gated."""

    def __init__(
        self,
        *,
        enable_cross_episode_adapt: bool = False,
        persist_path: Optional[str] = None,
        initial_det_thr: float = DET_THR_DEFAULT,
    ) -> None:
        self.history: List[CallRecord] = []
        self.failure_patterns: Dict[str, int] = {}
        self.adaptive_params: Dict[str, Dict[str, Any]] = {
            "og": {"det_thr": float(initial_det_thr)},
        }
        self.enable_cross_episode_adapt = bool(enable_cross_episode_adapt)
        self.persist_path = persist_path
        self._episode_failure_counts: Dict[str, int] = {}

    def record(
        self,
        skill: str,
        inputs: Dict[str, Any],
        outputs: Any,
        success: bool,
        failure_type: Optional[str] = None,
        *,
        episode_id: str = "",
        step: int = -1,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CallRecord:
        ft = failure_type
        if ft is not None and ft not in FAILURE_TYPES and ft not in SOFT_FAILURE_TYPES:
            meta = dict(meta or {})
            meta.setdefault("raw_failure_type", ft)
            ft = "unknown"

        rec = CallRecord(
            skill=str(skill),
            inputs=dict(inputs or {}),
            outputs=outputs,
            success=bool(success),
            failure_type=None if success else ft,
            episode_id=episode_id,
            step=step,
            meta=dict(meta or {}),
        )
        self.history.append(rec)

        if not success and rec.failure_type:
            self.failure_patterns[rec.failure_type] = (
                self.failure_patterns.get(rec.failure_type, 0) + 1
            )
            self._episode_failure_counts[rec.failure_type] = (
                self._episode_failure_counts.get(rec.failure_type, 0) + 1
            )

        if self.persist_path:
            self._append_jsonl(rec)

        return rec

    def adapt_params(self, skill: str, failure_type: Optional[str]) -> Dict[str, Any]:
        """Optionally nudge skill-local params. No-op unless cross-ep adapt enabled.

        Safe knob: OG det_thr down on detection_miss (clamped). Never mutates Brain.
        """
        skill = str(skill)
        cur = dict(self.adaptive_params.get(skill, {}))
        if not self.enable_cross_episode_adapt:
            return cur

        if skill == "og" and failure_type == "detection_miss":
            thr = float(cur.get("det_thr", DET_THR_DEFAULT))
            thr = max(DET_THR_MIN, round(thr - DET_THR_STEP, 4))
            cur["det_thr"] = thr
            self.adaptive_params[skill] = cur
            return cur

        self.adaptive_params.setdefault(skill, cur)
        return cur

    def get_context_hint(self, skill: str, current_query: Any) -> str:
        """Prompt enhancement from history.

        Cross-episode hints only if enable_cross_episode_adapt (default OFF) —
        avoid ALFWorld OOD / EB eval contamination. Within-episode streak hint
        is allowed even when adapt is OFF (fair: no other-ep leakage).
        """
        skill = str(skill)
        if self._episode_failure_counts:
            top = sorted(
                self._episode_failure_counts.items(), key=lambda x: -x[1]
            )
            kind, n = top[0]
            if n >= 2:
                return (
                    f"[skill_memory within-ep] {skill} query={current_query!r}: "
                    f"repeated {kind} x{n}"
                )

        if not self.enable_cross_episode_adapt:
            return ""

        if not self.failure_patterns:
            return ""
        parts = [f"{k}:{v}" for k, v in sorted(self.failure_patterns.items())]
        return (
            f"[skill_memory cross-ep] {skill} query={current_query!r}; "
            f"failures {{{', '.join(parts)}}}"
        )

    def dump_failure_histogram(self) -> Dict[str, int]:
        return dict(self.failure_patterns)

    def reset_episode(self) -> None:
        """Clear within-ep counters; keep append-only history / patterns for log mode."""
        self._episode_failure_counts.clear()

    def reset_all(self) -> None:
        """Eval hygiene: drop all state (use between fair-eval runs)."""
        self.history.clear()
        self.failure_patterns.clear()
        self._episode_failure_counts.clear()
        self.adaptive_params = {"og": {"det_thr": DET_THR_DEFAULT}}

    def _append_jsonl(self, rec: CallRecord) -> None:
        assert self.persist_path
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), default=str) + "\n")


def env_mode(default: str = "off") -> str:
    """ROBOAGENT_SKILL_MEMORY=off|log|adapt (default off)."""
    import os

    m = os.environ.get("ROBOAGENT_SKILL_MEMORY", default).strip().lower()
    if m in ("1", "true", "yes"):
        return "log"
    if m in ("off", "0", "false", "no", "log", "adapt"):
        return "off" if m in ("0", "false", "no") else m
    return default


def normalize_failure_type(raw: Optional[str]) -> Optional[str]:
    """Map loose labels to canonical taxonomy; unknown -> 'unknown'."""
    if raw is None:
        return None
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "miss": "detection_miss",
        "og_miss": "detection_miss",
        "not_found": "detection_miss",
        "far": "navigation_fail",
        "nav": "navigation_fail",
        "nav_fail": "navigation_fail",
        "pick_fail": "grasp_fail",
        "put_fail": "grasp_fail",
        "manipulate": "grasp_fail",
        "eg_none": "exploration_exhausted",
        "explore_exhaust": "exploration_exhausted",
        "exhausted": "exploration_exhausted",
    }
    s = aliases.get(s, s)
    if s in FAILURE_TYPES:
        return s
    if s in SOFT_FAILURE_TYPES:
        return s
    return "unknown"


ES_SUGGEST_NEXT = frozenset(
    {
        "retry_ad",
        "og_redetect",
        "eg_reposition",
        "sd_refresh",
        "abort_subgoal",
        "none",
    }
)


def make_es_structured(
    *,
    success: bool,
    failure_type: Optional[str] = None,
    progress: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    suggest_next: str = "none",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Build es_v1 dict; does not call any model."""
    ft = None if success else normalize_failure_type(failure_type)
    sug = suggest_next if suggest_next in ES_SUGGEST_NEXT else "none"
    out: Dict[str, Any] = {
        "$schema_version": "es_v1",
        "success": bool(success),
        "failure_type": ft,
        "progress": progress or ("ok" if success else "failed"),
        "evidence": dict(evidence or {}),
        "suggest_next": sug,
    }
    if confidence is not None:
        out["confidence"] = float(confidence)
    return out


def es_feedback_line(es: Dict[str, Any]) -> str:
    """Human-readable line compatible with Summarization feedback: prefix."""
    return str(es.get("progress") or "")


if __name__ == "__main__":
    # GPU-free unit smoke
    mem = SkillMemory(enable_cross_episode_adapt=False)
    assert mem.get_context_hint("og", "Apple 1") == ""

    mem.record(
        "og",
        {"query": "Apple 1"},
        False,
        success=False,
        failure_type="detection_miss",
        episode_id="ep0",
        step=1,
    )
    mem.record(
        "og",
        {"query": "Apple 1"},
        False,
        success=False,
        failure_type="detection_miss",
        episode_id="ep0",
        step=2,
    )
    hint = mem.get_context_hint("og", "Apple 1")
    assert "within-ep" in hint and "detection_miss" in hint

    # adapt OFF -> det_thr unchanged
    before = mem.adaptive_params["og"]["det_thr"]
    after = mem.adapt_params("og", "detection_miss")
    assert after["det_thr"] == before == DET_THR_DEFAULT

    mem_adapt = SkillMemory(enable_cross_episode_adapt=True, initial_det_thr=0.35)
    mem_adapt.adapt_params("og", "detection_miss")
    assert mem_adapt.adaptive_params["og"]["det_thr"] == 0.33
    mem_adapt.adaptive_params["og"]["det_thr"] = DET_THR_MIN
    mem_adapt.adapt_params("og", "detection_miss")
    assert mem_adapt.adaptive_params["og"]["det_thr"] == DET_THR_MIN

    assert mem_adapt.get_context_hint("og", "cup") == ""
    mem_adapt.record("eg", {}, None, False, "exploration_exhausted")
    assert "cross-ep" in mem_adapt.get_context_hint("eg", "cup")

    assert normalize_failure_type("far") == "navigation_fail"
    assert normalize_failure_type("eg_none") == "exploration_exhausted"
    assert normalize_failure_type("weird") == "unknown"

    es = make_es_structured(
        success=False,
        failure_type="far",
        progress="apple still far",
        evidence={"env_feedback": "far"},
        suggest_next="eg_reposition",
        confidence=0.6,
    )
    assert es["$schema_version"] == "es_v1"
    assert es["failure_type"] == "navigation_fail"
    assert es_feedback_line(es) == "apple still far"

    hist = mem.dump_failure_histogram()
    assert hist.get("detection_miss") == 2

    mem.reset_episode()
    assert mem.get_context_hint("og", "Apple 1") == ""
    assert mem.dump_failure_histogram().get("detection_miss") == 2

    mem.reset_all()
    assert mem.history == [] and mem.failure_patterns == {}

    print("skill_memory.py: OK")
