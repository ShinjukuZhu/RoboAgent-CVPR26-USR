"""USR OG backend — wraps remap v3 pipeline, emits USR to Brain.

Pipeline:  LLMDet raw → remap v3 alignment (canonicalization/functional/no-veto/
           found/fallback) → USR → Brain input.
The decision LABEL is EXACTLY remap v3's (Qwen label or fix1 fallback). USR adds
strict field separation:
  - environment_facts.object.class = v3 canonical label (Brain's input)
  - decision_signals = found / confidence / uncertainty / fallback_reason
  - provenance = detector / alignment_path / threshold / model-private artifacts
Model-private fields (det_query, raw box, qwen_raw) stay in `meta`, never in USR.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from agents.og_remap_only import ground_remap_only  # type: ignore


def _no_det_usr(meta: dict) -> dict:
    return {
        "environment_facts": {},
        "task_semantics": {},
        "decision_signals": {
            "found": False,
            "confidence": meta.get("best_score", meta.get("score")),
            "uncertainty": {"level": "high", "reason": "no_detection"},
            "fallback_reason": "no_det",
        },
        "provenance": {
            "skill": "object_grounding",
            "detector": meta.get("detector", "llmdet"),
            "alignment_path": meta.get("path", "no_det_false"),
            "threshold": meta.get("threshold"),
        },
        "temporal_context": {},
        "schema_version": "2.0",
    }


def _found_usr(label: str, meta: dict) -> dict:
    score = meta.get("best_score", meta.get("score"))
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None
    path = meta.get("path", "remap_qwen_label")
    return {
        "environment_facts": {"object": {"class": label}},
        "task_semantics": {},
        "decision_signals": {
            "found": True,
            "confidence": score_f,
            "uncertainty": {
                "level": "low" if (score_f or 0) >= 0.5 else "medium",
                "reason": "high_conf_no_veto" if path == "remap_qwen_label" else "fix1_fallback",
            },
            "fallback_reason": (
                "none" if path == "remap_qwen_label" else "fix1_fallback"
            ),
        },
        "provenance": {
            "skill": "object_grounding",
            "detector": meta.get("detector", "llmdet"),
            "alignment_path": path,
            "threshold": meta.get("threshold"),
        },
        "temporal_context": {},
        "schema_version": "2.0",
    }


def ground_usr(
    image_path: str,
    target_obj: str,
    base_prompt: str,
    qwen_infer: Callable[[str], str],
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
    detector: str = "llmdet",
) -> Tuple[Any, dict]:
    """Remap v3 decision, wrapped in USR. Returns (ret, meta_with_usr)."""
    ret, meta = ground_remap_only(
        image_path,
        target_obj,
        base_prompt,
        qwen_infer,
        threshold=threshold,
        last_goto=last_goto,
        observed_objects=observed_objects,
        parse_fn=parse_fn,
    )
    meta["detector"] = detector
    if ret is False:
        usr = _no_det_usr(meta)
    else:
        usr = _found_usr(ret[0]["label"], meta)
    meta["usr"] = usr
    return ret, meta


ground = ground_usr
