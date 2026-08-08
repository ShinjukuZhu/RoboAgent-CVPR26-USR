#!/usr/bin/env python3
"""Gated OG v2: raise Stage-C rate OR force atomic B-accept.

Env (in addition to v1):
  ROBOAGENT_OG_BACKEND=llmdet_qwen_gated_v2   (alias: gated_cascade_v2)
  ROBOAGENT_OG_GATE_THRESHOLD                 default 0.65 (v1 was 0.50)
  ROBOAGENT_OG_ATOMIC_BACCEPT=1               if set: atomic + score>=min → always B-accept
  ROBOAGENT_OG_ATOMIC_BACCEPT_MIN             default 0.50

Hypothesis (post G2 FAIL):
  v1 gate_thr=0.50 over-B-accepted mid-confidence dets → EB dipped vs ungated.
  Raising thr to 0.65 sends more mid-score cases to Qwen Stage C (EB-friendly).
  Optional atomic_baccept ablation keeps high-conf atomic presence wins on AW.
"""
from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    from agents.og_cascade_gated import (  # type: ignore
        _parse_qwen,
        _empty_cuda_cache,
        _import_cascade_helpers,
        gate_threshold as _gate_threshold_v1,
    )
    from agents.contract_adapter import adapt_label  # type: ignore
    from agents.skill_interface import SkillResult  # type: ignore
except ImportError:
    from og_cascade_gated import (  # type: ignore
        _parse_qwen,
        _empty_cuda_cache,
        _import_cascade_helpers,
        gate_threshold as _gate_threshold_v1,
    )
    from contract_adapter import adapt_label  # type: ignore
    from skill_interface import SkillResult  # type: ignore


def gate_threshold(default: float = 0.65) -> float:
    return float(os.environ.get("ROBOAGENT_OG_GATE_THRESHOLD", str(default)))


def atomic_baccept_enabled() -> bool:
    return os.environ.get("ROBOAGENT_OG_ATOMIC_BACCEPT", "0").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def atomic_baccept_min(default: float = 0.50) -> float:
    return float(os.environ.get("ROBOAGENT_OG_ATOMIC_BACCEPT_MIN", str(default)))


def should_invoke_stage_c_v2(
    evidence: dict,
    adapter,
    thr_gate: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """v2 gating.

    Base: same as v1 (compositional / miss / score < thr / adapter ambiguity).
    Override: if ATOMIC_BACCEPT and not compositional and score >= min → force B-accept.
    """
    if thr_gate is None:
        thr_gate = gate_threshold()

    reasons: List[str] = []
    compositional = bool(evidence.get("compositional")) or bool(getattr(adapter, "compositional", False))
    if compositional:
        reasons.append("compositional")

    score = evidence.get("score")
    if score is None or not evidence.get("detector_found"):
        reasons.append("no_or_miss_detector")
    else:
        try:
            if float(score) < float(thr_gate):
                reasons.append("score_below_gate")
        except (TypeError, ValueError):
            reasons.append("score_unparseable")

    if getattr(adapter, "ambiguity", False):
        for r in list(getattr(adapter, "ambiguity_reasons", []) or []):
            if r not in reasons:
                reasons.append(f"adapter:{r}")
        if "adapter_ambiguity" not in reasons and not any(x.startswith("adapter:") for x in reasons):
            reasons.append("adapter_ambiguity")

    # Dedup
    seen = set()
    ordered: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)

    # Atomic high-conf force B-accept ablation
    if atomic_baccept_enabled() and ordered:
        try:
            sc = float(score) if score is not None else None
        except (TypeError, ValueError):
            sc = None
        if (
            (not compositional)
            and evidence.get("detector_found")
            and sc is not None
            and sc >= atomic_baccept_min()
            and getattr(adapter, "label", None)
        ):
            return False, ["forced_atomic_baccept"]

    return bool(ordered), ordered


def ground_gated_v2(
    image_path: str,
    target_obj: str,
    base_prompt: str,
    qwen_infer: Callable[[str], str],
    threshold: Optional[float] = None,
    thr_gate: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[Any, dict]:
    """Gated cascade v2 → (False|list, meta)."""
    import time

    detect_evidence, augment_og_prompt = _import_cascade_helpers()
    if thr_gate is None:
        thr_gate = gate_threshold()

    evidence, meta = detect_evidence(
        image_path, target_obj, threshold=threshold, last_goto=last_goto
    )
    if observed_objects is not None:
        meta["n_observed"] = len(list(observed_objects))

    adapter = adapt_label(
        target_obj,
        last_goto=last_goto,
        detector_found=bool(evidence.get("detector_found")),
        det_score=evidence.get("score"),
        det_label=evidence.get("det_label"),
    )
    meta["stage_b"] = "contract_adapter"
    meta["adapter_path"] = adapter.path
    meta["adapter_label"] = adapter.label
    meta["adapter_ambiguity"] = adapter.ambiguity
    meta["adapter_reasons"] = list(adapter.ambiguity_reasons)
    meta["thr_gate"] = thr_gate
    meta["policy"] = "gated_v2"
    meta["atomic_baccept"] = atomic_baccept_enabled()

    need_c, gate_reasons = should_invoke_stage_c_v2(evidence, adapter, thr_gate=thr_gate)
    meta["gate_reasons"] = gate_reasons
    meta["stage_c_invoked"] = need_c

    if not need_c:
        meta["phase"] = "gated_cascade_v2"
        meta["path"] = "b_accept"
        meta["stage_c"] = "skipped"
        if not evidence.get("detector_found") or not adapter.label:
            meta["reject"] = "b_accept_without_evidence"
            skill = SkillResult(
                found=False,
                failure_type="b_accept_without_evidence",
                meta=meta,
            )
            return skill.to_og_return(), meta

        meta["canonical_label"] = adapter.label
        meta["qwen_found"] = None
        skill = SkillResult(
            found=True,
            label=adapter.label,
            meta={
                **meta,
                "box": evidence.get("box"),
                "score": evidence.get("score"),
                "detector_found": evidence.get("detector_found"),
                "query": adapter.query,
            },
        )
        _empty_cuda_cache()
        return skill.to_og_return(), meta

    aug_prompt = augment_og_prompt(base_prompt, evidence)
    if adapter.label:
        aug_prompt = (
            f"[Adapter candidate label={adapter.label!s} path={adapter.path} "
            f"— auxiliary; YOU decide final answer]\n"
            + aug_prompt
        )
    meta["prompt_augmented"] = True
    meta["stage_c"] = "qwen"

    t1 = time.time()
    res = qwen_infer(aug_prompt)
    if res is None:
        res = "no"
    res = str(res).strip()
    meta["stage_c_ms"] = round((time.time() - t1) * 1000.0, 2)
    meta["qwen_raw"] = res[:2000]

    ret = _parse_qwen(res, parse_fn, meta)
    if ret is False or ret is None:
        ret = False
    elif not (
        isinstance(ret, list)
        and ret
        and isinstance(ret[0], dict)
        and "label" in ret[0]
    ):
        meta["parse_error"] = meta.get("parse_error") or "invalid_og_shape"
        ret = False

    meta["qwen_found"] = ret is not False
    meta["phase"] = "gated_cascade_v2"
    meta["path"] = "stage_c_qwen"

    if ret is not False:
        meta["canonical_label"] = ret[0].get("label")
        if evidence.get("box") is not None:
            ret = [dict(ret[0])]
            ret[0].setdefault("box", evidence["box"])
            ret[0].setdefault("score", evidence.get("score"))
            ret[0].setdefault("detector_found", evidence["detector_found"])
    else:
        meta["reject"] = meta.get("reject") or "qwen_abstain"

    _empty_cuda_cache()
    return ret, meta


ground_cascade_gated_v2 = ground_gated_v2
