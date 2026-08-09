"""Gated OG cascade: LLMDet (A) → contract adapter (B) → optional Qwen (C).

Stage C runs ONLY when gated in (G3):
  compositional OR det_score < thr_gate OR adapter.ambiguity
  [v2 also:] multiword query OR non-type-match atomic soft label
Otherwise B-accept the Stage A/B label (atomic high-confidence type-match).

Env:
  ROBOAGENT_OG_BACKEND=llmdet_qwen_gated | llmdet_qwen_gated_v2
  ROBOAGENT_OG_GATE_V2=1                  Enable hybrid v2 gate (also via *_v2 backend)
  ROBOAGENT_LLMDET_THRESHOLD              Stage A detect thr (default 0.35)
  ROBOAGENT_OG_GATE_THRESHOLD             Stage C gate; score < this → C (default 0.50)
  ROBOAGENT_OG_STRIP_IDS                  forwarded to contract adapter
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    from agents.contract_adapter import AdapterResult, adapt_label  # type: ignore
    from agents.skill_interface import SkillResult  # type: ignore
except ImportError:
    from contract_adapter import AdapterResult, adapt_label  # type: ignore
    from skill_interface import SkillResult  # type: ignore


def gate_threshold(default: float = 0.50) -> float:
    return float(os.environ.get("ROBOAGENT_OG_GATE_THRESHOLD", str(default)))


def gate_v2_enabled() -> bool:
    """Hybrid v2: B-accept only type-match high-score atomics; else prefer C."""
    v = os.environ.get("ROBOAGENT_OG_GATE_V2", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    backend = os.environ.get("ROBOAGENT_OG_BACKEND", "").strip().lower()
    return backend in ("llmdet_qwen_gated_v2", "gated_cascade_v2")


def _is_multiword_query(adapter: AdapterResult) -> bool:
    q = (adapter.query or adapter.meta.get("query") or "").strip()
    if not q:
        return False
    # Multi-token object phrases (e.g. "dining table", "butter knife").
    return len(q.split()) >= 2


def _is_non_type_match_soft(adapter: AdapterResult) -> bool:
    """Atomic path that did not type-match last_goto → may lose EB remap if B-accepted."""
    if adapter.compositional:
        return False
    path = (adapter.path or "").strip()
    if path == "atomic_type_match_instance":
        return False
    # atomic_naive_label / other non-type-match soft paths
    return path.startswith("atomic_") or path == "atomic_naive_label"


def should_invoke_stage_c(
    evidence: dict,
    adapter: AdapterResult,
    thr_gate: Optional[float] = None,
    v2: Optional[bool] = None,
) -> Tuple[bool, List[str]]:
    """Pure gating predicate (unit-testable, no GPU).

    Invoke Stage C iff any of:
      - compositional query
      - detector score missing or < thr_gate
      - adapter flags ambiguity
      - [v2] multiword query
      - [v2] non-type-match atomic soft label (even if score high)
    """
    if thr_gate is None:
        thr_gate = gate_threshold()
    if v2 is None:
        v2 = gate_v2_enabled()
    reasons: List[str] = []

    compositional = bool(evidence.get("compositional")) or adapter.compositional
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

    if adapter.ambiguity:
        for r in adapter.ambiguity_reasons:
            if r not in reasons:
                reasons.append(f"adapter:{r}")
        if "adapter_ambiguity" not in reasons and not any(
            x.startswith("adapter:") for x in reasons
        ):
            reasons.append("adapter_ambiguity")

    if v2:
        if _is_multiword_query(adapter):
            reasons.append("multiword_query")
        if _is_non_type_match_soft(adapter):
            reasons.append("non_type_match_soft")

    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return bool(ordered), ordered


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _import_cascade_helpers():
    """Prefer agents.* on server; fall back to sibling module name."""
    try:
        from agents.llmdet_qwen_og import (  # type: ignore
            augment_og_prompt,
            detect_evidence,
        )

        return detect_evidence, augment_og_prompt
    except ImportError:
        from llmdet_qwen_og import augment_og_prompt, detect_evidence  # type: ignore

        return detect_evidence, augment_og_prompt


def _parse_qwen(
    res: str,
    parse_fn: Optional[Callable[[str], Any]],
    meta: dict,
) -> Any:
    if parse_fn is None:
        try:
            from agents.stage0_utils import parse_og_response  # type: ignore

            parse_fn = parse_og_response
        except ImportError:
            parse_fn = None

    if parse_fn is not None:
        try:
            return parse_fn(res)
        except Exception as e:
            meta["parse_error"] = str(e)[:300]

    low = res.lower().strip()
    if low == "no" or low.startswith("no"):
        return False
    import json
    import re

    m = re.search(r"```json\s*(.*?)\s*```", res, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return False
    return False


def ground_gated(
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
    """Full gated cascade → (False|list, meta)."""
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
    v2 = gate_v2_enabled()
    meta["gate_v2"] = v2

    need_c, gate_reasons = should_invoke_stage_c(
        evidence, adapter, thr_gate=thr_gate, v2=v2
    )
    meta["gate_reasons"] = gate_reasons
    meta["stage_c_invoked"] = need_c

    # ----- B-accept: atomic high-conf type-match (v2) / high-conf atomic (v1) -----
    if not need_c:
        meta["phase"] = "gated_cascade_v2" if v2 else "gated_cascade"
        meta["path"] = "b_accept"
        meta["stage_c"] = "skipped"
        if not evidence.get("detector_found") or not adapter.label:
            # Defensive: gate should have caught this; abstain rather than invent.
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

    # ----- Stage C: Qwen arbiter (gated) -----
    aug_prompt = augment_og_prompt(base_prompt, evidence)
    # Hint adapter candidate without forcing Fix1.5 soft.
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
    meta["phase"] = "gated_cascade_v2" if v2 else "gated_cascade"
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


# Alias for agent wiring
ground_cascade_gated = ground_gated
