"""Two-stage OG cascade: LLMDet detect → fine-tuned Qwen canonicalize / abstain.

Serial pipeline (NOT the old Hybrid router):
  Stage A: LLMDet presence/box evidence on image+query
  Stage B: same fine-tuned Qwen OG as baseline decides final label or no

Return contract identical to baseline OG:
  - False when not found
  - [{"label": <str>, ...}] when found  (label MUST be Qwen's canonical output)

Env:
  ROBOAGENT_OG_BACKEND=llmdet_qwen|cascade  (wired in agent.py)
  ROBOAGENT_LLMDET_PATH / ROBOAGENT_LLMDET_THRESHOLD  (Stage A)
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, Sequence, Tuple


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def detect_evidence(
    image_path: str,
    target_obj: str,
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Stage A: run LLMDet and return (evidence, meta_fields).

    Always returns evidence even on miss so Stage B can still judge.
    """
    from PIL import Image

    # Reuse Fix1.5 helpers / model loader (same process, shared GPU).
    if os.environ.get("ROBOAGENT_DETECTOR", "llmdet").lower() != "llmdet":
        from agents.detector_registry import detect as _detect  # type: ignore
        from agents.detector_registry import detector_name
        from agents.contract_adapter import is_compositional as _is_compositional
        from agents.contract_adapter import normalize_query as _normalize_query
        from agents.contract_adapter import soft_type as _soft_type
        def _ensure_model():
            from agents.detector_registry import _ensure_model as _em
            _em()
        meta_ext = {"detector": detector_name()}
    else:
        from agents.llmdet_og import (  # type: ignore
            _detect,
            _ensure_model,
            _is_compositional,
            _normalize_query,
            _soft_type,
        )
        meta_ext = {}

    _ensure_model()
    thr = threshold
    if thr is None:
        thr = float(os.environ.get("ROBOAGENT_LLMDET_THRESHOLD", "0.35"))

    image = Image.open(image_path).convert("RGB")
    query = _normalize_query(target_obj)
    compositional = _is_compositional(target_obj) or _is_compositional(query)

    # Prefer detecting soft_type(last_goto) for compositional goals (sink etc.).
    det_query = query
    if compositional:
        soft = _soft_type(last_goto)
        if soft:
            det_query = soft

    t0 = time.time()
    best, dmeta = _detect(image, det_query, thr)
    # Also try raw query if compositional soft miss and queries differ.
    if best is None and det_query != query:
        best2, dmeta2 = _detect(image, query, thr)
        if best2 is not None:
            best, dmeta = best2, dmeta2
            det_query = query
        else:
            dmeta = {**dmeta, "alt_query": query, "alt_n_dets": dmeta2.get("n_dets", 0)}

    evidence = {
        "detector_found": best is not None,
        "score": None if best is None else best.get("score"),
        "box": None if best is None else best.get("box"),
        "det_label": None if best is None else best.get("det_label"),
        "det_query": det_query,
        "query": query,
        "compositional": compositional,
        "last_goto": last_goto,
        "threshold": thr,
    }
    meta = {
        "phase": "cascade",
        "path": "llmdet_then_qwen",
        **meta_ext,
        "stage_a": "llmdet",
        "model_id": dmeta.get("model_id")
        or os.environ.get("ROBOAGENT_LLMDET_PATH")
        or os.environ.get("ROBOAGENT_LLMDET_MODEL", "llmdet"),
        **{k: v for k, v in dmeta.items() if k != "model_id"},
        "detector_found": evidence["detector_found"],
        "best_score": evidence["score"],
        "det_query": det_query,
        "compositional": compositional,
        "last_goto": last_goto,
        "stage_a_ms": round((time.time() - t0) * 1000.0, 2),
    }
    _empty_cuda_cache()
    return evidence, meta


def augment_og_prompt(base_prompt: str, evidence: dict) -> str:
    """Inject short detector evidence prefix; Qwen remains final arbiter."""
    found = bool(evidence.get("detector_found"))
    score = evidence.get("score")
    box = evidence.get("box")
    det_q = evidence.get("det_query")
    lg = evidence.get("last_goto")
    score_s = "none" if score is None else f"{float(score):.3f}"
    box_s = "none" if not box else "[" + ", ".join(f"{float(x):.1f}" for x in box) + "]"
    hint = (
        "[Detector evidence — auxiliary only; YOU decide the final answer]\n"
        f"detector_found={found} detector_score={score_s} "
        f"detector_box={box_s} det_query={det_q!s} last_goto={lg!s}\n"
        "Primary evidence is the image. Keep the required reply format "
        "(```json ...``` with a canonical object label, or no).\n"
        "Do not echo the detector query as the label unless it is the correct "
        "canonical name. If unsure the target is present, answer no.\n\n"
    )
    return hint + (base_prompt or "")


def ground_cascade(
    image_path: str,
    target_obj: str,
    base_prompt: str,
    qwen_infer: Callable[[str], str],
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[Any, dict]:
    """Full cascade: LLMDet evidence → Qwen OG → (False|list, meta).

    qwen_infer(augmented_prompt) -> raw string response (already stripped OK).
    parse_fn defaults to agents.stage0_utils.parse_og_response.
    """
    if parse_fn is None:
        from agents.stage0_utils import parse_og_response  # type: ignore

        parse_fn = parse_og_response

    evidence, meta = detect_evidence(
        image_path, target_obj, threshold=threshold, last_goto=last_goto
    )
    if observed_objects is not None:
        meta["n_observed"] = len(list(observed_objects))

    aug_prompt = augment_og_prompt(base_prompt, evidence)
    meta["prompt_augmented"] = True

    t1 = time.time()
    res = qwen_infer(aug_prompt)
    if res is None:
        res = "no"
    res = str(res).strip()
    meta["stage_b_ms"] = round((time.time() - t1) * 1000.0, 2)
    meta["stage_b"] = "qwen"
    meta["qwen_raw"] = res[:2000]

    # Tolerate missing fence / unexpected formats: abstain rather than crash eval.
    try:
        ret = parse_fn(res)
    except Exception as e:
        meta["parse_error"] = str(e)[:300]
        low = res.lower().strip()
        if low == "no" or low.startswith("no"):
            ret = False
        else:
            # Last-resort: extract ```json ... ``` if present.
            import json
            import re

            m = re.search(r"```json\s*(.*?)\s*```", res, flags=re.S | re.I)
            if m:
                try:
                    ret = json.loads(m.group(1))
                except Exception:
                    ret = False
            else:
                ret = False

    if ret is False or ret is None:
        ret = False
    elif not (isinstance(ret, list) and ret and isinstance(ret[0], dict) and "label" in ret[0]):
        meta["parse_error"] = meta.get("parse_error") or "invalid_og_shape"
        ret = False

    meta["qwen_found"] = ret is not False
    if ret is not False:
        meta["canonical_label"] = ret[0].get("label")
        # Attach detector box as optional extra (runner uses label only).
        if evidence.get("box") is not None:
            ret = [dict(ret[0])]
            ret[0].setdefault("box", evidence["box"])
            ret[0].setdefault("score", evidence.get("score"))
            ret[0].setdefault("detector_found", evidence["detector_found"])

    meta["phase"] = "cascade"
    meta["path"] = "llmdet_then_qwen"
    _empty_cuda_cache()
    return ret, meta
