"""Naive detector backend: use detector's raw label directly (no alignment).

For generalization comparison: shows that naive GDINO replacement (like naive
LLMDet) breaks the Brain contract, while the Alignment layer restores it.
"""
from __future__ import annotations
import os
from typing import Any, Optional, Sequence, Tuple


def _normalize_query(target_obj: str) -> str:
    import re
    t = (target_obj or "").strip()
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()
    t = re.sub(r"\s+\d+$", "", t).strip()
    return t if t else (target_obj or "").strip()


def ground_naive(
    image_path: str,
    target_obj: str,
    base_prompt: str = "",
    qwen_infer=None,
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn=None,
    env_name: str = "alfworld",
) -> Tuple[Any, dict]:
    """Run active detector, return its raw best label directly (no remap)."""
    from PIL import Image
    from agents.detector_registry import detect

    if threshold is None:
        threshold = float(os.environ.get("ROBOAGENT_DETECTOR_THRESHOLD", os.environ.get("ROBOAGENT_LLMDET_THRESHOLD", "0.35")))
    image = Image.open(image_path).convert("RGB")
    query = _normalize_query(target_obj)
    best, meta = detect(image, query, threshold)
    if best is None:
        meta["path"] = "naive_no_det"
        return False, meta
    det_label = best.get("det_label") or query
    meta["path"] = "naive_det_label"
    meta["detector_label"] = det_label
    return [{"label": det_label, "score": best["score"], "box": best["box"]}], meta


ground = ground_naive
