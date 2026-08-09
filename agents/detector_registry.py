"""Model-agnostic Stage-A detector registry for OG Skill Alignment.

Switches the zero-shot open-vocabulary detector via env:
  ROBOAGENT_DETECTOR = llmdet | gdino
  ROBOAGENT_DETECTOR_PATH (optional local ckpt)
  ROBOAGENT_DETECTOR_THRESHOLD (default 0.35)

Both LLMDet (MMGroundingDinoForObjectDetection) and GroundingDINO
(GroundingDinoForObjectDetection) implement the same HF grounded-OD API:
  AutoProcessor(images, text=[[q, "a q"]]) -> model -> post_process_grounded_object_detection
so the SAME caller works for both. No per-model rule rewriting.

This is the generalization mechanism: the Alignment layer is detector-agnostic.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None
_MODEL_ID = None
_DEVICE = None


def detector_name() -> str:
    return os.environ.get("ROBOAGENT_DETECTOR", "llmdet").strip().lower()


def _ensure_model():
    global _MODEL, _PROCESSOR, _MODEL_ID, _DEVICE
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        name = detector_name()
        if name == "gdino":
            model_id = os.environ.get("ROBOAGENT_DETECTOR_MODEL", "IDEA-Research/grounding-dino-base")
            local = os.environ.get("ROBOAGENT_DETECTOR_PATH", "").strip()
        else:
            model_id = os.environ.get("ROBOAGENT_DETECTOR_MODEL", "iSEE-Laboratory/llmdet_large")
            local = os.environ.get("ROBOAGENT_DETECTOR_PATH", "").strip() or os.environ.get("ROBOAGENT_LLMDET_PATH", "").strip()
        load_id = local if local else model_id
        device = os.environ.get("ROBOAGENT_DETECTOR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        _PROCESSOR = AutoProcessor.from_pretrained(load_id)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(load_id)
        _MODEL.to(device)
        _MODEL.eval()
        _MODEL_ID = load_id
        _DEVICE = device


def detect(
    image,
    text_query: str,
    threshold: float = 0.35,
) -> Tuple[Optional[dict], dict]:
    """Run the active detector. Returns (best_det_or_None, meta)."""
    import torch

    _ensure_model()
    text_labels = [[text_query, f"a {text_query.lower()}"]]
    inputs = _PROCESSOR(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    t0 = __import__("time").time()
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    latency_ms = (__import__("time").time() - t0) * 1000.0

    try:
        results = _PROCESSOR.post_process_grounded_object_detection(
            outputs, threshold=threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=text_labels,
        )
    except Exception:
        results = _PROCESSOR.post_process_grounded_object_detection(
            outputs, threshold=threshold,
            target_sizes=[(image.height, image.width)],
        )

    result = results[0]
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    labels = result.get("text_labels") or result.get("labels") or []

    meta = {
        "det_query": text_query,
        "detector": detector_name(),
        "model_id": _MODEL_ID,
        "latency_ms": round(latency_ms, 2),
        "n_dets": int(len(scores)) if scores is not None else 0,
    }
    if scores is None or len(scores) == 0:
        return None, meta

    best_i = int(scores.argmax().item()) if hasattr(scores, "argmax") else 0
    best_score = float(scores[best_i])
    box = None
    if boxes is not None and len(boxes) > best_i:
        box = [float(x) for x in boxes[best_i].tolist()]
    det_label = None
    if labels is not None and len(labels) > best_i:
        det_label = labels[best_i]
    return {
        "score": best_score,
        "box": box,
        "det_label": det_label,
    }, meta
