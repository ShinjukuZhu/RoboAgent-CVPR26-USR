"""LLMDet-backed Object Grounding with RoboAgent-compatible return format.

Return contract (same as original OG / naive LLMDet that scored AW 81.3%):
  - False when not found
  - [{"label": <str>, ...}] when found  (bbox optional; runner mainly uses label)

Semantics (aligned with agent.py history + Qwen OG behavior):
  - history already records location as: found at {last_goto}
  - ret[0]["label"] is the OBJECT / tool name for SD & AD (book, sink, knife...)
  - last_goto is often a receptacle (Desk/Shelf/Sink), NOT the grasp target.
    Never replace a small-object query's label with a furniture last_goto (Phase0 bug).

Fix1 (no Qwen hybrid; Fix1.5 soft REMOVED — rejected):
  - Compositional: detect/return soft_type(last_goto) (e.g. sink).
  - Atomic type-match → instance id; else keep query label (book on Desk stays book).
  - NEVER apply atomic_prefer_last_goto_soft (couch→sofa etc.).
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Optional, Sequence, Tuple

_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None
_MODEL_ID = None
_DEVICE = None

_COMPOSITIONAL_RE = re.compile(
    r"(?i)\b("
    r"some tool|something to|somewhere|another |a place to|tool for|"
    r"the back of|on to the|onto the|for cleaning|for slicing|for heating|"
    r"for cooling|for washing"
    r")\b"
)

# Surfaces / large receptacles / place targets (ALFRED-style). Used only to
# decide whether last_goto is a *search location* vs the *intended target*.
_FURNITURE_RE = re.compile(
    r"(?i)\b("
    r"table|couch|sofa|chair|bed|desk|shelf|drawer|cabinet|counter|"
    r"sink|toilet|fridge|refrigerator|microwave|stove|oven|garbage|"
    r"safe|bathtub|dresser|stand|nightstand|ottoman|armchair|"
    r"sidetable|diningtable|coffeetable|tvstand|laundryhamper|"
    r"dining\s*table|side\s*table|coffee\s*table|tv\s*stand|"
    r"garbage\s*can|counter\s*top|sink\s*basin"
    r")\b"
)


def _normalize_query(target_obj: str) -> str:
    """Turn 'Apple 1' / 'sliced Apple 2' into a detector text query."""
    t = (target_obj or "").strip()
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()
    t = re.sub(r"\s+\d+$", "", t).strip()
    t = re.sub(r"(?i)^sliced\s+", "", t).strip()
    return t if t else (target_obj or "").strip()


def _is_compositional(target_obj: str) -> bool:
    q = (target_obj or "").strip()
    if not q:
        return False
    if _COMPOSITIONAL_RE.search(q):
        return True
    # Long descriptive phrases are not atomic object nouns.
    if len(q.split()) >= 5:
        return True
    return False


def _soft_type(name: Optional[str]) -> Optional[str]:
    """'Sink 1' / 'DiningTable 2' -> 'sink' / 'dining table' (Qwen-style)."""
    if not name:
        return None
    t = str(name).strip()
    t = re.sub(r"\s+\d+$", "", t).strip()
    if not t:
        return None
    # CamelCase / PascalCase -> spaced lowercase
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", t)
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t or None


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _looks_like_furniture(name: Optional[str]) -> bool:
    if not name:
        return False
    soft = _soft_type(name) or str(name)
    return bool(_FURNITURE_RE.search(soft))


def _type_match(last_goto: Optional[str], target_obj: str) -> bool:
    """True if last_goto refers to the same type as the atomic query."""
    a = _soft_type(last_goto)
    b = _soft_type(target_obj) or _normalize_query(target_obj).lower()
    if not a or not b:
        return False
    ca, cb = _compact(a), _compact(b)
    return ca == cb or ca in cb or cb in ca


def _prefer_last_goto_soft(query: str, last_goto: Optional[str]) -> bool:
    """Use soft_type(last_goto) as AD label when goto is the navigated target.

    Keep query when last_goto is only a search receptacle for a different
    small object (book found at Desk) — that was the Phase0 AW killer.
    """
    if not last_goto:
        return False
    if _type_match(last_goto, query):
        return False  # handled via instance id path
    goto_furn = _looks_like_furniture(last_goto)
    query_furn = _looks_like_furniture(query)
    # Object-on-receptacle: EG at Desk/Shelf/Fridge, query is apple/book/...
    if goto_furn and not query_furn:
        return False
    # Place synonym (couch→Sofa) or pickupable synonym (bar of soap→SoapBar).
    return True


def _ensure_model():
    global _MODEL, _PROCESSOR, _MODEL_ID, _DEVICE
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = os.environ.get("ROBOAGENT_LLMDET_MODEL", "iSEE-Laboratory/llmdet_large")
        local = os.environ.get("ROBOAGENT_LLMDET_PATH", "").strip()
        load_id = local if local else model_id
        device = os.environ.get(
            "ROBOAGENT_LLMDET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        _PROCESSOR = AutoProcessor.from_pretrained(load_id)
        _MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(load_id)
        _MODEL.to(device)
        _MODEL.eval()
        _MODEL_ID = load_id
        _DEVICE = device


def _detect(image, text_query: str, thr: float) -> Tuple[Optional[dict], dict]:
    """Run one LLMDet pass. Returns (best_det_or_None, meta_fields)."""
    import torch

    text_labels = [[text_query, f"a {text_query.lower()}"]]
    inputs = _PROCESSOR(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = _MODEL(**inputs)
    latency_ms = (time.time() - t0) * 1000.0

    results = _PROCESSOR.post_process_grounded_object_detection(
        outputs,
        threshold=thr,
        target_sizes=[(image.height, image.width)],
        text_labels=text_labels,
    )
    result = results[0]
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    labels = result.get("text_labels") or result.get("labels") or []

    meta = {
        "det_query": text_query,
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


def ground(
    image_path: str,
    target_obj: str,
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
) -> Any:
    """Run LLMDet OG. Returns (False|list[dict], meta)."""
    from PIL import Image

    _ensure_model()
    thr = threshold
    if thr is None:
        thr = float(os.environ.get("ROBOAGENT_LLMDET_THRESHOLD", "0.35"))

    image = Image.open(image_path).convert("RGB")
    query = _normalize_query(target_obj)
    compositional = _is_compositional(target_obj) or _is_compositional(query)

    meta = {
        "model_id": _MODEL_ID,
        "query": query,
        "threshold": thr,
        "last_goto": last_goto,
        "compositional": compositional,
        "phase": "fix1",
    }

    # ----- Compositional / functional goals (no Qwen hybrid) -----
    if compositional:
        soft = _soft_type(last_goto)
        if soft:
            best, dmeta = _detect(image, soft, thr)
            meta.update(dmeta)
            if best is not None:
                det = {
                    "label": soft,
                    "score": best["score"],
                    "box": best["box"],
                    "query": query,
                    "det_label": best.get("det_label"),
                    "det_query": soft,
                }
                meta["best_score"] = best["score"]
                meta["canonical_label"] = soft
                meta["path"] = "compositional_detect_last_goto_type"
                return [det], meta
            det = {
                "label": soft,
                "score": None,
                "box": None,
                "query": query,
                "det_query": soft,
            }
            meta["canonical_label"] = soft
            meta["path"] = "compositional_trust_last_goto"
            meta["reject"] = "no_dets_but_trust_last_goto"
            return [det], meta

        best, dmeta = _detect(image, query, thr)
        meta.update(dmeta)
        if best is None:
            meta["reject"] = "no_dets"
            meta["path"] = "compositional_no_goto"
            return False, meta
        det = {
            "label": target_obj,
            "score": best["score"],
            "box": best["box"],
            "query": query,
            "det_label": best.get("det_label"),
        }
        meta["best_score"] = best["score"]
        meta["path"] = "compositional_phrase_naive_label"
        return [det], meta

    # ----- Atomic object goals -----
    best, dmeta = _detect(image, query, thr)
    meta.update(dmeta)
    if best is None:
        meta["reject"] = "no_dets"
        meta["path"] = "atomic"
        return False, meta

    # Same type as query → prefer instance id (Ladle 1).
    if last_goto and _type_match(last_goto, target_obj):
        label = last_goto
        path = "atomic_type_match_instance"
    else:
        # Fix1: book on Desk keeps label=book (NOT desk). Fix1.5 soft disabled.
        label = target_obj
        path = "atomic_naive_label"

    det = {
        "label": label,
        "score": best["score"],
        "box": best["box"],
        "query": query,
        "det_label": best.get("det_label"),
    }
    meta["best_score"] = best["score"]
    meta["canonical_label"] = label
    meta["path"] = path
    return [det], meta
