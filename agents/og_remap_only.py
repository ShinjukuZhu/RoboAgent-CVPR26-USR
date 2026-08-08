"""Remap-only OG v3: LLMDet → Qwen remap; forbid False when det≥thr.

History:
  v1 FINAL: AW 86.7% / EB 80% — Over-abstain fixed; GATE fail on EB +2pp
  v2 FINAL: AW 76.7% / EB 72% — aggressive instance-trust/synonym **REGRESSED**
  v3: restore v1 control flow + **surgical** Brain synonyms only
      (phone→CellPhone, keys→KeyChain, soap→SoapBar, butterknife→ButterKnife)
      No instance_nav_trust / no observed rewrite / no remap_instance_direct.

Env:
  ROBOAGENT_OG_BACKEND=llmdet_qwen_remap
  ROBOAGENT_LLMDET_THRESHOLD=0.35
  ROBOAGENT_OG_REMAP_NO_DET=false|qwen
  ROBOAGENT_OG_REMAP_V3=1 (default on; disables v2 aggressiveness)
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Optional, Sequence, Tuple

try:
    from agents.contract_adapter import (  # type: ignore
        adapt_label,
        functional_canonical,
        soft_type,
    )
except ImportError:
    from contract_adapter import (  # type: ignore
        adapt_label,
        functional_canonical,
        soft_type,
    )

# Surgical EB synonym map only (v2 blanket remap hurt AW/EB).
_SAFE_SYNONYM = {
    "phone": "CellPhone",
    "cellphone": "CellPhone",
    "cell phone": "CellPhone",
    "key": "KeyChain",
    "keys": "KeyChain",
    "keychain": "KeyChain",
    "key chain": "KeyChain",
    "set of keys": "KeyChain",
    "soap": "SoapBar",
    "soapbar": "SoapBar",
    "soap bar": "SoapBar",
    "bar of soap": "SoapBar",
    "butterknife": "ButterKnife",
    "butter knife": "ButterKnife",
}


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _import_detect():
    try:
        from agents.llmdet_qwen_og import detect_evidence  # type: ignore

        return detect_evidence
    except ImportError:
        from llmdet_qwen_og import detect_evidence  # type: ignore

        return detect_evidence


def _import_augment():
    try:
        from agents.llmdet_qwen_og import augment_og_prompt  # type: ignore

        return augment_og_prompt
    except ImportError:
        from llmdet_qwen_og import augment_og_prompt  # type: ignore

        return augment_og_prompt


def remap_no_det_mode() -> str:
    v = os.environ.get("ROBOAGENT_OG_REMAP_NO_DET", "false").strip().lower()
    if v in ("qwen", "full", "cascade"):
        return "qwen"
    return "false"


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def safe_synonym_label(label: str, target_obj: str = "") -> str:
    """Apply only safe ALFRED synonyms; never rewrite arbitrary observed ids."""
    if not _flag("ROBOAGENT_OG_SYNONYM", True):
        return label if label else label
    if not label:
        return label
    raw = str(label).strip()
    # Keep instance ids untouched (Drawer 1, CellPhone 2, …)
    if re.match(r"^.+\s+\d+$", raw):
        return raw
    soft = (soft_type(raw) or raw).lower().strip()
    # Query-conditioned: only upgrade when query family matches
    q = (soft_type(target_obj) or target_obj or "").lower()
    syn = _SAFE_SYNONYM.get(soft) or _SAFE_SYNONYM.get(_compact(soft))
    if not syn:
        return raw
    # Don't remap unrelated labels (e.g. random 'key' substring elsewhere)
    q_hit = any(k in q or k in soft for k in ("phone", "key", "soap", "butter"))
    if not q_hit and soft not in _SAFE_SYNONYM:
        return raw
    return syn


def refine_fix1_label(target_obj: str, adapter_label: Optional[str]) -> str:
    """v3 Fix1: keep adapter label; override compositional with functional sink/knife."""
    label = adapter_label or target_obj
    func = functional_canonical(target_obj) if _flag("ROBOAGENT_OG_FUNCTIONAL", True) else None
    if func:
        # Only replace when adapter gave a non-functional receptacle
        soft = (soft_type(label) or "").lower()
        if soft in ("cabinet", "stove", "stove burner", "countertop", "table", "dining table") or not soft:
            label = func
        elif _compact(soft) != _compact(func):
            # Prefer function for cleaning/slicing always
            label = func
    return safe_synonym_label(label, target_obj)


def remap_only_prompt(base_prompt: str, evidence: dict, fix1_label: Optional[str]) -> str:
    score = evidence.get("score")
    box = evidence.get("box")
    det_q = evidence.get("det_query")
    lg = evidence.get("last_goto")
    score_s = "none" if score is None else f"{float(score):.3f}"
    box_s = "none" if not box else "[" + ", ".join(f"{float(x):.1f}" for x in box) + "]"
    hint = (
        "[REMAP-ONLY — detector FOUND the target; YOU MUST output a canonical "
        "object class label. Do NOT answer no. Do NOT abstain.]\n"
        f"detector_found=True detector_score={score_s} "
        f"detector_box={box_s} det_query={det_q!s} last_goto={lg!s}\n"
        f"fix1_fallback_label={fix1_label!s}\n"
        "Primary evidence is the image + detector box. Reply in the required "
        "format with a canonical object label only "
        "(```json [{\"label\": \"...\"}] ```). Never reply with bare 'no'.\n"
        "Prefer ALFRED names when obvious (CellPhone, KeyChain, SoapBar, "
        "ButterKnife). Do not echo a furniture receptacle when the query is "
        "a small object.\n\n"
    )
    return hint + (base_prompt or "")


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

    m = re.search(r"```json\s*(.*?)\s*```", res, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return False
    m2 = re.search(r'"label"\s*:\s*"([^"]+)"', res)
    if m2:
        return [{"label": m2.group(1)}]
    return False


def _valid_found(ret: Any) -> bool:
    return (
        isinstance(ret, list)
        and bool(ret)
        and isinstance(ret[0], dict)
        and "label" in ret[0]
        and ret[0]["label"]
    )


def _attach_det(ret: list, evidence: dict) -> list:
    out = [dict(ret[0])]
    if evidence.get("box") is not None:
        out[0].setdefault("box", evidence["box"])
    if evidence.get("score") is not None:
        out[0].setdefault("score", evidence.get("score"))
    out[0].setdefault("detector_found", evidence.get("detector_found"))
    return out


def _fix1_fallback_ret(fix1_label: str, evidence: dict, meta: dict, path: str) -> Tuple[Any, dict]:
    meta["path"] = path
    meta["canonical_label"] = fix1_label
    meta["qwen_found"] = False
    meta["fallback"] = "fix1_label"
    skill = [
        {
            "label": fix1_label,
            "score": evidence.get("score"),
            "box": evidence.get("box"),
            "detector_found": evidence.get("detector_found"),
            "query": evidence.get("query"),
        }
    ]
    return skill, meta


def _apply_safe_label(ret: list, target_obj: str, meta: dict) -> list:
    lab = ret[0]["label"]
    lab2 = safe_synonym_label(lab, target_obj)
    if lab2 != lab:
        meta["label_synonym_from"] = lab
        ret = [dict(ret[0])]
        ret[0]["label"] = lab2
    meta["canonical_label"] = ret[0]["label"]
    return ret


def ground_remap_only(
    image_path: str,
    target_obj: str,
    base_prompt: str,
    qwen_infer: Callable[[str], str],
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
) -> Tuple[Any, dict]:
    """Full remap-only cascade → (False|list, meta). v3 = v1 + safe synonyms."""
    detect_evidence = _import_detect()
    augment_og_prompt = _import_augment()

    if threshold is None:
        threshold = float(os.environ.get("ROBOAGENT_LLMDET_THRESHOLD", "0.35"))

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
    fix1_label = refine_fix1_label(target_obj, adapter.label)
    meta["phase"] = "remap_only"
    meta["remap_v3"] = True
    meta["stage_b"] = "contract_adapter_fix1"
    meta["adapter_path"] = adapter.path
    meta["adapter_label"] = adapter.label
    meta["fix1_fallback_label"] = fix1_label
    meta["threshold"] = threshold

    found = bool(evidence.get("detector_found"))
    score = evidence.get("score")
    try:
        score_f = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_f = None

    # ----- no dets -----
    if not found or score_f is None:
        mode = remap_no_det_mode()
        meta["no_det_mode"] = mode
        if mode == "false":
            meta["path"] = "no_det_false"
            meta["reject"] = "no_dets"
            meta["qwen_found"] = None
            meta["stage_c"] = "skipped"
            _empty_cuda_cache()
            return False, meta
        aug = augment_og_prompt(base_prompt, evidence)
        meta["prompt_augmented"] = True
        meta["stage_c"] = "qwen_full_no_det"
        t1 = time.time()
        res = qwen_infer(aug)
        if res is None:
            res = "no"
        res = str(res).strip()
        meta["stage_c_ms"] = round((time.time() - t1) * 1000.0, 2)
        meta["qwen_raw"] = res[:2000]
        ret = _parse_qwen(res, parse_fn, meta)
        if not _valid_found(ret):
            meta["path"] = "no_det_qwen_abstain"
            meta["qwen_found"] = False
            meta["reject"] = "qwen_abstain"
            _empty_cuda_cache()
            return False, meta
        meta["path"] = "no_det_qwen_found"
        meta["qwen_found"] = True
        ret = _apply_safe_label(ret, target_obj, meta)
        ret = _attach_det(ret, evidence)
        _empty_cuda_cache()
        return ret, meta

    # ----- high-conf: remap-only (never False) -----
    if score_f >= float(threshold):
        aug = remap_only_prompt(base_prompt, evidence, fix1_label)
        meta["prompt_augmented"] = True
        meta["stage_c"] = "qwen_remap_only"
        meta["remap_forbid_false"] = True
        t1 = time.time()
        res = qwen_infer(aug)
        if res is None:
            res = "no"
        res = str(res).strip()
        meta["stage_c_ms"] = round((time.time() - t1) * 1000.0, 2)
        meta["qwen_raw"] = res[:2000]

        ret = _parse_qwen(res, parse_fn, meta)
        if _valid_found(ret):
            meta["path"] = "remap_qwen_label"
            meta["qwen_found"] = True
            ret = _apply_safe_label(ret, target_obj, meta)
            ret = _attach_det(ret, evidence)
            _empty_cuda_cache()
            return ret, meta

        # Ablations: allow veto (gated-like) or drop fallback
        if not _flag("ROBOAGENT_OG_NO_VETO", True):
            meta["path"] = "remap_veto_allowed"
            meta["qwen_found"] = False
            meta["reject"] = "qwen_abstain_veto"
            _empty_cuda_cache()
            return False, meta
        if not _flag("ROBOAGENT_OG_FALLBACK", True):
            meta["path"] = "remap_no_fallback"
            meta["qwen_found"] = False
            _empty_cuda_cache()
            return False, meta

        ret, meta = _fix1_fallback_ret(
            fix1_label, evidence, meta, path="remap_fix1_fallback"
        )
        ret = _apply_safe_label(ret, target_obj, meta)
        _empty_cuda_cache()
        return ret, meta

    # ----- low-conf dets: full Qwen OG may abstain -----
    aug = augment_og_prompt(base_prompt, evidence)
    if fix1_label:
        aug = (
            f"[Adapter candidate label={fix1_label!s} path={adapter.path} "
            f"— auxiliary; YOU decide final answer including no]\n"
            + aug
        )
    meta["prompt_augmented"] = True
    meta["stage_c"] = "qwen_full_low_score"
    meta["remap_forbid_false"] = False
    t1 = time.time()
    res = qwen_infer(aug)
    if res is None:
        res = "no"
    res = str(res).strip()
    meta["stage_c_ms"] = round((time.time() - t1) * 1000.0, 2)
    meta["qwen_raw"] = res[:2000]

    ret = _parse_qwen(res, parse_fn, meta)
    if not _valid_found(ret):
        meta["path"] = "low_score_qwen_abstain"
        meta["qwen_found"] = False
        meta["reject"] = "qwen_abstain"
        _empty_cuda_cache()
        return False, meta

    meta["path"] = "low_score_qwen_label"
    meta["qwen_found"] = True
    ret = _apply_safe_label(ret, target_obj, meta)
    ret = _attach_det(ret, evidence)
    _empty_cuda_cache()
    return ret, meta


ground = ground_remap_only
ground_cascade_remap = ground_remap_only
