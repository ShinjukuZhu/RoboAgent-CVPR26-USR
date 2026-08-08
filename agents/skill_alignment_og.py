"""Skill-Aligned OG: clean remap-only (v1) + data-driven canonical label adapter.

Based on remap v1 principles (AW 86.7 / EB 80, verified):
  - high-conf det NEVER returns False (no veto)  → keeps AW presence
  - Qwen remaps label on high-conf; Fix1 fallback if Qwen abstains
  - no_det → False (allows exploration)

Adds (Phase 1 Skill Alignment):
  - canonicalize_label with data-driven vocabulary (skill_alignment.py)
    → fixes EB label contract (phone→CellPhone, key→KeyChain, cd→CD, teapot→Kettle)
  - query-family drift guard (never drift to wrong family)

Removes (v2 harmful paths):
  - no instance_nav_trust / low_score_instance_trust (forced found)
  - no remap_instance_direct skipping Qwen
  - no hardcoded-synonym-table canonicalization (use data-driven vocab)

Env:
  ROBOAGENT_OG_BACKEND=llmdet_qwen_aligned
  ROBOAGENT_LLMDET_THRESHOLD=0.35
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Optional, Sequence, Tuple

try:
    from agents.skill_alignment import canonicalize_label, query_family_ok  # type: ignore
except ImportError:
    from skill_alignment import canonicalize_label, query_family_ok  # type: ignore

try:
    from agents.contract_adapter import (  # type: ignore
        adapt_label, soft_type, type_match, normalize_query,
    )
except ImportError:
    from contract_adapter import (
        adapt_label, soft_type, type_match, normalize_query,
    )


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


def _parse_qwen(res: str, parse_fn: Optional[Callable[[str], Any]], meta: dict) -> Any:
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
    )


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
        "Prefer ALFRED names (CellPhone, KeyChain, SoapBar, ButterKnife, Sofa, "
        "CD, Sink, Fridge). Keep query family: teapot≠kettle; keys≠single key; "
        "bar of soap→SoapBar; phone→CellPhone. Do not echo a furniture receptacle "
        "when the query is a small object.\n\n"
    )
    return hint + (base_prompt or "")


def _finalize_found(ret: list, path: str, qwen_found: Optional[bool],
                    target_obj: str, last_goto: Optional[str],
                    observed_objects: Optional[Sequence[str]], meta: dict,
                    env_name: str) -> Tuple[Any, dict]:
    """Finalize a found label; apply data-driven canonicalization."""
    lab = ret[0]["label"]
    canon = canonicalize_label(
        lab, query=target_obj, last_goto=last_goto,
        observed_objects=observed_objects, env_name=env_name,
    )
    if canon != lab:
        meta["label_aligned_from"] = lab
        lab = canon
        ret = [dict(ret[0])]
        ret[0]["label"] = lab
    # Drift guard: if canonicalization couldn't fix a family break, and query
    # family conflicts, fall back to fix1 candidate (handled by caller).
    meta["path"] = path
    meta["canonical_label"] = lab
    meta["qwen_found"] = qwen_found
    if evidence_box := meta.get("_det_evidence"):
        if evidence_box.get("box") is not None:
            ret = [dict(ret[0])]
            ret[0]["box"] = evidence_box["box"]
            ret[0]["score"] = evidence_box.get("score")
            ret[0]["detector_found"] = evidence_box.get("detector_found")
    return ret, meta


def _fix1_fallback_ret(fix1_label: str, evidence: dict, meta: dict, path: str,
                       target_obj: str, last_goto: Optional[str],
                       observed_objects: Optional[Sequence[str]],
                       env_name: str) -> Tuple[Any, dict]:
    canon = canonicalize_label(
        fix1_label, query=target_obj, last_goto=last_goto,
        observed_objects=observed_objects, env_name=env_name,
    )
    meta["path"] = path
    meta["canonical_label"] = canon
    meta["qwen_found"] = False
    meta["fallback"] = "fix1_label"
    if canon != fix1_label:
        meta["label_aligned_from"] = fix1_label
    skill = [{
        "label": canon,
        "score": evidence.get("score"),
        "box": evidence.get("box"),
        "detector_found": evidence.get("detector_found"),
        "query": evidence.get("query"),
    }]
    return skill, meta


def ground_aligned(
    image_path: str,
    target_obj: str,
    base_prompt: str,
    qwen_infer: Callable[[str], str],
    threshold: Optional[float] = None,
    last_goto: Optional[str] = None,
    observed_objects: Optional[Sequence[str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
    env_name: str = "alfworld",
) -> Tuple[Any, dict]:
    """Clean remap-only (v1) + data-driven canonical label alignment."""
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
    fix1_label = adapter.label or target_obj
    meta["phase"] = "skill_alignment"
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

    meta["_det_evidence"] = evidence

    # ----- no dets -----
    if not found or score_f is None:
        mode = os.environ.get("ROBOAGENT_OG_REMAP_NO_DET", "false").strip().lower()
        meta["no_det_mode"] = mode
        if mode in ("qwen", "full", "cascade"):
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
                _empty_cuda_cache()
                return False, meta
            ret, meta = _finalize_found(ret, "no_det_qwen_found", True,
                                        target_obj, last_goto, observed_objects,
                                        meta, env_name)
            _empty_cuda_cache()
            return ret, meta
        meta["path"] = "no_det_false"
        meta["reject"] = "no_dets"
        meta["qwen_found"] = None
        meta["stage_c"] = "skipped"
        _empty_cuda_cache()
        return False, meta

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
            ret, meta = _finalize_found(ret, "remap_qwen_label", True,
                                        target_obj, last_goto, observed_objects,
                                        meta, env_name)
            # drift guard: if aligned label still breaks query family, fallback
            if not query_family_ok(target_obj, meta["canonical_label"]):
                meta["drift_rejected"] = meta["canonical_label"]
                ret, meta = _fix1_fallback_ret(
                    fix1_label, evidence, meta, "remap_fix1_fallback",
                    target_obj, last_goto, observed_objects, env_name)
            _empty_cuda_cache()
            return ret, meta

        ret, meta = _fix1_fallback_ret(
            fix1_label, evidence, meta, "remap_fix1_fallback",
            target_obj, last_goto, observed_objects, env_name)
        _empty_cuda_cache()
        return ret, meta

    # ----- low-conf: Qwen may abstain -----
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

    ret, meta = _finalize_found(ret, "low_score_qwen_label", True,
                                target_obj, last_goto, observed_objects,
                                meta, env_name)
    _empty_cuda_cache()
    return ret, meta


# Alias for agent wiring
ground = ground_aligned
