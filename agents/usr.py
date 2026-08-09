"""Unified Skill Representation (USR) — v2: full remap-v3 alignment in the OG Adapter.

Stage-0 requirement: the OG Adapter must reproduce remap v3's decision EXACTLY
(canonicalization, functional override, no-veto, found, fallback), so USR does
NOT degrade AW/EB. Field separation is strict:
  - environment_facts / task_semantics: facts (what is true)
  - decision_signals: confidence / uncertainty / found / fallback_reason (how to decide)
  - provenance / temporal: audit metadata (NOT decision inputs)
Model-private artifacts (det_query raw string, raw bbox) are NOT stored in public fields.
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Reuse remap v3 alignment primitives
try:
    from agents.og_remap_only import (  # type: ignore
        safe_synonym_label,
        refine_fix1_label,
        remap_no_det_mode,
    )
    from agents.contract_adapter import (  # type: ignore
        adapt_label,
        functional_canonical,
        soft_type,
        type_match,
        normalize_query,
    )
except ImportError:
    from og_remap_only import (  # type: ignore
        safe_synonym_label,
        refine_fix1_label,
        remap_no_det_mode,
    )
    from contract_adapter import (  # type: ignore
        adapt_label,
        functional_canonical,
        soft_type,
        type_match,
        normalize_query,
    )


# ---------- USR construction ----------

def make_usr(
    *,
    skill: str,
    object_class: Optional[str] = None,
    object_state: Optional[List[str]] = None,
    relation: Optional[Dict[str, str]] = None,
    location: Optional[Dict[str, Any]] = None,
    subgoal: Optional[str] = None,
    role: Optional[str] = None,
    task_type: Optional[str] = None,
    confidence: Optional[float] = None,
    uncertainty_level: Optional[str] = None,
    uncertainty_reason: Optional[str] = None,
    found: Optional[bool] = None,
    fallback_reason: Optional[str] = None,
    detector: Optional[str] = None,
    alignment_path: Optional[str] = None,
    episode_step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    usr: Dict[str, Any] = {
        "environment_facts": {},
        "task_semantics": {},
        "decision_signals": {},
        "provenance": {"skill": skill},
        "temporal_context": {},
        "schema_version": "2.0",
    }
    if object_class:
        usr["environment_facts"]["object"] = {"class": object_class}
        if object_state:
            usr["environment_facts"]["object"]["state"] = object_state
    if relation:
        usr["environment_facts"]["relation"] = relation
    if location:
        usr["environment_facts"]["location"] = location

    if subgoal:
        usr["task_semantics"]["subgoal"] = subgoal
    if role:
        usr["task_semantics"]["role"] = role
    if task_type:
        usr["task_semantics"]["task_type"] = task_type

    ds = usr["decision_signals"]
    if confidence is not None:
        ds["confidence"] = confidence
    if uncertainty_level:
        ds["uncertainty"] = {"level": uncertainty_level}
        if uncertainty_reason:
            ds["uncertainty"]["reason"] = uncertainty_reason
    if found is not None:
        ds["found"] = found
    if fallback_reason is not None:
        ds["fallback_reason"] = fallback_reason

    if detector:
        usr["provenance"]["detector"] = detector
    if alignment_path:
        usr["provenance"]["alignment_path"] = alignment_path
    if episode_step is not None:
        usr["temporal_context"]["episode_step"] = episode_step
    if extra:
        usr["skill_specific"] = extra
    return usr


# ---------- Field access ----------

def usr_object_class(usr: Dict[str, Any]) -> Optional[str]:
    return usr.get("environment_facts", {}).get("object", {}).get("class")


def usr_found(usr: Dict[str, Any]) -> Optional[bool]:
    return usr.get("decision_signals", {}).get("found")


def usr_confidence(usr: Dict[str, Any]) -> Optional[float]:
    return usr.get("decision_signals", {}).get("confidence")


def brain_decision_key(usr: Dict[str, Any]) -> str:
    """Canonical decision key = object.class + location target (normalized)."""
    obj = usr_object_class(usr) or ""
    loc = usr.get("environment_facts", {}).get("location", {}).get("target") or ""
    return re.sub(r"[^a-z0-9@]", "", f"{obj}@{loc}".lower())


# ---------- Field ablation ----------

def ablate_usr(usr: Dict[str, Any], field: str) -> Dict[str, Any]:
    u = copy.deepcopy(usr)
    ef = u.get("environment_facts", {})
    if field == "relation":
        ef.pop("relation", None)
    elif field == "confidence":
        u.get("decision_signals", {}).pop("confidence", None)
    elif field == "temporal":
        u.pop("temporal_context", None)
    elif field == "provenance":
        u.pop("provenance", None)
    elif field == "location":
        ef.pop("location", None)
    elif field == "state":
        obj = ef.get("object", {})
        obj.pop("state", None)
    elif field == "fallback_reason":
        u.get("decision_signals", {}).pop("fallback_reason", None)
    return u


# ---------- OG Adapter with FULL remap-v3 logic ----------

def og_adapter_full(
    *,
    target_obj: str,
    det_found: bool,
    det_score: Optional[float],
    det_query: Optional[str],
    last_goto: Optional[str] = None,
    qwen_label: Optional[str] = None,   # Qwen remap output label (if available)
    observed_objects: Optional[Sequence[str]] = None,
    env_name: str = "eb-alfred",
    threshold: float = 0.35,
    episode_step: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reproduce remap v3 decision, emit USR.

    Returns (usr, meta). meta holds model-private / debugging fields (NOT in usr).
    """
    meta: Dict[str, Any] = {
        "det_query": det_query,
        "det_found_raw": det_found,
        "det_score_raw": det_score,
        "target": target_obj,
    }

    # ---- no-det path: found=False (allows exploration), with fallback reason ----
    if not det_found or det_score is None:
        mode = remap_no_det_mode()
        if mode == "false":
            usr = make_usr(
                skill="object_grounding", found=False,
                uncertainty_level="high", uncertainty_reason="no_detection",
                fallback_reason="no_det_false",
                detector=meta.get("detector", "unknown"),
                alignment_path="no_det_false", episode_step=episode_step,
            )
            meta["path"] = "no_det_false"
            return usr, meta
        # qwen full no-det (rare in v3 default)
        usr = make_usr(
            skill="object_grounding", found=False,
            uncertainty_level="high", uncertainty_reason="qwen_abstain",
            fallback_reason="no_det_qwen_abstain",
            detector=meta.get("detector", "unknown"),
            alignment_path="no_det_qwen_abstain", episode_step=episode_step,
        )
        meta["path"] = "no_det_qwen_abstain"
        return usr, meta

    # ---- high-conf path (det_found, score >= thr): remap-only, no-veto ----
    if det_score >= threshold:
        # canonical label via full v3 pipeline — SAME as v3:
        # final label = Qwen label (safe-synonym only), else fix1 fallback.
        # NO canonicalize (v3 keeps case, digits, 'sliced' prefix exactly).
        adapter = adapt_label(
            target_obj, last_goto=last_goto,
            detector_found=True, det_score=det_score, det_label=det_query,
        )
        fix1_label = refine_fix1_label(target_obj, adapter.label)
        if qwen_label:
            final_label = safe_synonym_label(qwen_label, target_obj)
            path = "remap_qwen_label"
        else:
            final_label = fix1_label  # refine_fix1 already applied safe_synonym
            path = "remap_fix1_fallback"

        usr = make_usr(
            skill="object_grounding",
            object_class=final_label,
            location={"receptacle": last_goto} if last_goto else None,
            confidence=det_score,
            uncertainty_level="low" if det_score >= 0.5 else "medium",
            uncertainty_reason="high_conf_no_veto",
            found=True,
            fallback_reason=None if qwen_label else "remap_fix1_fallback",
            detector=meta.get("detector", "unknown"),
            alignment_path=path,
            episode_step=episode_step,
        )
        meta["label"] = final_label
        meta["fix1_label"] = fix1_label
        meta["path"] = path
        return usr, meta

    # ---- low-conf path: Qwen may abstain (rare; keep found=False) ----
    usr = make_usr(
        skill="object_grounding", found=False,
        confidence=det_score,
        uncertainty_level="high", uncertainty_reason="low_score",
        fallback_reason="low_score_abstain",
        detector=meta.get("detector", "unknown"),
        alignment_path="low_score_abstain", episode_step=episode_step,
    )
    meta["path"] = "low_score_abstain"
    return usr, meta


# ---------- SD / EG adapters ----------

def sd_adapter_to_usr(
    *, description: str, confidence: Optional[float] = None,
    detector: str = "florence2", alignment_path: str = "naive",
) -> Dict[str, Any]:
    return make_usr(
        skill="scene_description", confidence=confidence,
        detector=detector, alignment_path=alignment_path,
        extra={"description": description},
    )


def eg_adapter_to_usr(
    *, direction: str, detector: str = "validated_ft",
    alignment_path: str = "explicit_validator",
) -> Dict[str, Any]:
    m = re.match(r"^(in|on|target)\s+(.+)$", direction, re.I)
    relation = {"type": m.group(1)} if m else None
    location = {"target": m.group(2).strip()} if m else None
    return make_usr(
        skill="exploration_guidance", relation=relation, location=location,
        confidence=1.0 if m else 0.0,
        uncertainty_level="low" if m else "high",
        uncertainty_reason="validated" if m else "invalid_direction",
        detector=detector, alignment_path=alignment_path,
        extra={"direction": direction},
    )
