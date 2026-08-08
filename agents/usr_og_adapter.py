"""USR Adapter for OG — migrates remap v3 alignment logic into USR.

Maps Foundation Model (LLMDet/GDINO) raw output → USR, using the SAME alignment
logic as remap v3 (safe_synonym + functional override + no-veto + found + fallback).

STRICT separation:
  - environment_facts: object.class (canonical), state, relation, location
  - decision_signals: confidence, uncertainty, found, fallback_reason
  - provenance: detector, alignment_path, threshold
  - model-specific artifacts (det_query, raw box) are STRIPPED from public USR
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from agents.contract_adapter import (  # type: ignore
        adapt_label, functional_canonical, soft_type, normalize_query,
    )
except ImportError:
    from contract_adapter import (  # type: ignore
        adapt_label, functional_canonical, soft_type, normalize_query,
    )

try:
    from agents.skill_alignment import canonicalize_label  # type: ignore
except ImportError:
    from skill_alignment import canonicalize_label  # type: ignore

# Same safe synonyms as remap v3
_SAFE_SYNONYM = {
    "phone": "CellPhone", "cellphone": "CellPhone", "cell phone": "CellPhone",
    "key": "KeyChain", "keys": "KeyChain", "keychain": "KeyChain",
    "key chain": "KeyChain", "set of keys": "KeyChain",
    "soap": "SoapBar", "soapbar": "SoapBar", "soap bar": "SoapBar",
    "bar of soap": "SoapBar",
    "butterknife": "ButterKnife", "butter knife": "ButterKnife",
}


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _safe_synonym_label(label: str, target_obj: str = "") -> str:
    """Same as remap v3 safe_synonym_label."""
    if not label:
        return label
    raw = str(label).strip()
    if re.match(r"^.+\s+\d+$", raw):  # keep instance ids
        return raw
    soft = (soft_type(raw) or raw).lower().strip()
    q = (soft_type(target_obj) or target_obj or "").lower()
    syn = _SAFE_SYNONYM.get(soft) or _SAFE_SYNONYM.get(_compact(soft))
    if not syn:
        return raw
    q_hit = any(k in q or k in soft for k in ("phone", "key", "soap", "butter"))
    if not q_hit and soft not in _SAFE_SYNONYM:
        return raw
    return syn


def _refine_fix1_label(target_obj: str, adapter_label: Optional[str]) -> str:
    """Same as remap v3 refine_fix1_label (functional override)."""
    label = adapter_label or target_obj
    func = functional_canonical(target_obj)
    if func:
        soft = (soft_type(label) or "").lower()
        if soft in ("cabinet", "stove", "stove burner", "countertop", "table", "dining table") or not soft:
            label = func
        elif _compact(soft) != _compact(func):
            label = func
    return _safe_synonym_label(label, target_obj)


def build_og_usr(
    *,
    target_obj: str,
    det_found: bool,
    det_score: Optional[float],
    det_label: Optional[str],
    det_query: Optional[str],
    last_goto: Optional[str],
    threshold: float = 0.35,
    detector: str = "llmdet",
    env_name: str = "eb-alfred",
    no_veto: bool = True,
    episode_step: Optional[int] = None,
    # v3-grounded reconstruction (from trace meta) — authoritative
    v3_path: Optional[str] = None,
    v3_canonical_label: Optional[str] = None,
    v3_fix1_label: Optional[str] = None,
    v3_qwen_found: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build USR from detector raw output, applying remap v3 alignment.

    STRICT separation:
      - environment_facts.object.class = v3 canonical_label (Brain's actual input)
      - decision_signals = found/confidence/uncertainty/fallback_reason
      - provenance = detector/path/threshold
      - NO model-private fields (det_query, raw box) in public facts

    If v3_* reconstruction fields are given, they are authoritative (this is how
    the USR matches remap v3 exactly). Otherwise decision logic replicates v3:
      1. no_det (score < thr or not found) -> found=False, fallback="no_det"
      2. high-conf -> found=True, object.class=fix1 canonical, no-veto
    """
    adapter = adapt_label(
        target_obj, last_goto=last_goto,
        detector_found=bool(det_found), det_score=det_score, det_label=det_label,
    )
    fix1_label = _refine_fix1_label(target_obj, adapter.label)
    score_f = float(det_score) if det_score is not None else None

    usr: Dict[str, Any] = {
        "environment_facts": {},
        "task_semantics": {},
        "decision_signals": {},
        "provenance": {
            "skill": "object_grounding",
            "detector": detector,
            "alignment_path": v3_path or "",
            "threshold": threshold,
        },
        "temporal_context": {},
    }

    # ----- authoritative reconstruction (matches v3 exactly) -----
    if v3_path is not None:
        found = v3_canonical_label is not None and bool(v3_canonical_label)
        if v3_path == "no_det_false":
            found = False
        usr["provenance"]["alignment_path"] = v3_path
        if v3_fix1_label is not None:
            usr["provenance"]["fix1_label"] = v3_fix1_label
        if v3_canonical_label:
            usr["environment_facts"]["object"] = {"class": v3_canonical_label}
        usr["decision_signals"] = {
            "found": found,
            "confidence": score_f,
            "fallback_reason": (
                "no_det" if v3_path == "no_det_false"
                else ("fix1_fallback" if v3_path == "remap_fix1_fallback" else "none")
            ),
            "uncertainty": {
                "level": ("high" if v3_path == "no_det_false"
                          else ("low" if (score_f or 0) >= 0.5 else "medium")),
                "reason": (
                    "no_detection" if v3_path == "no_det_false"
                    else ("high_conf" if (score_f or 0) >= 0.5 else "low_score")
                ),
            },
        }
        if episode_step is not None:
            usr["temporal_context"]["episode_step"] = episode_step
        return usr

    # ----- standalone mode: replicate v3 decision logic -----
    if (not det_found) or score_f is None or score_f < threshold:
        usr["decision_signals"] = {
            "found": False,
            "fallback_reason": "no_det",
            "confidence": score_f,
            "uncertainty": {"level": "high", "reason": "no_detection"},
        }
        usr["provenance"]["alignment_path"] = "no_det_false"
        return usr

    usr["environment_facts"]["object"] = {"class": fix1_label}
    usr["decision_signals"] = {
        "found": True,
        "confidence": score_f,
        "uncertainty": {
            "level": "low" if score_f >= 0.5 else "medium",
            "reason": "high_conf" if score_f >= 0.5 else "low_score",
        },
    }
    usr["provenance"]["alignment_path"] = "remap_qwen_label"
    usr["provenance"]["fix1_label"] = fix1_label

    if episode_step is not None:
        usr["temporal_context"]["episode_step"] = episode_step

    return usr


# Brain decision key (what Brain bases action on)
def og_decision_key(usr: Dict[str, Any]) -> str:
    obj = usr.get("environment_facts", {}).get("object", {}).get("class", "")
    found = usr.get("decision_signals", {}).get("found", False)
    if not found:
        return "not_found"
    return re.sub(r"[^a-z0-9]", "", (obj or "").lower())


# Ablation helpers
def ablate_og_field(usr: Dict[str, Any], field: str) -> Dict[str, Any]:
    import copy
    u = copy.deepcopy(usr)
    if field == "object.class":
        u.get("environment_facts", {}).pop("object", None)
    elif field == "confidence":
        u.get("decision_signals", {}).pop("confidence", None)
    elif field == "uncertainty":
        u.get("decision_signals", {}).pop("uncertainty", None)
    elif field == "found":
        u.get("decision_signals", {}).pop("found", None)
    elif field == "provenance":
        u.pop("provenance", None)
    return u
