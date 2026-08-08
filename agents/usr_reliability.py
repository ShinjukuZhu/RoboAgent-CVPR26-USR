"""USR Reliability Module — schema/version/type/missing/conflict/round-trip/provenance.

Ensures USR objects are well-formed and audit-safe:
  - schema_version check (2.0)
  - type constraints (str/float/bool)
  - missing/invalid/conflict handling
  - round-trip validation (facts -> text -> facts)
  - provenance audit (never enters Brain decision)
  - status markers: observed / inferred / fallback
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "2.0"

# ---- type constraints ----

REQUIRED_SECTIONS = ["environment_facts", "task_semantics", "decision_signals", "provenance"]
OPTIONAL_SECTIONS = ["temporal_context", "skill_specific"]

TYPES = {
    ("environment_facts", "object", "class"): str,
    ("environment_facts", "object", "state"): list,
    ("environment_facts", "relation", "type"): str,
    ("environment_facts", "location", "target"): str,
    ("task_semantics", "role"): str,
    ("decision_signals", "found"): bool,
    ("decision_signals", "confidence"): float,
}


def _get(usr: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur = usr
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def validate_schema(usr: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return {section: [errors]}. Missing sections / type violations."""
    errors: Dict[str, List[str]] = {}
    if not isinstance(usr, dict):
        return {"root": ["not a dict"]}
    v = usr.get("schema_version")
    if v != SCHEMA_VERSION:
        errors.setdefault("schema_version", []).append(f"expected {SCHEMA_VERSION}, got {v!r}")
    for sec in REQUIRED_SECTIONS:
        if sec not in usr:
            errors.setdefault(sec, []).append("missing section")
    for path, typ in TYPES.items():
        val = _get(usr, path)
        if val is None:
            continue  # optional
        if not isinstance(val, typ):
            errors.setdefault(".".join(path), []).append(
                f"type {type(val).__name__} != {typ.__name__} ({val!r})")
    # confidence range
    conf = _get(usr, ("decision_signals", "confidence"))
    if conf is not None and isinstance(conf, (int, float)):
        if not (0.0 <= float(conf) <= 1.0):
            errors.setdefault("decision_signals.confidence", []).append(
                f"out of range {conf!r}")
    # found must be bool if present
    found = _get(usr, ("decision_signals", "found"))
    if found is not None and not isinstance(found, bool):
        errors.setdefault("decision_signals.found", []).append(
            f"type {type(found).__name__} != bool ({found!r})")
    return errors


# ---- missing / invalid / conflict handling ----

def sanitize(usr: Dict[str, Any]) -> Dict[str, Any]:
    """Repair invalid USR: coerce types, drop invalid confidence, resolve conflicts."""
    out = json.loads(json.dumps(usr))  # deep copy
    # confidence: clamp to [0,1]
    conf = _get(out, ("decision_signals", "confidence"))
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        out["decision_signals"]["confidence"] = max(0.0, min(1.0, float(conf)))
    elif conf is not None:
        out["decision_signals"].pop("confidence", None)
    # found: coerce truthy to bool
    found = _get(out, ("decision_signals", "found"))
    if found is not None and not isinstance(found, bool):
        out["decision_signals"]["found"] = bool(found)
    # conflict: found=false but object.class present -> mark fallback + clear class
    ef = out.get("environment_facts", {})
    obj = ef.get("object", {})
    if out.get("decision_signals", {}).get("found") is False and obj.get("class"):
        out["decision_signals"]["fallback_reason"] = "conflict_found_class_cleared"
        out["environment_facts"]["object"] = {}
    # conflict: found=true but no object.class -> mark inferred status
    if out.get("decision_signals", {}).get("found") is True and not obj.get("class"):
        out["decision_signals"]["fallback_reason"] = "found_without_class"
        out["decision_signals"]["uncertainty"] = {
            "level": "high", "reason": "found_without_class"}
    return out


# ---- status markers: observed / inferred / fallback ----

def mark_status(usr: Dict[str, Any]) -> Dict[str, Any]:
    """Add decision_signals.status in {observed, inferred, fallback}.

    observed: object.class present with found=true and confidence >= 0.5
    inferred: class present but low confidence (or no confidence)
    fallback: found=false or conflict-resolved
    """
    out = json.loads(json.dumps(usr))
    ds = out.setdefault("decision_signals", {})
    found = ds.get("found")
    conf = ds.get("confidence")
    obj = out.get("environment_facts", {}).get("object", {}).get("class")
    if found is False:
        ds["status"] = "fallback"
        ds.setdefault("fallback_reason", "no_detection")
    elif found is True and obj:
        ds["status"] = "observed" if (conf is None or conf >= 0.5) else "inferred"
    elif found is True and not obj:
        ds["status"] = "fallback"
    else:
        ds["status"] = "inferred"
    return out


# ---- round-trip validation ----

def roundtrip_check(usr: Dict[str, Any]) -> Dict[str, List[str]]:
    """Verify facts survive serialization round-trip and class/location readable."""
    errors: Dict[str, List[str]] = {}
    try:
        serialized = json.dumps(usr)
        parsed = json.loads(serialized)
    except Exception as e:
        return {"roundtrip": [f"serialize failed: {e}"]}
    if parsed != usr:
        errors["roundtrip"] = ["json round-trip mismatch"]
    # class readability
    obj_class = _get(usr, ("environment_facts", "object", "class"))
    if obj_class is not None and not isinstance(obj_class, str):
        errors["readable"] = ["object.class not str"]
    # location target readable
    loc = _get(usr, ("environment_facts", "location", "target"))
    if loc is not None and not isinstance(loc, str):
        errors["readable"] = ["location.target not str"]
    return errors


# ---- provenance audit (never enters decision) ----

PROVENANCE_KEYWORDS = ["detector", "alignment_path", "model_id", "threshold",
                       "latency_ms", "det_query", "raw", "qwen_raw", "box", "bbox"]


def audit_provenance(usr: Dict[str, Any]) -> Dict[str, List[str]]:
    """Ensure model-private fields stay in provenance; not in facts/signals."""
    errors: Dict[str, List[str]] = {}
    protected = ["environment_facts", "task_semantics", "decision_signals"]
    for sec in protected:
        blk = json.dumps(usr.get(sec, {}))
        # exact key match (not substring) to avoid false positives like 'drawer'~'raw'
        try:
            keys = json.loads(blk).keys() if isinstance(json.loads(blk), dict) else []
        except Exception:
            keys = []
        for kw in PROVENANCE_KEYWORDS:
            if kw in keys or f'"{kw}"' in blk:
                errors.setdefault(sec, []).append(f"model-private '{kw}' leaked")
    # provenance itself is allowed but must not be consumed as decision input
    prov = usr.get("provenance", {})
    if not isinstance(prov, dict):
        errors["provenance"] = ["not a dict"]
    return errors


# ---- full validation pipeline ----

def validate_usr(usr: Dict[str, Any]) -> Dict[str, Any]:
    """Full reliability pipeline. Returns {ok, errors, status, sanitized}."""
    errors: Dict[str, List[str]] = {}
    errors.update(validate_schema(usr))
    errors.update(roundtrip_check(usr))
    errors.update(audit_provenance(usr))
    sanitized = sanitize(usr)
    sanitized = mark_status(sanitized)
    # schema_version preserved
    sanitized["schema_version"] = SCHEMA_VERSION
    ok = not errors
    return {"ok": ok, "errors": errors, "sanitized": sanitized,
            "status": sanitized.get("decision_signals", {}).get("status")}
