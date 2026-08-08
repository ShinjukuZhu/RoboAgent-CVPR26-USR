"""USR adapters for SD and EG — emit USR, keep Brain-facing text equivalent.

Goal (gate 2): USR must not damage Soft (SD) / Explicit (EG) Contracts. Strategy:
  - Parse raw FM output into USR structure (environment_facts / task_semantics /
    decision_signals), with STRICT separation (no model-private artifacts).
  - Reconstruct a Brain-facing TEXT string from USR that is decision-equivalent
    to the raw interface, so existing Brain prompt consumption is unchanged.

SD (Soft Contract): free-text scene description. USR extracts the described
  object -> location/relation facts; text reconstruction re-renders them.

EG (Explicit Contract): "<rel> <ReceptacleId>" direction. USR puts rel into
  task_semantics.role and receptacle into environment_facts.location.target;
  text reconstruction is lossless (same string).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---- shared USR schema helpers ----

SCHEMA_VERSION = "2.0"


def _base_usr(skill: str, detector: str, alignment_path: str) -> Dict[str, Any]:
    return {
        "environment_facts": {},
        "task_semantics": {},
        "decision_signals": {},
        "provenance": {
            "skill": skill,
            "detector": detector,
            "alignment_path": alignment_path,
            "schema_version": SCHEMA_VERSION,
        },
        "temporal_context": {},
    }


# ---- SD (Scene Description) ----

# pattern: "<object> <prep> <location>"  (a/some/the/NOUN optional)
_SD_LOC_RE = re.compile(
    r"(?i)(?:there (?:is|are)\s+|i see\s+|there\s+)?"
    r"(?P<obj>a |an |the |some )?(?P<objname>[\w\- ]+?)\s+"
    r"(?P<rel>on|in|under|behind|next to|near|at|by|on top of)\s+"
    r"(?:a |an |the |some )?(?P<loc>[\w\- ]+)"
)


def sd_raw_to_usr(
    raw_text: str,
    detector: str = "qwen25",
    target: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    usr = _base_usr("scene_description", detector, "raw_text_parse")
    usr["decision_signals"]["confidence"] = confidence
    # parse object->location relations into environment_facts
    rels: List[Dict[str, str]] = []
    text = raw_text.strip()
    for m in _SD_LOC_RE.finditer(text):
        objname = m.group("objname").strip()
        rel = m.group("rel").strip()
        loc = m.group("loc").strip()
        if not objname or not loc:
            continue
        rels.append({"object": objname, "type": rel, "target": loc})
    if rels:
        usr["environment_facts"]["relations"] = rels
    else:
        usr["environment_facts"]["description"] = text
    # decision signal: parse quality
    usr["decision_signals"]["found"] = bool(rels)
    usr["decision_signals"]["uncertainty"] = (
        {"level": "low", "reason": "parsed_relations"}
        if rels else {"level": "medium", "reason": "no_relations_parsed"}
    )
    usr["temporal_context"]["target"] = target
    return usr


def sd_usr_to_text(usr: Dict[str, Any]) -> str:
    """Reconstruct Brain-facing text from USR (decision-equivalent)."""
    ef = usr.get("environment_facts", {})
    rels = ef.get("relations")
    if rels:
        parts = []
        for r in rels:
            parts.append(f"{r['object']} is {r['type']} the {r['target']}")
        return "There are " + ", ".join(parts) + "."
    desc = ef.get("description")
    if desc:
        return desc
    return ""


# ---- EG (Exploration Guidance) ----

_EG_RE = re.compile(r"^(in|on|target)\s+(.+)$", re.I)


def eg_raw_to_usr(
    direction: str,
    detector: str = "qwen25",
    observed_objects: Optional[Sequence[str]] = None,
    validated: bool = False,
) -> Dict[str, Any]:
    """Parse "<rel> <ReceptacleId>" -> USR. If invalid, keep raw in decision."""
    usr = _base_usr("exploration_guidance", detector, "validated" if validated else "raw_parse")
    raw = str(direction).strip()
    m = _EG_RE.match(raw)
    if m:
        rel = m.group(1).lower()
        loc = m.group(2).strip()
        usr["task_semantics"]["role"] = rel
        usr["environment_facts"]["location"] = {"target": loc}
        usr["decision_signals"]["found"] = True
        usr["decision_signals"]["confidence"] = 1.0 if validated else 0.5
        usr["decision_signals"]["uncertainty"] = {
            "level": "low" if validated else "medium",
            "reason": "validated" if validated else "raw_parsed",
        }
        # observed check is decision signal, not fact
        if observed_objects is not None:
            in_observed = loc in set(observed_objects or [])
            usr["decision_signals"]["observed_match"] = in_observed
            usr["decision_signals"]["uncertainty"]["level"] = "low" if in_observed else "high"
            usr["decision_signals"]["uncertainty"]["reason"] = (
                "observed_match" if in_observed else "not_in_observed"
            )
    else:
        usr["decision_signals"]["found"] = False
        usr["decision_signals"]["confidence"] = 0.0
        usr["decision_signals"]["uncertainty"] = {"level": "high", "reason": "invalid_direction"}
    usr["decision_signals"]["raw_direction"] = raw
    return usr


def eg_usr_to_text(usr: Dict[str, Any]) -> str:
    """Lossless reconstruction of "<rel> <ReceptacleId>"."""
    role = usr.get("task_semantics", {}).get("role")
    loc = usr.get("environment_facts", {}).get("location", {}).get("target")
    if role and loc:
        return f"{role} {loc}"
    return usr.get("decision_signals", {}).get("raw_direction", "")


# ---- decision keys ----

def sd_decision_key(usr: Dict[str, Any]) -> str:
    rels = usr.get("environment_facts", {}).get("relations", [])
    parts = []
    for r in rels:
        parts.append(f"{re.sub(r'[^a-z0-9]', '', r['object'].lower())}:"
                     f"{r['type']}:{re.sub(r'[^a-z0-9]', '', r['target'].lower())}")
    return "|".join(sorted(parts))


def eg_decision_key(usr: Dict[str, Any]) -> str:
    role = usr.get("task_semantics", {}).get("role", "")
    loc = usr.get("environment_facts", {}).get("location", {}).get("target", "")
    return re.sub(r"[^a-z0-9@]", "", f"{role}@{loc}".lower())
