"""Stage B contract adapter — Fix1 label semantics only (no Fix1.5 soft).

Rules (aligned with Fix1 / rejected Fix1.5):
  - Compositional / functional goals → ``soft_type(last_goto)`` when available.
  - Atomic + type-match(last_goto, query) → keep instance id (or soft_type if strip_ids).
  - Atomic otherwise → keep query / target_obj as label (book on Desk stays book).
  - NEVER apply Fix1.5 ``atomic_prefer_last_goto_soft`` (furniture synonym remap
    on atomic paths) — that path was rejected as harmful.

Env:
  ROBOAGENT_OG_STRIP_IDS=1  → prefer soft_type over ``Sink 1`` / ``Apple 1`` forms
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_COMPOSITIONAL_RE = re.compile(
    r"(?i)\b("
    r"some tool|something to|somewhere|another |a place to|tool for|"
    r"the back of|on to the|onto the|for cleaning|for slicing|for heating|"
    r"for cooling|for washing"
    r")\b"
)


def soft_type(name: Optional[str]) -> Optional[str]:
    """'Sink 1' / 'DiningTable 2' -> 'sink' / 'dining table'."""
    if not name:
        return None
    t = str(name).strip()
    t = re.sub(r"\s+\d+$", "", t).strip()
    if not t:
        return None
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", t)
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t or None


def normalize_query(target_obj: str) -> str:
    """Turn 'Apple 1' / 'sliced Apple 2' into a detector-ish text query."""
    t = (target_obj or "").strip()
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()
    t = re.sub(r"\s+\d+$", "", t).strip()
    t = re.sub(r"(?i)^sliced\s+", "", t).strip()
    return t if t else (target_obj or "").strip()


def is_compositional(target_obj: str) -> bool:
    q = (target_obj or "").strip()
    if not q:
        return False
    if _COMPOSITIONAL_RE.search(q):
        return True
    if len(q.split()) >= 5:
        return True
    return False


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def type_match(last_goto: Optional[str], target_obj: str) -> bool:
    """True if last_goto refers to the same type as the atomic query."""
    a = soft_type(last_goto)
    b = soft_type(target_obj) or normalize_query(target_obj).lower()
    if not a or not b:
        return False
    ca, cb = _compact(a), _compact(b)
    return ca == cb or ca in cb or cb in ca


# Compositional / functional goals → preferred ALFRED-ish tool/recep.
_FUNCTIONAL_MAP = (
    (re.compile(r"(?i)\b(clean|wash|rins)", re.I), "sink"),
    (re.compile(r"(?i)\b(slic|cut)", re.I), "knife"),
    (re.compile(r"(?i)\b(heat|microwave)", re.I), "microwave"),
    (re.compile(r"(?i)\b(cool|chill|refriger)", re.I), "fridge"),
    (re.compile(r"(?i)\b(garbage|trash|throw)", re.I), "garbage can"),
)


def functional_canonical(target_obj: str) -> Optional[str]:
    """Map 'some tool for cleaning X' → sink, etc. None if not functional."""
    q = (target_obj or "").strip()
    if not q:
        return None
    for pat, lab in _FUNCTIONAL_MAP:
        if pat.search(q):
            return lab
    return None


def strip_ids_enabled() -> bool:
    return os.environ.get("ROBOAGENT_OG_STRIP_IDS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class AdapterResult:
    """Stage B output before gated Stage C decision."""

    label: Optional[str]
    compositional: bool
    ambiguity: bool
    ambiguity_reasons: List[str] = field(default_factory=list)
    path: str = "atomic_naive_label"
    query: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


def adapt_label(
    target_obj: str,
    last_goto: Optional[str] = None,
    detector_found: bool = False,
    det_score: Optional[float] = None,
    det_label: Optional[str] = None,
    strip_ids: Optional[bool] = None,
) -> AdapterResult:
    """Normalize / remap label with Fix1 compositional rules only.

    Does **not** apply Fix1.5 atomic furniture soft remap.
    """
    if strip_ids is None:
        strip_ids = strip_ids_enabled()

    query = normalize_query(target_obj)
    compositional = is_compositional(target_obj) or is_compositional(query)
    reasons: List[str] = []
    meta: Dict[str, Any] = {
        "query": query,
        "target_obj": target_obj,
        "last_goto": last_goto,
        "compositional": compositional,
        "detector_found": detector_found,
        "det_score": det_score,
        "det_label": det_label,
        "strip_ids": strip_ids,
        "fix15_soft_disabled": True,
    }

    if not detector_found:
        reasons.append("no_detector")

    if compositional:
        # Functional ontology first (cleaning→sink, slicing→knife, …).
        # Prefer this over a wrong last_goto receptacle (cabinet/stove on
        # "tool for cleaning") — EB ep26 failure mode under remap v1.
        func = functional_canonical(target_obj) or functional_canonical(query)
        soft = soft_type(last_goto)
        if func:
            label = func
            path = "compositional_functional"
            reasons.append("compositional")
            reasons.append("functional_ontology")
            meta["canonical_from"] = "functional"
            # Keep last_goto soft only when it agrees with the function.
            if soft and _compact(soft) == _compact(func):
                meta["last_goto_agrees_functional"] = True
            elif soft:
                meta["last_goto_soft_ignored"] = soft
        elif soft:
            label = soft
            path = "compositional_last_goto"
            reasons.append("compositional")
            meta["canonical_from"] = "last_goto_soft"
        else:
            label = soft_type(query) or query or target_obj
            path = "compositional_no_goto"
            reasons.append("compositional")
            reasons.append("missing_last_goto")
            meta["canonical_from"] = "query_fallback"
        if strip_ids and label:
            label = soft_type(label) or label
        return AdapterResult(
            label=label,
            compositional=True,
            ambiguity=True,
            ambiguity_reasons=list(dict.fromkeys(reasons)),
            path=path,
            query=query,
            meta=meta,
        )

    # ----- Atomic (Fix1 only) -----
    if last_goto and type_match(last_goto, target_obj):
        label = soft_type(last_goto) if strip_ids else last_goto
        path = "atomic_type_match_instance"
        meta["canonical_from"] = "last_goto_instance"
    else:
        # Fix1 atomic_naive_label — keep object query, never furniture soft.
        # Explicitly refuse Fix1.5 prefer_last_goto_soft.
        label = soft_type(target_obj) if strip_ids else target_obj
        path = "atomic_naive_label"
        meta["canonical_from"] = "query"
        meta["fix15_soft_skipped"] = True

    ambiguity = bool(reasons)
    return AdapterResult(
        label=label,
        compositional=False,
        ambiguity=ambiguity,
        ambiguity_reasons=list(dict.fromkeys(reasons)),
        path=path,
        query=query,
        meta=meta,
    )
