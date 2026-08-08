"""Minimal Skill I/O contract for RoboAgent modularization (Phase 1).

Syntactic OG surface stays ``False | [{"label": ...}]`` at the agent boundary;
this dataclass is the internal semantic carrier (found / label / meta / failure).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class SkillResult:
    """Canonical skill output used by adapters and gated cascades."""

    found: bool
    label: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None

    def to_og_return(self) -> Union[bool, List[Dict[str, Any]]]:
        """Map to RoboAgent OG wire format (False | [{"label": ...}])."""
        if not self.found or not self.label:
            return False
        item: Dict[str, Any] = {"label": self.label}
        for k in ("box", "score", "detector_found", "det_label", "query"):
            if k in self.meta and self.meta[k] is not None:
                item[k] = self.meta[k]
        return [item]


def og_return_to_skill(
    ret: Any,
    meta: Optional[Dict[str, Any]] = None,
    failure_type: Optional[str] = None,
) -> SkillResult:
    """Parse OG wire format into SkillResult."""
    meta = dict(meta or {})
    if ret is False or ret is None:
        return SkillResult(
            found=False,
            label=None,
            meta=meta,
            failure_type=failure_type or meta.get("reject") or "not_found",
        )
    if isinstance(ret, list) and ret and isinstance(ret[0], dict) and "label" in ret[0]:
        lab = ret[0].get("label")
        for k in ("box", "score", "detector_found", "det_label", "query"):
            if k in ret[0] and k not in meta:
                meta[k] = ret[0][k]
        return SkillResult(found=True, label=str(lab) if lab is not None else None, meta=meta)
    return SkillResult(
        found=False,
        label=None,
        meta=meta,
        failure_type=failure_type or "invalid_og_shape",
    )
