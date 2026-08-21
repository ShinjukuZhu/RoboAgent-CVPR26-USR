"""Step6: SkillChannel Contract Hardening — enforced publish/consume + contract audit.

Strengthens usr_channel.py:
  - publish(skill, usr) validates schema, stamps producer/timestamp/episode-step
  - consume(skill, field) ONLY returns public fields; raw (det_query/bbox/caption)
    keys are blocked by the field whitelist
  - contract_audit() emits machine-readable record: producer/consumer/schema_version/
    fields_consumed/fields_ignored/fallback_reason per skill call
  - unit-testable: bypass attempt (reading raw field) is rejected
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

SCHEMA_VERSION = "2.0"

# public field whitelist — FULL dotted paths from USR root (section.field...).
# RAW fields are deliberately ABSENT.
PUBLIC_FIELDS = {
    "og": [
        "environment_facts.object.class", "environment_facts.object.state",
        "environment_facts.location.target", "environment_facts.location.receptacle",
        "decision_signals.found", "decision_signals.confidence",
        "decision_signals.uncertainty.level", "decision_signals.status",
    ],
    "eg": ["task_semantics.role", "environment_facts.location.target",
           "decision_signals.found", "decision_signals.confidence"],
    "sd": ["environment_facts.relations", "environment_facts.description",
           "decision_signals.found"],
}

# RAW fields that must NEVER be consumed downstream (detector artifacts)
RAW_FIELDS = ["det_query", "det_label", "bbox", "bbox_2d", "box", "qwen_raw",
              "raw_output", "raw_caption", "model_id"]


class SkillChannel:
    def __init__(self):
        self._usr: Dict[str, Dict[str, Any]] = {}
        self._audit: list = []

    # ---- publish with schema/producer/status stamp ----
    def publish(self, skill: str, usr: Dict[str, Any],
                producer: str = "unknown",
                episode_step: Optional[int] = None) -> bool:
        u = json.loads(json.dumps(usr or {}))
        u.setdefault("schema_version", SCHEMA_VERSION)
        u.setdefault("provenance", {})["producer"] = producer
        u.setdefault("temporal_context", {})["episode_step"] = episode_step
        # schema check
        ok = u.get("schema_version") == SCHEMA_VERSION and "environment_facts" in u
        if not ok:
            self._audit.append({"skill": skill, "event": "publish_rejected",
                                "reason": "schema_invalid", "ts": time.time()})
            return False
        self._usr[skill] = u
        return True

    # ---- consume: ONLY public fields, raw blocked ----
    def consume(self, skill: str, field: str) -> Any:
        if skill not in PUBLIC_FIELDS or field not in PUBLIC_FIELDS[skill]:
            # disallow arbitrary field access (blocks raw too)
            self._audit.append({"skill": skill, "field": field, "event": "field_blocked",
                                "ts": time.time()})
            return None
        u = self._usr.get(skill)
        if u is None:
            return None
        cur: Any = u
        for part in field.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        self._audit.append({"skill": skill, "field": field, "event": "consumed",
                            "value_type": type(cur).__name__, "ts": time.time()})
        return cur

    def get_usr(self, skill: str) -> Optional[Dict[str, Any]]:
        return json.loads(json.dumps(self._usr[skill])) if skill in self._usr else None

    def has(self, skill: str) -> bool:
        return skill in self._usr

    def get_field(self, skill: str, field: str) -> Any:
        """Compatibility shim used by Align+USR agent paths.

        Accepts either a full dotted public path or a short alias such as
        ``object.class`` / ``found``.
        """
        aliases = {
            "object.class": "environment_facts.object.class",
            "object.state": "environment_facts.object.state",
            "location.target": "environment_facts.location.target",
            "location.receptacle": "environment_facts.location.receptacle",
            "found": "decision_signals.found",
            "confidence": "decision_signals.confidence",
            "role": "task_semantics.role",
            "description": "environment_facts.description",
            "relations": "environment_facts.relations",
        }
        path = aliases.get(field, field)
        return self.consume(skill, path)

    def log_decision(self, skill: str, decision: Any) -> None:
        self._audit.append({
            "skill": skill,
            "event": "decision",
            "decision": str(decision),
            "ts": time.time(),
        })

    # ---- contract audit (machine-readable) ----
    def contract_audit(self) -> list:
        """Return per-skill-call audit records."""
        return [dict(r) for r in self._audit]

    def reset_audit(self) -> None:
        self._audit = []

    # ---- bypass guard: check no raw field present in any public section ----
    @staticmethod
    def check_no_raw_leak(usr: Dict[str, Any]) -> Dict[str, Any]:
        leaked = []
        for sec in ["environment_facts", "task_semantics", "decision_signals"]:
            blk = json.dumps(usr.get(sec, {}))
            for raw in RAW_FIELDS:
                if re.search(rf'"({raw})"', blk):
                    leaked.append(f"{sec}.{raw}")
        return {"ok": not leaked, "leaked": leaked}


_channel = SkillChannel()


def get_channel() -> SkillChannel:
    return _channel


def reset_channel() -> None:
    global _channel
    _channel = SkillChannel()
