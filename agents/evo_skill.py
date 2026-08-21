"""Versioned effect-verified Skill for the USR RoboAgent runtime.

The Skill file is the trainable artifact. This module is a frozen executor:
it compiles one markdown JSON block and enforces only the contracts named
in that block. SkillOpt may edit a bounded field set; it may not edit this
file during a training run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SKILL_ENV = "ROBOAGENT_EVO_SKILL"

EDITABLE_FIELDS = {
    "repeated_effect_miss_limit",
    "verify_grounded_object",
    "invalidate_stale_suffix",
    "expose_progress_to_scheduler",
    "skip_confirmed_effects",
    "recovery_instruction",
    "scheduler_context_mode",
    "grounding_contract_mode",
    "skip_feedback_mode",
    "invalidate_perception_after_world_change",
}

WORLD_CHANGE_PREFIXES = (
    "take ",
    "pick ",
    "pick up ",
    "put ",
    "put down ",
    "open ",
    "close ",
    "slice ",
    "heat ",
    "cool ",
    "clean ",
    "turn on ",
    "turn off ",
    "use ",
)


def _compact(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", str(value or ""))
    value = re.sub(r"\b(hint|except)\b.*$", " ", value, flags=re.I)
    value = re.sub(r"\b(the|a|an|somewhere|target|object)\b", " ", value, flags=re.I)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\b\d+\b", " ", value)
    return re.sub(r"[^a-z]+", "", value.lower())


DEFAULT_ALIASES = {
    "slicedapple": "apple",
    "applesliced": "apple",
    "slicedlettuce": "lettuce",
    "lettucesliced": "lettuce",
    "slicedbread": "bread",
    "breadsliced": "bread",
    "slicedtomato": "tomato",
    "tomatosliced": "tomato",
    "slicedpotato": "potato",
    "potatosliced": "potato",
    "key": "keychain",
    "garbagecan": "garbagecan",
    "desk lamp": "desklamp",
    "floor lamp": "floorlamp",
    "soap bottle": "soapbottle",
    "spray bottle": "spraybottle",
    "watering can": "wateringcan",
    "hand towel": "handtowel",
    "dish sponge": "dishsponge",
    # Instruction paraphrases for the same ALFRED dining-table receptacle.
    # Do NOT collapse coffee/side tables into dining table.
    "kitchentable": "diningtable",
    "woodentable": "diningtable",
    "dinnertable": "diningtable",
    "dining table": "diningtable",
    "kitchen table": "diningtable",
    "wooden table": "diningtable",
    "on the table": "diningtable",
    "table": "diningtable",
    "fridge": "fridge",
    "refrigerator": "fridge",
    "bar of soap": "soapbar",
    "barofsoap": "soapbar",
    "soap bar": "soapbar",
    "soapbar": "soapbar",
    "tv remote": "remotecontrol",
    "tvremote": "remotecontrol",
    "remote": "remotecontrol",
    "remote control": "remotecontrol",
    "remotecontrol": "remotecontrol",
    "metal rack": "shelf",
    "metalrack": "shelf",
    "rack": "shelf",
}

# Compact-form clusters: paraphrases of one class, not distinct fixtures.
RECEPTACLE_PARAPHRASE_CLUSTERS = (
    frozenset({"diningtable", "kitchentable", "woodentable", "dinnertable", "table"}),
    frozenset({"garbagecan", "trashcan", "rubbishbin"}),
    frozenset({"sofa", "couch"}),
    frozenset({"tvstand", "televisionstand"}),
    frozenset({"fridge", "refrigerator"}),
    frozenset({"shelf", "metalrack", "rack", "shelving"}),
    frozenset({"soapbar", "barofsoap"}),
    frozenset({"remotecontrol", "tvremote", "remote"}),
)


@dataclass(frozen=True)
class EvoSkillSpec:
    schema_version: str
    version: int
    name: str
    repeated_effect_miss_limit: int
    verify_grounded_object: bool
    invalidate_stale_suffix: bool
    expose_progress_to_scheduler: bool
    skip_confirmed_effects: bool
    recovery_instruction: str
    scheduler_context_mode: str = "on_intervention"
    grounding_contract_mode: str = "referential_only"
    skip_feedback_mode: str = "virtual_success"
    invalidate_perception_after_world_change: bool = True
    aliases: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""
    sha256: str = ""

    @classmethod
    def from_markdown(cls, path: str | os.PathLike[str]) -> "EvoSkillSpec":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if len(matches) != 1:
            raise ValueError(f"Skill must contain exactly one JSON block: {source}")
        raw = json.loads(matches[0])
        required = {
            "schema_version",
            "version",
            "name",
            "repeated_effect_miss_limit",
            "verify_grounded_object",
            "invalidate_stale_suffix",
            "expose_progress_to_scheduler",
            "skip_confirmed_effects",
            "recovery_instruction",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Missing Skill fields: {missing}")
        if raw["schema_version"] != "roboagent_evo_skill_v1":
            raise ValueError(f"Unsupported schema: {raw['schema_version']}")
        limit = int(raw["repeated_effect_miss_limit"])
        if not 1 <= limit <= 5:
            raise ValueError("repeated_effect_miss_limit must be in [1, 5]")
        recovery = str(raw["recovery_instruction"]).strip()
        if not recovery or len(recovery) > 600:
            raise ValueError("recovery_instruction must contain 1..600 characters")
        scheduler_context_mode = str(raw.get("scheduler_context_mode", "on_intervention"))
        if scheduler_context_mode not in {"always", "on_intervention", "off"}:
            raise ValueError("unsupported scheduler_context_mode")
        grounding_contract_mode = str(raw.get("grounding_contract_mode", "referential_only"))
        if grounding_contract_mode not in {"literal", "referential_only"}:
            raise ValueError("unsupported grounding_contract_mode")
        skip_feedback_mode = str(raw.get("skip_feedback_mode", "virtual_success"))
        if skip_feedback_mode not in {"silent", "virtual_success"}:
            raise ValueError("unsupported skip_feedback_mode")
        aliases = dict(DEFAULT_ALIASES)
        aliases.update({str(k): str(v) for k, v in dict(raw.get("aliases", {})).items()})
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            schema_version=raw["schema_version"],
            version=int(raw["version"]),
            name=str(raw["name"]),
            repeated_effect_miss_limit=limit,
            verify_grounded_object=bool(raw["verify_grounded_object"]),
            invalidate_stale_suffix=bool(raw["invalidate_stale_suffix"]),
            expose_progress_to_scheduler=bool(raw["expose_progress_to_scheduler"]),
            skip_confirmed_effects=bool(raw["skip_confirmed_effects"]),
            recovery_instruction=recovery,
            scheduler_context_mode=scheduler_context_mode,
            grounding_contract_mode=grounding_contract_mode,
            skip_feedback_mode=skip_feedback_mode,
            invalidate_perception_after_world_change=bool(
                raw.get("invalidate_perception_after_world_change", True)
            ),
            aliases=aliases,
            source_path=str(source.resolve()),
            sha256=digest,
        )

    def canonical_object(self, value: str) -> str:
        raw = str(value or "").lower().replace("_", " ").replace("-", " ")
        raw = re.sub(r"\b\d+\b", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        aliased = self.aliases.get(raw, raw)
        compact = _compact(aliased)
        return _compact(self.aliases.get(compact, compact))

    def objects_compatible(self, expected: str, observed: str) -> bool:
        if not expected or not observed:
            return True
        if expected == observed or expected in observed or observed in expected:
            return True
        for cluster in RECEPTACLE_PARAPHRASE_CLUSTERS:
            if expected in cluster and observed in cluster:
                return True
        return False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "name": self.name,
            "repeated_effect_miss_limit": self.repeated_effect_miss_limit,
            "verify_grounded_object": self.verify_grounded_object,
            "invalidate_stale_suffix": self.invalidate_stale_suffix,
            "expose_progress_to_scheduler": self.expose_progress_to_scheduler,
            "skip_confirmed_effects": self.skip_confirmed_effects,
            "recovery_instruction": self.recovery_instruction,
            "scheduler_context_mode": self.scheduler_context_mode,
            "grounding_contract_mode": self.grounding_contract_mode,
            "skip_feedback_mode": self.skip_feedback_mode,
            "invalidate_perception_after_world_change": self.invalidate_perception_after_world_change,
        }


@dataclass
class SkillIntervention:
    kind: str
    reason: str
    invalidate_suffix: bool = False
    replacement: Any = None


class EffectVerifiedSkill:
    """Per-episode runtime state for a frozen :class:`EvoSkillSpec`."""

    def __init__(self, spec: EvoSkillSpec, trace_fn=None):
        self.spec = spec
        self._trace_fn = trace_fn
        self.task_instruction = ""
        self.confirmed_effects: List[str] = []
        self.unverified_effects: List[str] = []
        self._failure_counts: Dict[str, int] = {}
        self._replan_requested = False
        self._replan_reason = ""
        self._scheduler_context_pending = False
        self.perception_stale = False
        self.observation_version = 0

    @classmethod
    def from_environment(cls, trace_fn=None) -> Optional["EffectVerifiedSkill"]:
        path = os.environ.get(SKILL_ENV, "").strip()
        if not path:
            return None
        return cls(EvoSkillSpec.from_markdown(path), trace_fn=trace_fn)

    def reset(self) -> None:
        self.task_instruction = ""
        self.confirmed_effects.clear()
        self.unverified_effects.clear()
        self._failure_counts.clear()
        self._replan_requested = False
        self._replan_reason = ""
        self._scheduler_context_pending = False
        self.perception_stale = False
        self.observation_version = 0

    def set_task(self, instruction: str) -> None:
        self.task_instruction = str(instruction or "")
        self._trace("skill_task", instruction=self.task_instruction)

    def note_new_observation(self) -> int:
        self.observation_version += 1
        self._trace("observation_version", observation_version=self.observation_version)
        return self.observation_version

    def should_bypass_last_goto_shortcut(self) -> bool:
        return bool(
            self.spec.invalidate_perception_after_world_change and self.perception_stale
        )

    def note_fresh_grounding(self) -> None:
        self.perception_stale = False

    def precheck_action(self, action: str) -> Optional[SkillIntervention]:
        if not self.spec.skip_confirmed_effects:
            return None
        effect = self._expected_effect(action)
        if not effect or effect not in self.confirmed_effects:
            return None
        kind = effect.split("(", 1)[0]
        if kind not in {"at", "open", "closed", "on", "off", "holding"}:
            return None
        reason = f"Skip redundant action; effect is already confirmed: {effect}."
        self._trace(
            "action_precondition_check",
            action=action,
            expected_effect=effect,
            already_satisfied=True,
            skipped=True,
        )
        return SkillIntervention(kind="effect_already_satisfied", reason=reason)

    def validate_grounding(self, target: str, result: Any) -> Tuple[Any, Optional[SkillIntervention]]:
        self.note_fresh_grounding()
        if not self.spec.verify_grounded_object or result is False or not result:
            return result, None
        try:
            label = str(result[0]["label"])
        except (KeyError, IndexError, TypeError):
            return result, None
        expected = self.spec.canonical_object(target)
        observed = self.spec.canonical_object(label)
        if (
            self.spec.grounding_contract_mode == "referential_only"
            and not self._is_referential_target(target)
        ):
            self._trace(
                "grounding_effect_check",
                expected=expected,
                observed=observed,
                verified=None,
                abstained=True,
                reason="functional_or_abstract_target",
            )
            return result, None
        compatible = self.spec.objects_compatible(expected, observed)
        if compatible or not expected or not observed:
            self._trace(
                "grounding_effect_check",
                expected=expected,
                observed=observed,
                verified=True,
            )
            return result, None
        reason = (
            f"Grounding effect was rejected: requested {target!r}, but the Skill "
            f"returned category {label!r}. Re-observe the requested object."
        )
        self._request_replan(reason)
        self._trace(
            "grounding_effect_check",
            expected=expected,
            observed=observed,
            verified=False,
            reason=reason,
        )
        return False, SkillIntervention(
            kind="object_mismatch",
            reason=reason,
            # Do not wipe the ability buffer on a single grounding conflict:
            # false paraphrase rejects were cascading into episode failures.
            invalidate_suffix=False,
            replacement=False,
        )

    def observe_action_result(self, action: str, success: bool) -> Optional[SkillIntervention]:
        if self._is_world_changing(action):
            if self.spec.invalidate_perception_after_world_change:
                self.perception_stale = True
                self._trace(
                    "perception_stale_after_world_change",
                    action=action,
                    observation_version=self.observation_version,
                )
        signature = self._action_signature(action)
        expected_effect = self._expected_effect(action)
        if success:
            self._failure_counts.pop(signature, None)
            self._record_confirmed_effect(expected_effect)
            self._trace(
                "action_effect_check",
                action=action,
                expected_effect=expected_effect,
                verified=True,
            )
            return None

        count = self._failure_counts.get(signature, 0) + 1
        self._failure_counts[signature] = count
        if expected_effect:
            self.unverified_effects.append(expected_effect)
            self.unverified_effects[:] = self.unverified_effects[-8:]
        self._trace(
            "action_effect_check",
            action=action,
            expected_effect=expected_effect,
            verified=False,
            repeated_count=count,
        )
        if count < self.spec.repeated_effect_miss_limit:
            return None
        reason = (
            f"Expected effect was not verified after {count} attempt(s): "
            f"{expected_effect or action}. {self.spec.recovery_instruction}"
        )
        self._request_replan(reason)
        return SkillIntervention(
            kind="effect_unverified",
            reason=reason,
            invalidate_suffix=self.spec.invalidate_stale_suffix,
        )

    def consume_replan_request(self) -> Optional[str]:
        if not self._replan_requested:
            return None
        reason = self._replan_reason
        self._replan_requested = False
        self._replan_reason = ""
        return reason

    def scheduler_context(self) -> str:
        if not self.spec.expose_progress_to_scheduler or self.spec.scheduler_context_mode == "off":
            return ""
        if self.spec.scheduler_context_mode == "on_intervention":
            if not self._scheduler_context_pending:
                return ""
            self._scheduler_context_pending = False
        confirmed = "; ".join(self.confirmed_effects[-6:]) or "none yet"
        unresolved = self.unverified_effects[-1] if self.unverified_effects else "none"
        return (
            "[Effect-verified Skill state]\n"
            f"Confirmed progress: {confirmed}.\n"
            f"Most recent unverified effect: {unresolved}.\n"
            f"Observation version: {self.observation_version}.\n"
            "Preserve confirmed progress and plan only the unfinished suffix.\n"
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.spec.name,
            "schema_version": self.spec.schema_version,
            "version": self.spec.version,
            "sha256": self.spec.sha256,
            "source_path": self.spec.source_path,
        }

    def virtual_skip_feedback_enabled(self) -> bool:
        return self.spec.skip_feedback_mode == "virtual_success"

    def expected_effect(self, action: str) -> str:
        return self._expected_effect(action)

    def _request_replan(self, reason: str) -> None:
        self._replan_requested = True
        self._replan_reason = reason
        self._scheduler_context_pending = True

    @staticmethod
    def _is_world_changing(action: str) -> bool:
        low = str(action or "").strip().lower()
        return any(low.startswith(prefix) for prefix in WORLD_CHANGE_PREFIXES)

    @staticmethod
    def _is_referential_target(target: str) -> bool:
        text = re.sub(r"\([^)]*\)", " ", str(target or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return False
        abstract = (
            r"\b(some|any|suitable|appropriate|available)\b",
            r"\b(tool|place|receptacle|container|appliance|surface)\s+(for|to)\b",
            r"\bsomewhere\b",
            r"\bwhere\s+to\b",
            # Location phrases are role hints, not hard class IDs.
            r"^(on|in|at|near|under|onto|into)\b",
            r"\b(on|in|at)\s+the\s+(table|counter|shelf|floor|ground)\b",
        )
        return not any(re.search(pattern, text) for pattern in abstract)

    def _trace(self, event: str, **fields: Any) -> None:
        if self._trace_fn is not None:
            self._trace_fn({"event": event, "skill_version": self.spec.version, **fields})

    def _action_signature(self, action: str) -> str:
        words = re.sub(r"\b\d+\b", " ", str(action or "").lower())
        words = re.sub(r"\s+", " ", words).strip()
        return words

    def _expected_effect(self, action: str) -> str:
        text = str(action or "").strip()
        low = text.lower()
        if low.startswith(("take ", "pick up ")):
            return f"holding({self._action_object(text)})"
        if low.startswith(("put ", "put down ")):
            return f"placed({self._action_object(text)})"
        if low.startswith("clean "):
            return f"clean({self._action_object(text)})"
        if low.startswith("heat "):
            return f"heated({self._action_object(text)})"
        if low.startswith("cool "):
            return f"cooled({self._action_object(text)})"
        if low.startswith("slice "):
            return f"sliced({self._action_object(text)})"
        if low.startswith("turn on "):
            return f"on({self._action_object(text)})"
        if low.startswith("turn off "):
            return f"off({self._action_object(text)})"
        if low.startswith("open "):
            return f"open({self._action_object(text)})"
        if low.startswith("close "):
            return f"closed({self._action_object(text)})"
        if low.startswith(("go to ", "find a ", "find the ")):
            return f"at({self._action_object(text)})"
        return f"executed({text})" if text else ""

    def _record_confirmed_effect(self, effect: str) -> None:
        if not effect:
            return
        if effect.startswith("at("):
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if not x.startswith("at(")]
        elif effect.startswith("holding("):
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if not x.startswith("holding(")]
        elif effect.startswith("placed("):
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if not x.startswith("holding(")]
        elif effect.startswith("open("):
            obj = effect[len("open("):-1]
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if x != f"closed({obj})"]
        elif effect.startswith("closed("):
            obj = effect[len("closed("):-1]
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if x != f"open({obj})"]
        elif effect.startswith("on("):
            obj = effect[len("on("):-1]
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if x != f"off({obj})"]
        elif effect.startswith("off("):
            obj = effect[len("off("):-1]
            self.confirmed_effects[:] = [x for x in self.confirmed_effects if x != f"on({obj})"]
        if effect not in self.confirmed_effects:
            self.confirmed_effects.append(effect)
        self.confirmed_effects[:] = self.confirmed_effects[-12:]

    def _action_object(self, action: str) -> str:
        text = re.sub(
            r"^(take|pick up|put down|put|clean|heat|cool|slice|turn on|turn off|open|close|go to|find a|find the)\s+(the\s+)?",
            "",
            action,
            flags=re.I,
        )
        return re.sub(r"\s+", " ", text).strip()


def load_skill(path: str) -> EvoSkillSpec:
    return EvoSkillSpec.from_markdown(path)
