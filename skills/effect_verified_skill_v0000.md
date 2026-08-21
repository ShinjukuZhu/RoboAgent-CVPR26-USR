# Effect-Verified Skill v0

Frozen executor artifact for RoboAgent-USR. The markdown JSON is the only
trainable object. Runtime code does not add task-specific rules.

This seed encodes three contracts that RoboAgent and most 2026 embodied agents
leave implicit, but that individual systems already demonstrated:

- OpenETA: a world-changing action makes prior perception stale.
- MineEvolve Adaptor: keep the confirmed prefix; drop only the unfinished suffix.
- SkillOpt / EmbodiSkill: versioned skill text, not hardcoded if-else repairs.

```json
{
  "schema_version": "roboagent_evo_skill_v1",
  "version": 0,
  "name": "effect_verified_replanning",
  "repeated_effect_miss_limit": 2,
  "verify_grounded_object": true,
  "invalidate_stale_suffix": true,
  "expose_progress_to_scheduler": true,
  "skip_confirmed_effects": true,
  "scheduler_context_mode": "on_intervention",
  "grounding_contract_mode": "referential_only",
  "skip_feedback_mode": "virtual_success",
  "invalidate_perception_after_world_change": true,
  "recovery_instruction": "Re-observe the current view, preserve confirmed progress, and replan only the unfinished suffix. Do not repeat the invalidated action chain.",
  "aliases": {
    "kitchen table": "diningtable",
    "wooden table": "diningtable",
    "dinner table": "diningtable",
    "dining table": "diningtable",
    "apple sliced": "apple",
    "bread sliced": "bread",
    "lettuce sliced": "lettuce",
    "potato sliced": "potato",
    "tomato sliced": "tomato"
  }
}
```

## Runtime meaning

- Environment success is recorded as progress only after the simulator confirms it.
- A clear requested-object / returned-object conflict is treated as not-found.
- Instruction paraphrases of the same dining-table receptacle (kitchen/wooden/dinner)
  are compatible; coffee/side tables remain distinct.
- Functional queries such as "some tool for cooling" are not forced into a
  literal object-class match.
- Repeated missing effects invalidate the remaining ability-buffer suffix.
- Replanning receives confirmed progress only after an intervention.
- After take/put/open/close/slice/heat/cool/clean, the last-goto grounding
  shortcut is disabled until the next real object-grounding call.
