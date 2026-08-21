# Fallback minimum standard: survey, reproduced workload, and SkillOpt on USR

Evidence type: literature + this lab's already-run reproductions + current-branch
code. Official AW OOD / EB base SR numbers in the result table are frozen
baselines until the sealed runs on this branch finish.

This branch is independent of the in-flight V2 refactor. It starts from
`research/next-stage-final` (USR + Align) and does not modify that working tree.

## 1. Claim

RoboAgent already chains named capabilities, but those capabilities are not a
Skill: they have no executable progress contract, no stale-perception rule, and
no held-out evolution gate. Several 2026 agents each solved one of those
pieces. Most agents, including RoboAgent, still lack the combination. The
minimum-standard intervention is therefore a **versioned Skill document** that
the USR runtime interprets, and that SkillOpt is allowed to edit.

This is not "add more modules". The executor stays frozen. The Skill is the
trainable object.

## 2. Reproduced workload (already complete, not re-claimed here)

| System | What was actually run | Scale of artifacts | Not claimed |
|---|---|---|---|
| **RoboAgent** | Full AW OOD and EB-ALFRED base, plus Native / Naive / Align / USR / FullIndep | AW134 Native SR 0.81, Align 0.84; EB50 Native 0.78, Naive 0.34, Align 0.80, Align+USR 0.78, FullIndep 0.80; 1,176 native events + 100 natural episodes | Training code of the original paper is still unreleased |
| **USR / SkillChannel** | Same EB50 protocol; OG→SD chain 186/186; shuffled-USR ablation 0.28 vs flat-USR 0.58 | Align vs Align+USR McNemar p=1.0 (USR is an interface, not an SR trick) | USR does not by itself raise SR |
| **Brain l10-35 LoRA** | Full AW134 + EB50 | AW 0.94, EB 0.80 | Separate from this fallback; not mixed into the SkillOpt claim |
| **RATs** | Structured plan outputs + public code audit | 160 plans | No full LIBERO-PRO |
| **MineEvolve** | Remedy/suffix outputs + public code | 160 suffixes | No full Minecraft eval |
| **HiMe** | Plan-revision outputs + public code | 160 revisions | No official simulator eval |
| **Cortex** | Official high-level planner + judge | 16/16 JSON; 8 judged samples | Not paper LIBERO/RoboTwin SR |
| **SkillOpt** | Official repo local problem-family run | 8 rounds; JSON-repair 4/26 → 16/26 | Document score ≠ embodied SR |
| **EmbodiSkill** | Public code audit (versioned manual, defect/lapse) | code-level | No in-lab ALFWorld rerun |
| **PCE / OpenETA / AgentCanvas** | Partial: planner format / code audit / 33/33 graph tests | see prior audit | No full host benchmark |

The point of listing this is not to advertise engineering volume. It is to
show that the failure modes below are **observed on real reproductions**, not
inferred from abstracts.

## 3. RoboAgent vs other embodied agents

| Failure | RoboAgent | Most 2026 agents | Individual system that already has a solution | This Skill field |
|---|---|---|---|---|
| After a world-changing action, old perception is reused | `last_goto == target` shortcut can skip OG after take/put/open | Common: history text is treated as current | **OpenETA**: world-changing tools are serial and force a new observation; tool result ≠ world change ≠ task success | `invalidate_perception_after_world_change` |
| Failed or stale suffix of a plan keeps executing | `ability_buffer` continues after a failed effect | Common: binary `last_action_success`, then continue | **MineEvolve Adaptor**: freeze the finished prefix, rewrite only the unfinished suffix | `invalidate_stale_suffix` + confirmed progress |
| Object identity is a string in history | label is written into `core_history`; wrong class can still proceed | HiMe tags, Cortex language memory: still no stable identity contract | **EmbodiSkill / this Skill**: reject a clear requested/returned class conflict; abstain on functional queries | `verify_grounded_object`, `grounding_contract_mode=referential_only` |
| Skill updates are hand-written or LLM-trusted | T1 EG-rule evolution: 8/8 held-out rejects, test 0.632 unchanged | EmbodiSkill trusts reflection type; RATs promotes by whole-trial success | **SkillOpt**: bounded edit + strict held-out SR gate + rejected buffer | `training/skillopt_evolve.py` |
| Confirmed work is repeated | planner may re-open / re-go | Common | OpenETA / this Skill skip already-confirmed effects | `skip_confirmed_effects` |

Problems that remain out of scope for this fallback (and are not faked as
solved): kinematic put failures (RoboAgent tasks 34/47), executor capability
gaps, cross-robot skill transfer, and independent probabilistic calibration.

## 4. Why USR is the right base

Align+USR EB50 is 0.78, statistically tied with Align 0.80. USR is therefore
**not** an SR hack. It is the only representation in this lab that already
separates environment facts, decision signals, and provenance, and that
blocks raw detector fields from downstream skills.

The Skill writes progress and observation version next to that USR object. It
does not replace USR, and it does not re-train Brain.

## 5. SkillOpt protocol (this branch)

Splits follow RoboAgent's own official protocol, not V2's in-distribution
debug jobs:

| Split | Use | Path |
|---|---|---|
| AW `eval_in_distribution` 0–19 | D_tr, trajectory evidence | development only |
| AW `eval_in_distribution` 20–39 | D_sel, SkillOpt ACCEPT/REJECT | held-out gate |
| AW `eval_out_of_distribution` 0–133 | D_test | sealed; SkillOpt must not read it |
| EB `base` 0–49 | D_test | sealed; SkillOpt must not read it |

Gate: candidate is ACCEPT only if selection SR strictly increases. Ties
reject. That is SkillOpt, not an efficiency engineering rule.

Seed Skill v0 already turns on the OpenETA / MineEvolve / identity contracts
above. SkillOpt may later change only `EDITABLE_FIELDS`. Executor code is
frozen during a run.

## 6. Frozen baselines (do not overwrite)

| Method | AW OOD n=134 | EB base n=50 |
|---|---|---|
| Native RoboAgent | 0.81 | 0.78 |
| Align | 0.84 | 0.80 |
| Align+USR | (not separately reported; AW Align is the AW reference) | 0.78 |
| Brain l10-35 (separate line) | 0.94 | 0.80 |

This branch reports **Align+USR+Skill** against Align+USR / Align / Native
under the same AW OOD and EB base commands. Live sealed runs write to
`/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt` (renamed away from
`fallback_usr_skillopt` so V2 hard-coded quarantine paths miss them) and must
not touch frozen manifests.

## 7. How to run

See `training/launch_fallback_aw_ood.sh`, `training/launch_fallback_eb50.sh`,
`training/launch_fallback_skillopt.sh`, and `training/fallback_keep_alive.sh`.
Server code lives in a **separate** directory from the V2 job:
`/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_USR_SkillOpt`.
