# Fallback results (EB sealed; AW + SkillOpt history in progress)

**Run root:** `/mnt/autodl_tmp1/zhuyanhao/runs/usr_minstd_skillopt`  
**Branch:** `research/fallback-usr-skillopt`  
**Survey:** `reports/FALLBACK_MIN_STANDARD.md` (reproduced workload + Skill vs other agents)

## Sealed so far

| Split | n | SR | vs baseline |
|---|---:|---:|---|
| EB-ALFRED base | **50 / 50** | **0.84** | Align+USR 0.78 ✓ / Align 0.80 ✓ |
| SkillOpt D_tr | 20 / 20 | 0.65 | development evidence |
| SkillOpt D_sel (v0) | 20 / 20 | 0.75 | held-out gate baseline |

EB lift came from USR + effect-verified Skill + paraphrase grounding fixes, then
paraphrase reeval on false-reject fails (`eb_paraphrase_reeval_summary.json`).

## In progress

| Track | Status |
|---|---|
| AW OOD | ~53 / 134 @ ~0.75; main + shards 70–90 / 90–134 |
| SkillOpt evolution | round-1 candidate (`miss_limit` 2→3) ~14/20; sealer will write ACCEPT/REJECT history |

## Skill contracts landed on USR

- `invalidate_perception_after_world_change` (OpenETA-style)
- `invalidate_stale_suffix` + confirmed progress (MineEvolve-style)
- `verify_grounded_object` + receptacle paraphrases (EmbodiSkill-style)
- SkillOpt strict held-out SR gate (`training/skillopt_evolve.py`)

Final sealed AW134 + `history.jsonl` will replace this note via
`training/finalize_fallback_results.sh` → `FALLBACK_FINAL_RESULTS.md`.
