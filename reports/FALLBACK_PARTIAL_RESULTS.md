# Fallback partial results (in-progress)

| Split | n / target | SR so far |
|---|---|---|
| AW OOD | 21 / 134 | 0.857 |
| EB base | 37 / 50 | 0.703 |
| SkillOpt D_tr | 20 / 20 | 0.650 |
| SkillOpt D_sel baseline | 2 / 20 | 1.000 |

Refs: Align AW 0.84 / EB 0.80; Align+USR EB 0.78.

Recent Skill fixes (hot on EB resume ≥37): paraphrase aliases, no buffer wipe on
object mismatch, abstain on location phrases and effect predicates like `closed(...)`.
SkillOpt evolve is running baseline D_sel on GPU6; paraphrase re-eval armed after EB50.
