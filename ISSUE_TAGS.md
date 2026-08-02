# Issue Tags

Reference for tagging issues consistently across repos. Prefix issue titles with the tag, e.g. `[ENH] Add SALI threshold sweep`.

## Type

| Tag          | Use for                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| `[ENH]`      | New feature or improvement to existing behavior                                                         |
| `[BUG]`      | Something is broken and needs fixing                                                                    |
| `[DOC]`      | Documentation-only changes, no code touched                                                             |
| `[REFACTOR]` | Restructuring code/config with no behavior change                                                       |
| `[CHORE]`    | Housekeeping — dependency bumps, gitignore, release prep                                                |
| `[PROC]`     | Process or convention change, not a code deliverable                                                    |
| `[EXP]`      | Exploratory/experimental work where the outcome is the deliverable                                      |
| `[VERIF]`    | Verification or reproduction check — confirming something works as expected, nothing necessarily broken |
| `[DEPR]`     | Deprecation notice                                                                                      |

## Priority / Status (stack with a type tag)

| Tag         | Use for                                               |
| ----------- | ----------------------------------------------------- |
| `[URG]`     | Urgent, needs attention soon                          |
| `[BLOCKED]` | Waiting on something external before work can proceed |
| `[DISCUSS]` | Needs a decision before work is scoped/assigned       |

## Examples

- `[ENH] Add SALI threshold sweep across test set`
- `[PROC] Adopt run-log convention for experiment tracking`
- `[VERIF] Confirm SALI matches known chaotic/regular labels`
- `[URG][BUG] uv sync fails on agama build`
