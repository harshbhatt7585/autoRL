# Experiment Journal

Use this file as the comprehensive narrative companion to `results.tsv`.
After every run, append one section with enough detail to explain outcomes and
guide the next iteration.

## Entry template

### Run: <YYYY-MM-DD HH:MM> | commit <hash> | budget <train>/<eval> | max_steps <n>
- Idea: <what you changed>
- Rationale: <why this should improve score>
- What worked:
  - <specific positive signal>
  - <metric or behavior evidence>
- What did not work:
  - <specific failure/regression/noise>
  - <metric or behavior evidence>
- Decision: <keep/discard/crash> - <brief reason>
- Next experiment:
  - <exact change to try next>
  - <why this is the best next step>
