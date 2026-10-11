# FraudLens — Build Guide (read this first)

You have 17 files: this guide, one project-wide context file (`CLAUDE.md`), and 15 build prompts, one per part. Feed them to your coding CLI **one at a time, in order**.

## Setup (once)

1. Create an empty folder `fraudlens/` and `cd` into it. Run `git init`.
2. Copy `CLAUDE.md` to the **repo root**. Claude Code reads it automatically. For other CLIs, rename it to match: `AGENTS.md` (Codex), `GEMINI.md` (Gemini CLI), or keep it and say "read CLAUDE.md first".
3. Create `prompts/` in the repo and copy all `part_XX_*.md` files in. They are your build log too.
4. Prerequisites: Python 3.11+, Docker (for PostgreSQL), git, 16 GB RAM recommended for the `main` dataset. Part 11 needs a free Kaggle account for the IEEE-CIS data.

## How to run a part

Start the CLI in the repo root and use a short wrapper instruction plus the file:

```bash
# Claude Code (interactive): start `claude`, then:
> Read CLAUDE.md, then execute prompts/part_02_simulator.md. Work in the sub-steps it lists, run the tests after each, and stop at the end of the part.

# or non-interactive
claude -p "Read CLAUDE.md, then execute prompts/part_02_simulator.md. Stop at the end of the part."
```

Tips:
- Use plan mode first for Parts 02, 05, 07 and 13 (the hardest ones). Read the plan before approving.
- After each part: run the acceptance commands yourself, skim `git diff`, read `docs/progress/part_XX.md`, then `git commit`.
- If the CLI goes off-spec, paste the relevant section of the prompt back and say "this requirement is not met".
- Never let it continue into the next part on its own.

## Order and checkpoints

| Part | File | Deliverable |
|---|---|---|
| 01 | part_01_setup.md | Repo skeleton, config, Docker Postgres, Makefile |
| 02 | part_02_simulator.md | Behavioral simulator with hard negatives, 5 fraud scenarios, drift, label delay |
| 03 | part_03_validation_eda.md | Data validation + EDA + **separability gate** |
| 04 | part_04_postgres.md | Schema, loader, analytical views |
| 05 | part_05_features_leakage.md | Point-in-time feature module + leakage test suite |
| 06 | part_06_splits_models.md | Temporal splits, Rules / LogReg / XGBoost, metrics |
| 07 | part_07_calibration_cost.md | Calibration, cost model, decision policy, capacity |
| 08 | part_08_ml_experiments.md | Imbalance, ablation, bias-variance, importance, drift/retrain |
| 09 | part_09_isolation_forest.md | Unseen-fraud experiment |
| 10 | part_10_shap_fairness.md | SHAP explanations + subgroup audit |
| 11 | part_11_external_validation.md | IEEE-CIS real-data validation |
| 12 | part_12_final_evaluation.md | Locked one-shot test evaluation + headline numbers |
| | | ✅ **Resume-ready checkpoint** (after Part 12, with a short README) |
| 13 | part_13_api.md | FastAPI service + latency benchmark |
| 14 | part_14_streamlit.md | Analyst app |
| 15 | part_15_monitoring_ci_docs.md | PSI monitoring, CI, README, model card, interview notes |

If time runs out, ship after Part 12 plus a minimal README. Parts 13 and 14 make it demoable, but 06–12 are what show ML understanding.

## Golden rules

1. **Numbers come from code.** No result in the README or on your resume that did not come out of `results/`.
2. **The test set is touched once**, in Part 12.
3. **Log every bug** in `docs/what_went_wrong.md`. That file is interview material.
4. **Understand every part before moving on.** After each part, ask the CLI: "Explain the design decisions in this part as if I were being interviewed on it", and read the answer.
5. Cost values (₹) are assumptions. Always report them next to any cost result.
