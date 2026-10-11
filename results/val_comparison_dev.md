# Validation comparison — profile `dev`

Train window used for fitting; validation window for selection. PR-AUC CI = cluster bootstrap over customers (n=500).

| model | pr_auc | pr_auc_lo | pr_auc_hi | roc_auc | best_f1 | recall_at_precision_0.9 | recall_at_top_0.01 |
|---|---|---|---|---|---|---|---|
| rules | 0.3787 | 0.2418 | 0.4943 | 0.8091 | 0.5612 | 0.0322 | 0.4678 |
| logreg | 0.9021 | 0.8531 | 0.9379 | 0.9939 | 0.8469 | 0.7924 | 0.8889 |
| xgb | 0.9667 | 0.9466 | 0.9810 | 0.9987 | 0.9212 | 0.9152 | 0.9561 |
