# Validation comparison — profile `tiny`

Train window used for fitting; validation window for selection. PR-AUC CI = cluster bootstrap over customers (n=500).

| model | pr_auc | pr_auc_lo | pr_auc_hi | roc_auc | best_f1 | recall_at_precision_0.9 | recall_at_top_0.01 |
|---|---|---|---|---|---|---|---|
| rules | 0.3661 | 0.3100 | 0.5833 | 0.8857 | 0.4444 | 0.2000 | 0.6000 |
| logreg | 0.9667 | 0.9429 | 1.0000 | 0.9996 | 0.9091 | 0.8000 | 0.8000 |
| xgb | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
