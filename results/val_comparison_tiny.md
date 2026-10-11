# Validation comparison — profile `tiny`

Train window used for fitting; validation window for selection. PR-AUC CI = cluster bootstrap over customers (n=500).

| model | pr_auc | pr_auc_lo | pr_auc_hi | roc_auc | best_f1 | recall_at_precision_0.9 | recall_at_top_0.01 |
|---|---|---|---|---|---|---|---|
| rules | 0.1250 | 0.0733 | 0.4084 | 0.9764 | 0.2222 | nan | 0.0000 |
| logreg | 0.5091 | 0.4523 | 0.6845 | 0.9254 | 0.5714 | 0.3333 | 0.6667 |
| xgb | 0.7667 | 0.7337 | 0.9167 | 0.9952 | 0.8000 | 0.6667 | 0.6667 |
