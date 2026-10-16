# Validation comparison — profile `dev`

Train window used for fitting; validation window for selection. PR-AUC CI = cluster bootstrap over customers (n=500).

| model | pr_auc | pr_auc_lo | pr_auc_hi | roc_auc | best_f1 | recall_at_precision_0.9 | recall_at_top_0.01 |
|---|---|---|---|---|---|---|---|
| rules | 0.2923 | 0.1773 | 0.3906 | 0.8044 | 0.4743 | nan | 0.3861 |
| logreg | 0.8638 | 0.7965 | 0.9063 | 0.9915 | 0.8302 | 0.7583 | 0.8167 |
| xgb | 0.9291 | 0.8824 | 0.9591 | 0.9957 | 0.8802 | 0.8500 | 0.8917 |
