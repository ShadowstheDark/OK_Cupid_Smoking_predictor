# Model results

Every figure below is produced by `scripts/report_metrics.py`, which trains the
pipeline in `smoking_model.py` and prints what it actually measured. The headline
point of the table is the comparison between the baseline and the models: on
imbalanced data, accuracy alone is not evidence of anything.

The repository ships a 10,000-row sample and git-ignores the full extract, so the
two datasets below give slightly different numbers. Both are correct; quoting one
without naming it is not, which is why every table is labelled.

## Committed sample — 10,000 rows

*reads `data/sample_profiles.csv`*

| Model | Accuracy | Precision (smoker) | Recall (smoker) | F1 (smoker) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline — always answer "does not smoke" | 0.806 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.817 | 0.591 | 0.185 | 0.281 | 0.727 |
| Logistic regression, `class_weight="balanced"` | 0.712 | 0.362 | 0.639 | 0.462 | 0.727 |
| Random forest, `max_depth=5` | 0.733 | 0.371 | 0.545 | 0.442 | 0.714 |
| HistGradientBoosting (balanced) | 0.728 | 0.377 | 0.625 | 0.471 | 0.729 |

- Profiles in file: **10,000**
- Profiles with a usable smoking label: **9,081**
- Held-out test split (20%, stratified): **1,817**
- Smoker base rate: **19.4%**
- Smoker rate when the drugs question was answered `never`: **12.4%** (7,214 profiles)
- Smoker rate when the drugs question was left blank: **23.1%** (1,867 profiles)

### What drives the balanced model

Coefficients from the balanced logistic regression, largest positive and
largest negative. Odds ratios above 1 increase the odds of the profile being
labelled a smoker, below 1 decrease them.

| Feature | Coefficient (log-odds) | Odds ratio |
|---|---|---|
| `drugs_often` | +2.426 | 11.31x |
| `drugs_sometimes` | +1.546 | 4.69x |
| `drinks_very often` | +0.853 | 2.35x |
| `drinks_often` | +0.782 | 2.19x |
| `drugs_unspecified` | +0.662 | 1.94x |
| `body_type_unspecified` | -0.381 | 0.68x |
| `body_type_fit` | -0.411 | 0.66x |
| `body_type_athletic` | -0.620 | 0.54x |
| `body_type_overweight` | -0.709 | 0.49x |
| `body_type_jacked` | -0.784 | 0.46x |

## Full 2012 extract — 59,946 rows

*reads `data/full/profiles.csv`*

| Model | Accuracy | Precision (smoker) | Recall (smoker) | F1 (smoker) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline — always answer "does not smoke" | 0.806 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.820 | 0.618 | 0.190 | 0.290 | 0.743 |
| Logistic regression, `class_weight="balanced"` | 0.716 | 0.364 | 0.625 | 0.460 | 0.742 |
| Random forest, `max_depth=5` | 0.754 | 0.402 | 0.553 | 0.466 | 0.738 |
| HistGradientBoosting (balanced) | 0.725 | 0.374 | 0.626 | 0.468 | 0.746 |

- Profiles in file: **59,946**
- Profiles with a usable smoking label: **54,434**
- Held-out test split (20%, stratified): **10,887**
- Smoker base rate: **19.4%**
- Smoker rate when the drugs question was answered `never`: **12.1%** (43,117 profiles)
- Smoker rate when the drugs question was left blank: **22.9%** (11,317 profiles)

### What drives the balanced model

Coefficients from the balanced logistic regression, largest positive and
largest negative. Odds ratios above 1 increase the odds of the profile being
labelled a smoker, below 1 decrease them.

| Feature | Coefficient (log-odds) | Odds ratio |
|---|---|---|
| `drugs_often` | +2.250 | 9.49x |
| `drugs_sometimes` | +1.596 | 4.93x |
| `drinks_very often` | +1.245 | 3.47x |
| `drinks_often` | +0.783 | 2.19x |
| `drinks_desperately` | +0.734 | 2.08x |
| `drinks_rarely` | -0.236 | 0.79x |
| `body_type_thin` | -0.302 | 0.74x |
| `body_type_unspecified` | -0.391 | 0.68x |
| `body_type_athletic` | -0.550 | 0.58x |
| `body_type_fit` | -0.558 | 0.57x |

## Why the notebook's numbers differ from the shipped pipeline's

The notebook encodes its features with `pd.get_dummies(..., drop_first=True)` on
the raw columns, so an unanswered question becomes an all-zero row that the model
cannot tell apart from the dropped reference level (`never` for drugs). The
shipped pipeline gives blank answers their own `unspecified` category instead.

Same rows, same four features, same split and seed — only the encoding differs:

| Encoding | Columns | Model | Accuracy | Recall | F1 |
|---|---|---|---|---|---|
| Notebook — blanks folded into the reference | 19 | unweighted | 0.820 | 0.184 | 0.283 |
| Notebook — blanks folded into the reference | 19 | balanced | 0.709 | 0.592 | 0.441 |
| Shipped — explicit `unspecified` category | 22 | unweighted | 0.820 | 0.190 | 0.290 |
| Shipped — explicit `unspecified` category | 22 | balanced | 0.716 | 0.625 | 0.460 |

Under the notebook's encoding, 46,729 of 54,434 rows (86%) encode identically for “reports never
using drugs” and “did not answer the question”, so the distinction measured
directly above is invisible to the model. Giving blanks their own category
recovers it: balanced-model recall moves 0.592 to 0.625 on the same split, with accuracy moving 0.709 to 0.716. That is one split rather than cross-validation, so read the direction as the finding and the size of the gap as an estimate.

## Reading the table

About 80.6% of profiles with a usable label do not smoke, so the
baseline row is not a weak model — it is the number to beat. The unweighted logistic
regression edges past it on accuracy while catching a small minority of smokers;
balancing the class weights gives back a large amount of recall and gives up accuracy
in exchange. Which of those two rows is "better" depends entirely on whether the cost
of a missed smoker or a false alarm is higher, and that is a product decision rather
than a statistical one.

Regenerate with:

```bash
python scripts/report_metrics.py --write
```
