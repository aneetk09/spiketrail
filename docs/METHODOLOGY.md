# SpikeTrail Methodology

SpikeTrail is a retrospective fraud-spike detector for one narrow pattern:
small transaction bursts followed by a larger transaction from the same
device/IP context. The project is intentionally evaluated on precision,
recall, F1, and false-positive cost instead of accuracy alone, because fraud is
rare and accuracy can look strong even when the detector catches nothing.

This document starts as the Stage 1 evaluation vocabulary draft. Later stages
will add the synthetic-data rules, feature methodology, threshold choice,
held-out test metrics, false-positive cost assumption, failure notes, and
production-readiness limits.

## Public Demo Config Vs. Private Reported-Results Config

The repository is designed to run from a fresh clone without requiring private
fraud-pattern thresholds. By default, the data generator reads
`config/pattern.demo.json`, a committed file containing illustrative demo
parameters. Those values prove that the pipeline can run end to end, but they
are not the values used for SpikeTrail's reported precision, recall, F1, or
confusion matrix.

The reported results are generated from a separate run using
`config/pattern.local.json`, which is intentionally gitignored. That file holds
the exact burst count, time windows, and amount cutoffs selected for the
project. Keeping those values out of the public repo is part of the
defense-only design: the code is inspectable and runnable, while the sensitive
configuration that could help someone tune around the detector is not
published.

To run with the private config locally:

```text
SPIKETRAIL_PATTERN_CONFIG=config/pattern.local.json python -m src.generate_data
```

To run the standalone public demo path, no override is required.

## Evaluation Vocabulary

Every transaction in the held-out test set will have two values:

- The ground-truth label from the synthetic data generator: fraud or clean.
- The model decision after thresholding: flagged or not flagged.

Those two values produce four possible outcomes:

| Term | Meaning in SpikeTrail |
| --- | --- |
| True positive (TP) | A fraudulent transaction was correctly flagged. |
| False positive (FP) | A legitimate transaction was wrongly flagged. |
| True negative (TN) | A legitimate transaction was correctly left alone. |
| False negative (FN) | A fraudulent transaction was missed. |

The confusion matrix shows these counts together:

|  | Predicted fraud | Predicted clean |
| --- | ---: | ---: |
| Actually fraud | TP | FN |
| Actually clean | FP | TN |

## Why Accuracy Is Not Enough

The planned dataset is about 2,000 transactions with about 4% fraud. That means
roughly 80 fraudulent transactions and 1,920 legitimate transactions.

A useless detector that always predicts "clean" would get:

| Outcome | Count |
| --- | ---: |
| True positives | 0 |
| False positives | 0 |
| True negatives | 1,920 |
| False negatives | 80 |

Accuracy would be:

```text
(TP + TN) / all transactions
(0 + 1,920) / 2,000 = 96%
```

That 96% accuracy sounds good, but the detector caught zero fraud. For this
project, that would be a failure. A fraud detector has to show how many frauds
it caught and how many legitimate customers it disrupted, not just how many
total rows it classified correctly.

## Precision

Precision answers: when SpikeTrail flags a transaction, how often is it right?

```text
precision = TP / (TP + FP)
```

If the model flags 25 transactions, and 15 are actually fraud while 10 are
legitimate, precision is:

```text
15 / (15 + 10) = 0.60
```

That means 60% of flagged transactions were truly fraudulent, while 40% were
false alarms. Precision matters because every false positive can block or
interrupt a legitimate customer.

## Recall

Recall answers: out of all real fraud, how much did SpikeTrail catch?

```text
recall = TP / (TP + FN)
```

If there are 80 fraudulent transactions and the model catches 60 of them, recall
is:

```text
60 / (60 + 20) = 0.75
```

That means the detector caught 75% of fraud and missed 25%. Recall matters
because missed fraud still creates chargeback, refund, and merchant-loss risk.

## F1 Score

F1 combines precision and recall into one score using their harmonic mean:

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

F1 is useful when both false positives and false negatives matter. It does not
replace the confusion matrix or false-positive cost, but it gives one compact
summary of the tradeoff.

For example, if precision is 0.60 and recall is 0.75:

```text
F1 = 2 * (0.60 * 0.75) / (0.60 + 0.75)
F1 = 0.67
```

## Hypothetical Worked Example At Project Scale

*(Note: The following is a hypothetical teaching example created during Stage 1 to explain the evaluation vocabulary, not the actual Stage 8 results.)*

Assume the full 2,000-transaction dataset has about 80 fraudulent transactions.
The locked plan calls for an 80/20 stratified split, so the held-out test set
should contain about 400 transactions:

| Test-set group | Approximate count |
| --- | ---: |
| Fraud | 16 |
| Clean | 384 |
| Total | 400 |

Suppose the final held-out evaluation produces this confusion matrix:

|  | Predicted fraud | Predicted clean |
| --- | ---: | ---: |
| Actually fraud | 12 | 4 |
| Actually clean | 8 | 376 |

The metrics would be:

```text
precision = 12 / (12 + 8) = 0.60
recall    = 12 / (12 + 4) = 0.75
F1        = 2 * (0.60 * 0.75) / (0.60 + 0.75) = 0.67
accuracy  = (12 + 376) / 400 = 0.97
```

The 97% accuracy is less informative than the other numbers. The important
story is that 12 of about 16 fraud transactions were caught, 4 were missed, and
8 legitimate transactions were incorrectly flagged. Those false positives need
a rupee-cost assumption before choosing the final decision threshold.

## Current Limitations Of This Draft

This Stage 1 draft does not yet contain real model results. The numbers above
are hand examples to make the evaluation vocabulary concrete before data
generation, feature engineering, or model training starts.

The final methodology will report the actual held-out metrics even if they are
messy. The small held-out fraud count, expected to be around 16 cases, will be
called out plainly because each missed or correctly caught fraud case can move
the final percentages noticeably.

## Failure Recovery: The Split Bug

During Stage 5, the initial data split used a standard row-level `train_test_split` stratified by label. This inadvertently shattered multi-transaction sequences across the train and test sets. As a result, large cashout transactions in the test set lost their preceding small-burst context (which ended up in the train set), causing point-in-time features like velocity and burst ratio to fail and resulting in near-random test performance. The fix was to replace the row-level split with a group-aware split (`GroupShuffleSplit` keyed on `sequence_id`), ensuring entire sequences land strictly in either train or test to preserve chronological context for feature extraction.

## False-Positive Cost and Threshold Selection (Stage 9)

In standard setups, a classification threshold defaults to 0.5. However, since the cost of a missed fraud case and a false alarm are highly unequal in a payment context, the threshold should be chosen to minimize the total expected cost.

**Cost Assumptions:**
- **False Positive (FP) Cost:** We estimate a false positive costs roughly **₹300**. This accounts for the lost margin on a typical legitimate transaction, plus the operational cost of fielding a customer support ticket for the wrongly blocked payment, plus the marginal risk of customer churn.
- **False Negative (FN) Cost:** We estimate a missed fraud case costs **₹5,500**. This assumes the fraudulent "cashout" averages ₹5,000, plus a standard ₹500 chargeback fee imposed on the merchant.

Given these figures, a missed fraud case is approximately 18 times more expensive than a false alarm.

**Threshold Selection (OOF Validation):**
To ensure the held-out test set remains strictly unseen during the decision-policy tuning, the optimal threshold was selected using **5-Fold Cross-Validation** on the training set (grouped by `sequence_id` to prevent sequence leakage).

We evaluated the total cost function `(FP * ₹300) + (FN * ₹5,500)` at every threshold from 0.0 to 1.0 on the out-of-fold validation predictions.
- The cost curve (`docs/threshold_cost_curve.png`) shows the minimum expected cost occurs at a threshold of **0.4335**.
- While the 18:1 cost asymmetry heavily favors recall, dropping the threshold below 0.4335 generates a steep spike in false positives that mathematically outweighs the cost of the remaining missed frauds.

**Final Held-Out Test Evaluation at Chosen Threshold (0.4335):**
Once the threshold was locked in, it was applied exactly once to the held-out test set.
- **Precision:** 0.1692
- **Recall:** 0.7333
- **F1 Score:** 0.2750
- **Confusion Matrix:** 11 True Positives, 54 False Positives, 323 True Negatives, 4 False Negatives.

Despite the lower threshold, the model still missed 4 sophisticated fraud cases. The precision drop (to 17%) reflects the aggressive tuning toward recall, accepting roughly 5 false alarms to catch 1 real fraud—a tradeoff perfectly aligned with the ₹5,500 vs ₹300 cost reality.
