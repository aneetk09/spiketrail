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

## Worked Example At Project Scale

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
