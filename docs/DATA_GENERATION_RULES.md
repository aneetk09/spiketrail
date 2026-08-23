# SpikeTrail Synthetic Data Generation Rules

No external dataset is supplied for SpikeTrail. The project generates a small,
controlled synthetic transaction dataset from scratch so the held-out
evaluation can be reproduced and audited.

This document is written before any generator code exists. It defines what the
generator must produce, what it must avoid, and how the resulting rows should
be interpreted.

## Dataset Shape

Each row is one transaction, not one customer and not one sequence. Related
transactions share a `sequence_id`.

The ground-truth `label` is per transaction:

- `1` means this transaction is part of a generated fraud-pattern sequence.
- `0` means this transaction is clean, including ambiguous-but-legitimate edge
  cases.

The dataset target is about 2,000 transaction rows with about 4% fraud rows.
The exact fraud-pattern numeric thresholds for reported results come from the
gitignored local pattern config and must not be committed. The committed demo
config is only for standalone reproducibility of the pipeline mechanics.

## Required Fields

The raw dataset must include these columns:

| Field | Purpose |
| --- | --- |
| `transaction_id` | Unique transaction identifier. |
| `sequence_id` | Shared identifier for transactions generated as a related sequence. |
| `customer_id` | Synthetic customer identifier. |
| `device_id` | Synthetic device identifier used for velocity and reuse features. |
| `ip` | Synthetic IP address used for velocity and reuse features. |
| `amount` | Transaction amount in rupees. |
| `timestamp` | Transaction timestamp across several simulated days. |
| `label` | Per-transaction ground truth: `1` for fraud-pattern rows, `0` for clean rows. |

## Category 1: Fraud-Pattern Rows

Fraud examples must match the private Stage 2 pattern config exactly:

1. A short burst of low-value transactions from the same suspicious device/IP
   context.
2. A larger follow-up transaction from that same context within the configured
   follow-up window.
3. A shared `sequence_id` across the related burst and follow-up transactions.
4. `label = 1` for every transaction in the fraud-pattern sequence.

Fraud sequences should vary enough to avoid looking copy-pasted:

- Amounts should vary within the private configured ranges.
- Timestamps should vary inside the private configured windows.
- Customers, devices, and IPs should include repeated risky contexts but not one
  single obvious identifier for all fraud.
- Fraud should appear across multiple simulated days and times.

The generator must not create a public column such as `matches_rule` or
`is_burst_then_cashout`. Labels are allowed; direct rule-reconstruction helper
columns are not.

## Category 2: Clean Normal Rows

Clean normal examples represent ordinary unrelated spending. They should make
up the majority of the dataset.

Clean normal rows may include:

- Single purchases.
- Occasional ordinary multi-transaction customer activity.
- A realistic amount distribution with many small and medium transactions and
  fewer larger purchases.
- Customer histories that make amount-deviation features meaningful.
- Device/IP reuse that happens at plausible low levels.

Clean normal rows must avoid accidentally matching the private fraud-pattern
config. If a generated clean sequence crosses the private rule, the generator
must alter it or label it consistently as fraud-pattern data.

## Category 3: Ambiguous Clean Rows

Ambiguous rows are legitimate edge cases carved out of the clean label, not a
third label. They are essential because a dataset with only obvious fraud and
obvious clean examples would be too easy and would make the metrics less
credible.

Ambiguous clean examples should include cases like:

- Several quick small purchases from the same customer/device/IP with no later
  larger transaction.
- A legitimate larger purchase with unrelated earlier small transactions that
  do not satisfy the private pattern config.
- Shared device/IP contexts that look mildly suspicious but do not complete the
  target sequence.
- Busy customer activity that raises velocity-like features without becoming
  fraud.

Every ambiguous row still has `label = 0`.

## Time And Identity Rules

Timestamps must be spread across several simulated days so time-of-day features
are not degenerate.

Customer histories must include enough prior spending to support historical
amount features. Device and IP identifiers must be reused sometimes, because
one-device-per-customer data would make reuse features meaningless.

The generator must use NumPy's fixed-seed random number generator. The seed is
defined once in `config.py` and reused everywhere.

## Validation Checks Required In The Generator

The generator must print or assert:

- Total row count is close to the planned dataset size.
- Fraud row ratio is close to the planned 4%.
- Required columns exist.
- `transaction_id` is unique.
- Every row has a non-empty `sequence_id`.
- Labels are only `0` or `1`.
- Timestamps cover multiple simulated days.
- A sample of ambiguous rows exists and can be manually inspected.

## Manual Review Requirement

After generation, manually inspect at least 10 ambiguous rows or sequences.
They should look like plausible legitimate edge cases, not obvious fraud with
the label flipped.

The generated CSV belongs in `data/raw/transactions.csv`, which is gitignored.
The project may commit generator code, documentation, the public demo config,
and split manifests, but not generated raw or split data files.
