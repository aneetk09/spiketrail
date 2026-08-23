# SpikeTrail Pattern Definition

SpikeTrail detects one narrow retrospective pattern: a completed burst of small
transactions from the same device/IP context followed soon after by a larger
transaction. The project treats this as a card-testing-then-cashout shape,
not as general-purpose fraud detection.

The exact numeric values used for reported results -- burst count,
small-transaction ceiling, burst window, follow-up window, and
large-transaction floor -- are intentionally not published. They live in
`config/pattern.local.json`, which is gitignored from the first commit that
introduced sensitive local configuration paths. The committed
`config/pattern.demo.json` contains illustrative demo parameters only, so a
fresh clone can still run the pipeline end to end.

## Public Rule Shape

A sequence is considered a positive fraud-pattern example only when all of
these conditions are true:

1. A customer/device/IP context produces a short burst of multiple low-value
   transactions.
2. The transactions in the burst share the same suspicious device/IP context.
3. A larger transaction from that same context happens shortly after the burst.
4. The larger transaction is part of the same generated sequence and is labeled
   as fraud along with the related burst transactions.

If the small burst exists but no later large transaction appears, the sequence
is not labeled fraud. If a large legitimate transaction appears without the
same preceding suspicious context, it is not labeled fraud. These near misses
become ambiguous clean examples during synthetic data generation.

## Why This Shape Is Realistic Enough For The Project

The pattern is grounded in common card-testing descriptions:

- Card testing commonly uses zero- or low-value transactions to validate stolen
  payment credentials before further use.
- Public fraud-prevention guidance describes velocity controls around repeated
  attempts from the same IP address, device, card, email, or customer profile.
- Public payment-risk writeups describe valid tested cards being used for larger
  purchases or sold after validation.

SpikeTrail narrows those public ideas into a small synthetic benchmark so the
project can evaluate precision, recall, false positives, and auditability in a
controlled way.

Public references checked during Stage 2:

- J.P. Morgan, "Preventing Card Testing: Spot It Early, Stop It Fast":
  https://www.jpmorgan.com/insights/payments/merchant-services/preventing-card-testing-attacks
- Mastercard, "Testing 1, 2, 3 ... cents? Why you shouldn't shrug off those
  tiny charges":
  https://www.mastercard.com/us/en/news-and-trends/stories/2024/testing-1-2-3-cents-why-you-shouldn-t-shrug-off-those-tiny-charges.html
- Visa Acceptance Support Center, "Mitigating Failed CVN Transactions and
  Preventing Card Testing Attacks":
  https://support.visaacceptance.com/knowledgebase/article/000003146/en-us
- Stripe Documentation, "Protect yourself from card testing":
  https://docs.stripe.com/disputes/prevention/card-testing

## Classification Examples

These examples intentionally omit the exact numeric cutoffs. The private local
config is the source of truth for the concrete values used by the generator.

| Scenario | Label |
| --- | --- |
| Several low-value transactions from the same device/IP, followed shortly by a larger transaction from that same context | Fraud-pattern sequence |
| Several low-value transactions from the same device/IP, with no later larger transaction | Clean ambiguous sequence |
| One large purchase with no suspicious preceding burst from the same context | Clean transaction |
| Low-value transactions spread too far apart to count as a burst | Clean or ambiguous, depending on surrounding context |
| Low-value transactions from unrelated devices/IPs | Clean or ambiguous, not the target fraud pattern |

## Defense-Only Boundary

This repository should explain the detector's intent and evaluation without
publishing a recipe for evading it. Public documents can describe the pattern
shape, methodology, and results. Exact thresholds, model coefficients, and final
decision thresholds must be reviewed before publication and kept out of the
public repo if they would make evasion easier.
