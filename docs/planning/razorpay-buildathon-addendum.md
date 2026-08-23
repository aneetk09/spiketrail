# Razorpay Buildathon — Addendum: Additional Considerations
**Things not covered in the stage list or full build spec — some of these are genuinely important, not just nice-to-haves.**

---

## 1. A real open decision: WHEN does detection happen?

*(Resolved in the locked plan: retrospective, not real-time.)*

This is a substantive gap, not a minor addition. There are two very different versions of this project:

- **Retrospective detection:** the burst-then-cashout pattern has already fully happened, and the system flags it after the fact for review/refund/chargeback action.
- **Real-time / predictive detection:** the system tries to catch the pattern *before* the final large transaction goes through — i.e., after seeing the burst of small transactions, predict that a large fraudulent one is coming and block it pre-emptively.

These require different feature engineering (real-time can only use data available *before* the large transaction; retrospective can use the whole completed sequence) and represent very different levels of difficulty and business value.

## 2. Class imbalance — real fraud data is rare, and this needs to be reflected and handled

In reality, the overwhelming majority of transactions are legitimate — fraud is a small minority. If the synthetic dataset is generated with, say, 40% fraud cases for convenience, it will look nothing like a real fraud problem and a knowledgeable judge will notice immediately. The dataset should reflect a realistic imbalance (fraud as a small minority), and — critically — **accuracy becomes a misleading metric under imbalance** (a model that always predicts "clean" can still score 95%+ accuracy while catching zero fraud). This is exactly why precision/recall/F1 matter more than accuracy here, and the write-up should explicitly say so — it signals real understanding of the problem, not just following instructions.

## 3. Threshold selection tied to the false-positive cost

Most classifiers output a probability, and a decision threshold turns that into a yes/no flag. The naive approach uses a default threshold (0.5) and reports whatever precision/recall comes out. The stronger approach: explicitly choose the threshold based on the false-positive cost estimate — e.g., "given that a false positive costs an estimated ₹X and a missed fraud costs ₹Y, we chose a threshold that optimizes for [X tradeoff]." This directly ties two required pieces (metrics and cost) together in a way that shows genuine reasoning rather than default settings, and is a natural place to visually show a precision-recall curve or cost curve in the results view.

## 4. Data leakage risk in feature engineering

Worth being careful that features don't accidentally "give away" the answer — e.g., if the label itself was generated using the exact same rule that a feature also encodes, the model will trivially achieve perfect scores that mean nothing. This is a common, embarrassing mistake in synthetic-data ML projects and worth explicitly checking for once features are built.

## 5. Model interpretability as part of the "audit trail" story

*(Resolved in the locked plan: logistic regression.)*

Since the whole track values explainability, it's worth choosing (or at least trying) logistic regression first — its coefficients are directly interpretable ("device reuse count contributed +0.8 to the fraud score"), which reinforces the audit-trail and explanation-layer story more naturally than a gradient-boosted model would (which would need an additional tool like SHAP to explain, adding complexity for arguably little benefit at this scope).

## 6. Reproducibility

Set and document a fixed random seed for both data generation and model training. This lets a reviewer re-run the project and get the same numbers — a small detail that meaningfully supports "would you trust it."

## 7. Environment and dependency hygiene

A pinned `requirements.txt` (or a lockfile) and a stated Python version. This is a common point of friction for reviewers trying to run someone else's project quickly — worth getting right given the stated "under 5 minutes to run" bar.

## 8. A short, honest methodology and limitations write-up

Separate from the README's "how to run this," a short section (in the README or a separate `METHODOLOGY.md`) stating plainly: how the synthetic data was generated, what its limitations are, what assumptions were made (especially the false-positive cost figure), and what this system would need before being production-ready. This kind of honest self-assessment directly matches "failure recovery" and "build quality" judging criteria, and is rare enough in student submissions that it stands out.

## 9. Basic automated tests

Even a handful of tests (does the feature engineering function produce expected outputs on a known input, does the API return the expected shape) meaningfully supports "would you trust it" — a repo with zero tests reads as less trustworthy than one with even minimal coverage.

## 10. A LICENSE file

Minor, but a public repo with no license technically has no clear usage terms. A simple MIT license is standard for a project like this and takes two minutes.

## 11. Grounding the synthetic data in something real, not pure invention

Consider briefly researching how real card-testing/fraud-spike patterns actually look (public references like the Kaggle credit card fraud dataset, or general fraud-detection literature on velocity-based patterns) — not to use real data, but to make sure the synthetic pattern rules reflect real-world shape rather than a guessed-at pattern. This directly strengthens "problem taste."

## 12. Timeline buffer

Reserve the last 1 full day purely as buffer — for a broken video recording, a last-minute bug, or the application form itself taking longer than expected. Don't schedule the submission for the literal deadline day.

## 13. The in-person Bangalore commitment question on the form

This is a real yes/no question on the application with real consequences, not just project material — worth actually deciding your honest answer to this before submitting, separately from anything about the build itself.

---

*This addendum should be read alongside the full build spec and the locked build plan — it fills gaps in technical rigor, reviewer trust, and a couple of decisions that were flagged open and have since been resolved.*
