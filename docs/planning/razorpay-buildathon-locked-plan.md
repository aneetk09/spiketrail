# Razorpay Buildathon — Locked Build Plan

Deadline: **September 4, 2026**. This document is the single execution path. Each stage lists: what you need decided/installed *before* starting it, the deliverable, and how you know it's done. Follow in order — don't skip ahead, don't go back except to fix a real bug.

---

## Locked decisions (do not revisit unless something breaks)

| Decision | Choice | Why |
|---|---|---|
| Detection timing | **Retrospective** | Real-time needs a strict pre-cutoff feature set — high leakage risk with no prior ML experience. Retrospective is still legitimate and matches the pitch. |
| Model | **Logistic regression** | Coefficients are directly readable — feeds the audit-trail story without extra tooling (no SHAP needed). |
| Dataset scale | **~2,000 records, ~4% fraud (~80 cases)** | Small enough to build fast, large enough that a ~16-case test set isn't pure noise. |
| LLM (explanation only) | **Anthropic API, cheapest/fastest available model** | One-line "why flagged" generation — not a reasoning task. |
| Deployment | **docker-compose, local-only** | No live-hosting failure surface in the last 48 hours. "Runs in 5 minutes" satisfies the judging bar. |
| Random seed | **Fixed constant, defined once, reused everywhere** | Reproducibility (addendum §6). |

---

## Stage 0 — Environment & repo skeleton
**Look into before starting:**
- Python version to pin (recommend 3.11)
- Create the public GitHub repo now, empty, so commit history starts honest from day one
- Folder layout: `/data/raw`, `/data/train`, `/data/test`, `/src`, `/api`, `/tests`, `/docs`
- Get an Anthropic API key now — don't discover you need one on Stage 10

**Deliverable:** repo pushed with `requirements.txt` (fastapi, uvicorn, scikit-learn, pandas, numpy, pydantic, anthropic, python-dotenv, pytest), `.gitignore`, MIT `LICENSE`, `config.py` with `SEED = 42`, empty README.

**Done when:** `git clone` + `pip install -r requirements.txt` works clean on nothing else installed.

---

## Stage 1 — Evaluation vocabulary (no code)
**Look into:** precision, recall, F1, confusion matrix terms (TP/FP/TN/FN), and work through one hand example using your actual planned scale (2,000 records, 4% fraud) so the numbers feel real before you see real ones.

**Deliverable:** a page in `docs/METHODOLOGY.md` (draft) explaining these in plain language with your worked example.

**Done when:** you could explain to a judge, without notes, why 95% accuracy can mean zero fraud caught.

---

## Stage 2 — Define the fraud pattern precisely
**Decide concrete numbers, not vague language:**
- N = minimum small transactions in a burst (suggest 3–5)
- X = max amount per small transaction (suggest ₹50–150)
- T = burst time window (suggest 5–15 minutes)
- W = window after burst in which the large transaction must land (suggest ≤30 minutes)
- Y = minimum amount to count as "the large transaction" (suggest ₹3,000+, or a multiple of the customer's typical spend)

**Look into:** briefly read how real card-testing/velocity fraud patterns are described (general fraud-detection writeups, not Kaggle data itself) — grounds your chosen numbers so they're not arbitrary (addendum §11).

**Deliverable:** the exact rule written down in `docs/PATTERN_DEFINITION.md`.

**Done when:** you can classify any hypothetical transaction sequence against the written rule with no ambiguity.

---

## Stage 3 — Synthetic data generation rules
**Decide granularity first:** each row = one transaction (not one sequence). A fraud sequence produces multiple rows sharing a `sequence_id`; the ground-truth `label` is per-transaction.

**Define three categories precisely, using Stage 2's numbers:**
1. **Fraud** — matches the rule exactly (~4% of records)
2. **Clean** — normal, unrelated spending, single or occasionally multiple purchases with no burst-then-large shape
3. **Ambiguous** (carved out of "clean," not a separate label) — resembles the pattern but isn't fraud: e.g. 3–4 quick small purchases (groceries → fuel → coffee) with *no* large transaction after, or a large legitimate purchase (rent, tuition) with unrelated small purchases earlier that day. **This category is what makes your eventual numbers credible — don't skip it or make it trivial.**

**Fields:** `transaction_id`, `sequence_id`, `customer_id`, `device_id`, `ip`, `amount`, `timestamp`, `label` (0/1)

**Deliverable:** generation rules written down before any generator code exists.

**Done when:** you could hand these rules to someone else and they'd produce a dataset that looks like yours.

---

## Stage 4 — Generate the dataset
**Look into:** use `numpy`'s fixed-seed RNG (not `random`) for reproducibility; spread timestamps across several simulated days so time-of-day features aren't degenerate.

**Deliverable:** `data/raw/transactions.csv`, plus a printed/asserted check that fraud ratio ≈ 4%.

**Done when:** the file exists, the ratio check passes, and you've eyeballed 10 random ambiguous rows and confirmed they genuinely look like real edge cases, not obvious fraud with the label flipped.

---

## Stage 5 — Held-out split (before ANY model work)
**Look into:** `sklearn.model_selection.train_test_split` with `stratify=label`, 80/20, fixed seed.

**Deliverable:** `data/train/train.csv`, `data/test/test.csv`. **Commit this as its own commit, before any model code exists** — this ordering in your commit history is part of your honesty story.

**Done when:** test.csv exists, is stratified, and you close the file and don't reopen it until Stage 8.

---

## Stage 6 — Feature engineering
**Build these as pure functions, fit on train stats only:**
- `tx_velocity` — count of transactions from same `device_id`/`ip` in preceding T minutes
- `amount_deviation` — this tx's amount vs. that customer's historical mean/std
- `time_of_day` pattern encoding
- `device_reuse_count` — distinct `customer_id`s seen on this device/IP recently
- `burst_ratio` — this tx's amount relative to the preceding burst's average

**Leakage check (addendum §4 — do not skip):** none of these features may directly reconstruct Stage 2's exact rule. Compute proxies (counts, ratios), not "does this match the fraud definition."

**Deliverable:** `src/features.py`, applied to train; same function applied to test later without refitting on test.

**Done when:** you've manually checked 5 fraud rows and 5 ambiguous rows and the feature values look like they'd actually help separate them — not identical, not a giveaway.

---

## Stage 7 — Train the model
**Look into:** `StandardScaler` fit on train only, `LogisticRegression(class_weight='balanced')` given the 4% imbalance.

**Deliverable:** trained model + scaler saved (`joblib`), trained on train split only.

**Done when:** model trains without error and coefficients exist for every feature.

---

## Stage 8 — Evaluate on the held-out test set
**Look into:** `sklearn.metrics` — precision, recall, F1, confusion matrix. Apply Stage 6's feature function (already fit on train) to test, then predict.

**Deliverable:** real numbers, whatever they are — even if mediocre.

**Done when:** you have a confusion matrix and can explain each cell.

---

## Stage 9 — False-positive cost & threshold
**Decide a concrete ₹ figure** for what a wrongly-blocked legitimate transaction costs (lost transaction value + a reasoned support/trust cost) and **label it as an assumption**, reasoning shown.

**Look into:** `precision_recall_curve` to pick a threshold that reflects the FP-cost vs. missed-fraud-cost tradeoff — not the default 0.5.

**Deliverable:** stated FP-cost figure + chosen threshold + the curve, in `docs/METHODOLOGY.md`.

**Done when:** you can defend the threshold choice in one sentence tied to the cost figure.

---

## Stage 10 — Explanation layer (the one LLM call)
**Look into:** Anthropic API key set as env var, cheapest/fastest model, a prompt template that feeds in only the feature values that drove the decision (so it can't hallucinate numbers not given to it).

**Deliverable:** for every flagged test transaction, a 1–2 sentence plain-language explanation.

**Done when:** explanations reference actual feature values, not generic fraud language.

---

## Stage 11 — Audit log
**Deliverable:** SQLite table or JSON log — `transaction_id`, `timestamp`, features used, decision, explanation — one write path so every decision is logged the same way.

**Done when:** every flagged transaction is traceable end to end from raw data to explanation.

---

## Stage 12 — Defense-only review
**Look into:** does anything public (Stage 2's exact N/X/T/W/Y numbers, model coefficients, exact threshold) let someone reverse-engineer how to stay under the detector? If yes, keep those specific values out of the public repo (config file, gitignored) and describe them only in general terms in docs.

**Done when:** you've explicitly checked this, not assumed it's fine.

---

## Stage 13 — API layer
**Deliverable:** FastAPI with `POST /detect` (batch), `GET /metrics`, `GET /audit-log`, pydantic request/response schemas.

---

## Stage 14 — Results view (static first)
**Deliverable:** one static/report-style page — headline precision/recall/F1, confusion matrix, transaction table with flags + explanations, FP-cost figure marked as assumption, audit log. Dark/muted palette, one accent color for "flagged," no red/green traffic lights. **Interactive dashboard only if time remains — do not start it before this works.**

---

## Stage 15 — Run in 5 minutes
**Look into:** test `docker-compose up` on a *fresh clone*, not your working directory — that's the actual reviewer experience.

**Deliverable:** working `docker-compose.yml` + README a stranger could follow.

---

## Stage 16 — Document the real failure
Ongoing from Stage 4 onward — keep a running note the moment something actually breaks. Do not wait, do not invent one on day 11.

---

## Stage 17 — Repo cleanup + tests
**Deliverable:** minimum 2 tests (feature function on a known input produces expected output; API returns expected shape) — addendum §9.

---

## Stage 18 — Pitch video
Script order: problem → what was built → live run showing metrics → the actual numbers → what broke and how you fixed it.

---

## Stage 19 — Application form
"What broke, and how you got out" is read first — write it from Stage 16's real note, not from memory.
**One thing on this whole list I can't decide for you:** the in-person-from-September Bangalore commitment is a personal logistics call, not a technical one — decide your honest answer before this stage, separately from the build.

---

## Buffer
Reserve the last full day before Sept 4 as pure slack — broken recording, last-minute bug, form taking longer than expected. Do not schedule submission for the deadline day itself.
