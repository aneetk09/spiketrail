# Razorpay AI Buildathon — Full Project Build Spec
**Self-contained document. No external context required. Written so a single capable AI assistant can take this document alone and build the complete project end to end.**

---

## 0. Background Context

**Razorpay** is a payments company running payments, risk, loyalty, and support infrastructure for millions of businesses in India. Razorpay is running an **AI Buildathon**: students build a real, working AI product solo, submit a public GitHub repo plus a 5-minute pitch video, and strong submissions are invited to an in-person round in Bangalore.

**How submissions are judged (stated directly by the organizers):** "We read the work, not the resume."
- **Problem taste** — did you pick something that actually matters
- **Build quality** — does it run, is it structured, would you trust it
- **AI judgment** — the right tool in the right place, and where you chose not to use one
- **Failure recovery** — what broke, and what you did about it

**Application form** (12 fields): full name, college, graduation year, in-person-from-September availability, 6-or-12-month program pick, resume, track, project name, what it solves, public GitHub repo URL, 5-minute pitch video, and **"what broke, and how you got out"** — this last field is explicitly stated to be read first, before anything else.

**The five available tracks (for context only — this project builds Track 02):**
1. AI Growth & Agentic Commerce — grow merchant revenue or make merchants transactable by AI buyers
2. **AI Risk Manager (this project)** — stop a merchant losing money to fraud, returns, or chargebacks
3. AI Revenue Recovery — detect and win back revenue at risk (failed payments, abandoned checkouts, overdue invoices)
4. AI Finance Controller — close a finance-ops loop across a 50+ record batch
5. Open Track — any real problem, meaningful AI use

**Why Track 02 specifically:** It has the hardest evaluation bar of the five — genuine precision/recall/false-positive-cost analysis on a held-out test set — and is the track most other applicants are likely to attempt but execute poorly, since honest, rigorous evaluation requires admitting a model's flaws with real numbers rather than presenting a polished but unverified result. That makes rigor and honesty the actual differentiator here, which matters because 109+ public repos already exist for this same buildathon across all tracks, including at least one other fraud-spike detector.

**Builder's background (for calibrating explanations and complexity level):** Final-year Computer Science student with strong iOS (Swift) and general full-stack experience (Python, Java/Spring Boot, React, FastAPI), but no prior project involving building, training, or rigorously evaluating a machine learning classifier — this is genuinely new territory. Comfortable learning new tools and concepts quickly with AI assistance, provided the learning is real rather than skipped over.

**Timeline:** Deadline September 4, 2026. Roughly 12 days available from the time this spec was written.

---

## 1. What to Build

A working fraud detector for one narrow, specific pattern: **a burst of small transactions from the same device/IP within a short time window, followed by one large transaction** (a "card-testing then cashout" pattern). It must be evaluated honestly on a held-out test set that is never touched during model-building, with every detection decision logged and explained in plain language.

**One-line description:** "A fraud-spike detector that catches burst-then-cashout transaction patterns, with measured precision/recall and honest false-positive cost — not just a model that looks accurate."

**Explicitly out of scope:** general-purpose or multi-pattern fraud detection, or any design choice optimized to look impressive rather than to be genuinely defensible under questioning. Narrow and rigorous is the goal, not broad and polished-looking.

---

## 2. The Exact Requirement to Satisfy

Verbatim from the track description:

> "Honest metrics including false-positive cost. Strictly defense-only: anything offense-capable is disqualified."

This means, concretely:
1. Report real precision, recall, and F1 on a genuinely held-out test set — not numbers tuned to look good.
2. Include a stated, reasoned estimate of what a false positive costs in practice (a wrongly-blocked legitimate customer).
3. Nothing in the public repository should function as a guide for evading this specific detector — do not publicly expose exact thresholds or feature weights in a way that would let someone reverse-engineer how to stay under them.

---

## 3. Fixed Design Decisions (build to these — do not redesign without a stated reason)

- **Detection method:** Classical machine learning (logistic regression or gradient boosting) trained on engineered features. Do NOT use an LLM as the actual fraud classifier.
- **LLM's only role:** Generating a human-readable explanation of why a specific transaction was flagged, after the classical model has already made the detection decision. The LLM never makes the detection decision itself.
- **Backend:** Python, using FastAPI.
- **Output/results presentation:** Start with a static or report-style results view (not a full interactive dashboard). Only build an interactive dashboard afterward, as optional polish, if time remains once the core pipeline and evaluation are solid.
- **Repository practice:** Public GitHub repo, with commits made incrementally throughout the build — never squashed into a single final commit. The commit history itself should show the honest build process (e.g., the held-out split happening before model training).
- **Deployment:** Either deploy on a free hosting tier (Render or Railway) or ensure the whole project runs with a single command locally (e.g., `docker-compose up`). Prioritize ease of review over hosting sophistication.
- **The failure story:** Must document something that genuinely goes wrong during the actual build process — never invent or dramatize a fake one.

---

## 4. Build Sequence (follow in order; each stage has a concrete deliverable)

1. **Establish the evaluation vocabulary.** Before writing any code, be able to explain precision, recall, F1, and false-positive cost concretely, ideally with a worked example. This project's entire value proposition rests on evaluation rigor, so this understanding must be genuine, not superficial.

2. **Define the fraud pattern precisely.** Write a clear, specific definition of what counts as a positive case for the chosen pattern (burst of small transactions from one device/IP, followed by one large transaction).

3. **Design synthetic data generation rules.** Define rules for generating three categories of transaction records:
   - Clearly fraudulent examples matching the defined pattern
   - Clearly clean/legitimate examples
   - **Genuinely ambiguous examples** — realistic edge cases that resemble the fraud pattern but are legitimate (e.g., a real customer making several quick small purchases — groceries, then fuel, then coffee). This category is the most important for making the eventual evaluation numbers credible; without it, the dataset will look artificially easy and undermine the whole exercise.

4. **Generate the dataset.** Produce the actual data file (CSV or JSON) with fields such as transaction ID, amount, timestamp, device/IP identifier, customer transaction history, and a ground-truth label, following the rules from step 3.

5. **Split off a held-out test set before any model-building begins.** This test set must not be touched again until final evaluation. This ordering should be reflected in the project's structure and commit history, since it's core to the honesty of the eventual metrics.

6. **Engineer features from the raw data.** Build features such as transaction velocity (transactions per unit time from the same device), amount deviation from a customer's typical spending, time-of-day pattern, and device/IP reuse count.

7. **Train the classical ML model** (logistic regression or gradient boosting) on the training portion only, using the engineered features.

8. **Evaluate strictly on the held-out test set.** Compute precision, recall, F1 score, and a confusion matrix. These numbers must come only from the untouched test set, never from data seen during training or tuning.

9. **Estimate and clearly state the false-positive cost.** Provide a reasoned estimate (e.g., in ₹) of the practical cost of a false positive, explicitly labeled as an assumption, with the reasoning behind the number shown transparently rather than presented as a hard fact.

10. **Build the explanation layer.** For each flagged transaction, generate a plain-language explanation of why it was flagged (e.g., "flagged because four small transactions occurred within 90 seconds from a new device, followed by one large transaction"). This is the one appropriate place to use an LLM call in this project.

11. **Build an audit log.** Record every detection decision with a timestamp, the features that drove the decision, and the generated explanation, so any decision can be inspected after the fact.

12. **Run a defense-only review.** Explicitly check whether anything exposed in the public repository (exact thresholds, feature weights, precise decision logic) could function as a guide for evading this detector. If so, redesign that part or keep the sensitive detail out of the public repository.

13. **Build the API layer** (FastAPI) exposing endpoints to run detection on a batch of transactions, retrieve the computed metrics, and retrieve the audit log.

14. **Build the results view.** Present the confusion matrix, headline metrics (precision/recall/F1), the false-positive cost figure, and a transaction-level table showing flag status and explanation. Follow the visual guidance in Section 5.

15. **Prepare for review.** Either deploy the project live on a free hosting tier, or ensure it runs from a single command locally. Write a README that lets an unfamiliar reviewer run the entire project in under 5 minutes.

16. **Document the real failure.** Once something genuinely breaks or goes wrong during the build (this cannot be predicted or scheduled — it must actually happen), write an honest account of what it was and how it was resolved.

17. **Clean up the repository.** Ensure a clear folder structure and a commit history that shows real incremental progress.

18. **Script and record a 5-minute pitch video** following this structure: the problem → what was built → a live run showing the metrics → the actual numbers → what broke and how it was resolved.

19. **Complete the application form**, giving particular care to the "what broke, and how you got out" field, since it is read first.

---

## 5. Visual and Experiential Guidance for the Results View

**Tone:** Serious, fintech-grade, and data-dense — closer to a trading terminal than a consumer app. Avoid playful visual elements. Avoid default red/green traffic-light coloring, which reads as unpolished in this context — use a deliberate, restrained palette instead: one accent color for "flagged," one neutral tone for "clean," and a dark or muted background.

**What a reviewer should see, in this order:**
1. A short README explaining the problem and how to run the project quickly.
2. The headline metrics (precision, recall, F1) as the first visually prominent element.
3. A clear confusion matrix.
4. A transaction-level table showing flag status and the plain-language explanation for each flagged transaction.
5. The stated false-positive cost figure, clearly marked as an assumption with its reasoning shown.
6. An inspectable audit log.

**Explicitly avoid:** a generic machine-learning tutorial notebook look, a demo that runs on obviously fake and cleanly-separable data, or a visually busy interface built to distract from unconvincing numbers.

---

## 6. Open Decisions to Resolve During the Build

- Exact feature set for the detector (precisely which signals to compute for velocity/deviation/pattern scoring)
- Model choice: logistic regression (simpler, more explainable) vs. gradient boosting (potentially more accurate, less transparent)
- Exact size and category ratio of the synthetic dataset
- The specific false-positive cost figure and its justification
- Whether to deploy live or ship as a one-command local run only
- Which LLM to call for the explanation-generation layer
- Final results-view format: static HTML, generated PDF, or a minimal web page
- The content of the real failure story — cannot be predetermined, must emerge honestly from the actual build

---

## 7. Known Risks to Actively Guard Against

- **Overly clean synthetic data** making precision/recall numbers look strong but meaningless to a skeptical reviewer — mitigated by the mandatory ambiguous-case category in step 3 of the build sequence.
- **Circular ground truth**, since both the data and its labels are self-generated — mitigate by documenting the label-generation logic transparently and being ready to explain and defend it directly.
- **Track crowding** — at least one very similar fraud-spike detector already exists publicly for this same buildathon — differentiate through genuine evaluation rigor and honesty rather than through the idea being novel.
- **Unfamiliar technical territory**, since no prior project involved this kind of ML work — mitigate by genuinely completing step 1 of the build sequence before proceeding, not skipping ahead.
- **Leaving the failure story and video for the very end** — mitigate by treating steps 16 through 19 as their own dedicated phase near the end of the timeline, not an afterthought squeezed into the final day.

---

*End of spec. This document is intended to contain everything needed — context, requirements, fixed decisions, build sequence, visual guidance, open questions, and risks — to build this project from a completely blank starting point.*
