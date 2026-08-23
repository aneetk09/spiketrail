# SpikeTrail Current Build Locks

These locks were supplied after the original planning documents and override
older project choices where they conflict.

- Project name: SpikeTrail.
- Track: Razorpay AI Buildathon Track 02, AI Risk Manager.
- Detection timing: retrospective, after the full sequence completes.
- Model: logistic regression only; improve features if metrics are weak.
- Dataset: synthetic, generated from scratch, about 2,000 transactions with about 4% fraud.
- LLM role: Anthropic explanation generation only, never detection.
- Deployment: local-only with docker-compose.
- Results view: FastAPI-served static HTML and CSS, no frontend framework.
- Random seed: one fixed constant reused everywhere.
- Commit history: incremental stage commits; held-out split must precede model code.
- Defense-only rule: exact pattern thresholds live only in gitignored local config from the first commit that defines them.
