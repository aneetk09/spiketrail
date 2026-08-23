from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("config/pattern.demo.json")
CONFIG_ENV_VAR = "SPIKETRAIL_PATTERN_CONFIG"


@dataclass(frozen=True)
class PatternConfig:
    min_small_transactions_in_burst: int
    max_small_transaction_amount_inr: float
    burst_window_minutes: int
    large_transaction_followup_window_minutes: int
    min_large_transaction_amount_inr: float


def load_pattern_config(path: Path | None = None) -> PatternConfig:
    if path is None:
        path = Path(os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))

    if not path.exists():
        raise FileNotFoundError(
            f"Missing pattern config at {path}. "
            f"Set {CONFIG_ENV_VAR}=config/pattern.local.json for the private "
            "reported-results config, or use the committed demo config for a "
            "standalone pipeline run."
        )

    with path.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file)

    return PatternConfig(
        min_small_transactions_in_burst=int(raw["min_small_transactions_in_burst"]),
        max_small_transaction_amount_inr=float(raw["max_small_transaction_amount_inr"]),
        burst_window_minutes=int(raw["burst_window_minutes"]),
        large_transaction_followup_window_minutes=int(
            raw["large_transaction_followup_window_minutes"]
        ),
        min_large_transaction_amount_inr=float(raw["min_large_transaction_amount_inr"]),
    )
