from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path("config/pattern.local.json")


@dataclass(frozen=True)
class PatternConfig:
    min_small_transactions_in_burst: int
    max_small_transaction_amount_inr: float
    burst_window_minutes: int
    large_transaction_followup_window_minutes: int
    min_large_transaction_amount_inr: float


def load_pattern_config(path: Path = CONFIG_PATH) -> PatternConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing private pattern config at {path}. "
            "Create this gitignored file before generating synthetic data."
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
