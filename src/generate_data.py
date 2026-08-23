from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED
from src.pattern_config import PatternConfig, load_pattern_config


OUTPUT_PATH = Path("data/raw/transactions.csv")
TARGET_ROWS = 2_000
TARGET_FRAUD_RATIO = 0.04
SIMULATED_DAYS = 8


@dataclass(frozen=True)
class TransactionRow:
    transaction_id: str
    sequence_id: str
    customer_id: str
    device_id: str
    ip: str
    amount: float
    timestamp: pd.Timestamp
    label: int


def _customer_id(index: int) -> str:
    return f"cust_{index:04d}"


def _device_id(index: int) -> str:
    return f"dev_{index:04d}"


def _ip(index: int) -> str:
    return f"10.{index // 65_536 % 256}.{index // 256 % 256}.{index % 256}"


def _sequence_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def _timestamp(
    rng: np.random.Generator,
    base: pd.Timestamp,
    day: int | None = None,
    minute_floor: int = 0,
) -> pd.Timestamp:
    simulated_day = int(rng.integers(0, SIMULATED_DAYS)) if day is None else day
    minute = int(rng.integers(minute_floor, 24 * 60))
    second = int(rng.integers(0, 60))
    return base + pd.Timedelta(days=simulated_day, minutes=minute, seconds=second)


def _normal_amount(rng: np.random.Generator) -> float:
    amount = rng.lognormal(mean=np.log(650), sigma=0.75)
    return float(round(np.clip(amount, 60, 9_500), 2))


def _small_amount(rng: np.random.Generator, config: PatternConfig) -> float:
    low = max(5.0, config.max_small_transaction_amount_inr * 0.12)
    high = max(low + 1.0, config.max_small_transaction_amount_inr * 0.95)
    return float(round(rng.uniform(low, high), 2))


def _large_amount(rng: np.random.Generator, config: PatternConfig) -> float:
    high = max(config.min_large_transaction_amount_inr * 3.0, 7_500)
    return float(round(rng.uniform(config.min_large_transaction_amount_inr, high), 2))


def _append_row(
    rows: list[TransactionRow],
    sequence_id: str,
    customer_id: str,
    device_id: str,
    ip: str,
    amount: float,
    timestamp: pd.Timestamp,
    label: int,
) -> None:
    rows.append(
        TransactionRow(
            transaction_id=f"tx_{len(rows) + 1:06d}",
            sequence_id=sequence_id,
            customer_id=customer_id,
            device_id=device_id,
            ip=ip,
            amount=amount,
            timestamp=timestamp,
            label=label,
        )
    )


def _generate_fraud_rows(
    rng: np.random.Generator,
    config: PatternConfig,
    base: pd.Timestamp,
    target_fraud_rows: int,
) -> list[TransactionRow]:
    rows: list[TransactionRow] = []
    rows_per_sequence = config.min_small_transactions_in_burst + 1
    sequence_count = target_fraud_rows // rows_per_sequence

    for sequence_index in range(sequence_count):
        day = int(rng.integers(0, SIMULATED_DAYS))
        start = _timestamp(rng, base, day=day, minute_floor=37)
        customer = _customer_id(1_000 + sequence_index)
        device = _device_id(8_000 + int(rng.integers(0, max(4, sequence_count // 2))))
        ip = _ip(40_000 + int(rng.integers(0, max(4, sequence_count // 2))))
        sequence = _sequence_id("fraud", sequence_index)

        offsets = np.sort(
            rng.integers(
                0,
                max(1, config.burst_window_minutes * 60),
                size=config.min_small_transactions_in_burst,
            )
        )
        for offset in offsets:
            _append_row(
                rows,
                sequence,
                customer,
                device,
                ip,
                _small_amount(rng, config),
                start + pd.Timedelta(seconds=int(offset)),
                1,
            )

        followup_seconds = int(
            rng.integers(
                config.burst_window_minutes * 60 + 1,
                (
                    config.burst_window_minutes
                    + config.large_transaction_followup_window_minutes
                )
                * 60,
            )
        )
        _append_row(
            rows,
            sequence,
            customer,
            device,
            ip,
            _large_amount(rng, config),
            start + pd.Timedelta(seconds=followup_seconds),
            1,
        )

    return rows


def _generate_ambiguous_rows(
    rng: np.random.Generator,
    config: PatternConfig,
    base: pd.Timestamp,
    target_rows: int,
) -> list[TransactionRow]:
    rows: list[TransactionRow] = []
    sequence_index = 0

    while len(rows) < target_rows:
        scenario_options = ("small_only", "large_with_history", "large_unrelated")
        scenario = scenario_options[sequence_index % len(scenario_options)]
        customer = _customer_id(2_000 + sequence_index)
        device = _device_id(3_000 + int(rng.integers(0, 90)))
        ip = _ip(20_000 + int(rng.integers(0, 90)))
        sequence = _sequence_id(f"amb_{scenario}", sequence_index)
        day = int(rng.integers(0, SIMULATED_DAYS))
        start = _timestamp(rng, base, day=day, minute_floor=20)

        if scenario == "small_only":
            small_count = max(1, config.min_small_transactions_in_burst - 1)
            for offset in np.sort(
                rng.integers(
                    0,
                    max(1, config.burst_window_minutes * 60),
                    size=small_count,
                )
            ):
                if len(rows) >= target_rows:
                    break
                _append_row(
                    rows,
                    sequence,
                    customer,
                    device,
                    ip,
                    _small_amount(rng, config),
                    start + pd.Timedelta(seconds=int(offset)),
                    0,
                )
        elif scenario == "large_with_history":
            small_count = max(1, config.min_small_transactions_in_burst - 1)
            for offset in np.sort(
                rng.integers(
                    0,
                    max(1, config.burst_window_minutes * 60),
                    size=small_count,
                )
            ):
                if len(rows) >= target_rows:
                    break
                _append_row(
                    rows,
                    sequence,
                    customer,
                    device,
                    ip,
                    _small_amount(rng, config),
                    start + pd.Timedelta(seconds=int(offset)),
                    0,
                )
            if len(rows) < target_rows:
                outside_followup = (
                    config.burst_window_minutes
                    + config.large_transaction_followup_window_minutes
                    + int(rng.integers(20, 240))
                )
                _append_row(
                    rows,
                    sequence,
                    customer,
                    device,
                    ip,
                    _large_amount(rng, config),
                    start + pd.Timedelta(minutes=outside_followup),
                    0,
                )
        else:
            small_count = max(1, config.min_small_transactions_in_burst - 1)
            early_offsets = np.sort(
                rng.integers(
                    0,
                    max(1, config.burst_window_minutes * 60),
                    size=small_count,
                )
            )
            for offset in early_offsets:
                if len(rows) >= target_rows:
                    break
                _append_row(
                    rows,
                    sequence,
                    customer,
                    device,
                    ip,
                    _small_amount(rng, config),
                    start + pd.Timedelta(seconds=int(offset)),
                    0,
                )
            if len(rows) < target_rows:
                same_day_later = int(
                    rng.integers(
                        (
                            config.burst_window_minutes
                            + config.large_transaction_followup_window_minutes
                            + 60
                        ),
                        8 * 60,
                    )
                )
                _append_row(
                    rows,
                    sequence,
                    customer,
                    device,
                    ip,
                    _large_amount(rng, config),
                    start + pd.Timedelta(minutes=same_day_later),
                    0,
                )

        sequence_index += 1

    return rows[:target_rows]


def _generate_normal_rows(
    rng: np.random.Generator,
    base: pd.Timestamp,
    target_rows: int,
) -> list[TransactionRow]:
    rows: list[TransactionRow] = []

    for index in range(target_rows):
        customer_index = int(rng.integers(0, 420))
        sequence = _sequence_id("clean", index)
        _append_row(
            rows,
            sequence,
            _customer_id(customer_index),
            _device_id(customer_index + int(rng.integers(0, 29))),
            _ip(customer_index + int(rng.integers(0, 80))),
            _normal_amount(rng),
            _timestamp(rng, base),
            0,
        )

    return rows


def generate_transactions(config: PatternConfig) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(SEED)
    base = pd.Timestamp("2026-08-01T09:00:00")
    target_fraud_rows = int(TARGET_ROWS * TARGET_FRAUD_RATIO)
    ambiguous_target = 360

    fraud_rows = _generate_fraud_rows(rng, config, base, target_fraud_rows)
    ambiguous_rows = _generate_ambiguous_rows(rng, config, base, ambiguous_target)
    normal_rows = _generate_normal_rows(
        rng,
        base,
        TARGET_ROWS - len(fraud_rows) - len(ambiguous_rows),
    )

    all_rows = fraud_rows + ambiguous_rows + normal_rows
    rng.shuffle(all_rows)

    frame = pd.DataFrame([asdict(row) for row in all_rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["transaction_id"] = [f"tx_{index + 1:06d}" for index in range(len(frame))]

    ambiguous_sequences = sorted(
        {
            row.sequence_id
            for row in ambiguous_rows
            if row.sequence_id.startswith("amb_")
        }
    )
    sample_sequences = list(rng.choice(ambiguous_sequences, size=10, replace=False))
    return frame, sample_sequences


def validate(frame: pd.DataFrame) -> None:
    required_columns = {
        "transaction_id",
        "sequence_id",
        "customer_id",
        "device_id",
        "ip",
        "amount",
        "timestamp",
        "label",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != TARGET_ROWS:
        raise ValueError(f"Expected {TARGET_ROWS} rows, found {len(frame)}")
    if not frame["transaction_id"].is_unique:
        raise ValueError("transaction_id values must be unique")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1 and no other values")
    if frame["sequence_id"].isna().any():
        raise ValueError("Every row must have a sequence_id")

    timestamps = pd.to_datetime(frame["timestamp"])
    if timestamps.dt.date.nunique() < 3:
        raise ValueError("Timestamps must cover multiple simulated days")

    fraud_ratio = float(frame["label"].mean())
    if abs(fraud_ratio - TARGET_FRAUD_RATIO) > 0.005:
        raise ValueError(
            f"Fraud ratio {fraud_ratio:.3f} is not close to {TARGET_FRAUD_RATIO:.3f}"
        )


def main() -> None:
    config = load_pattern_config()
    frame, ambiguous_samples = generate_transactions(config)
    validate(frame)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Rows: {len(frame)}")
    print(f"Fraud rows: {int(frame['label'].sum())}")
    print(f"Fraud ratio: {frame['label'].mean():.3%}")
    print(
        "Date span: "
        f"{frame['timestamp'].min()} to {frame['timestamp'].max()}"
    )
    print("Ambiguous sequence samples for manual review:")
    for sequence_id in ambiguous_samples:
        sample = frame[frame["sequence_id"] == sequence_id]
        print(sample.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
