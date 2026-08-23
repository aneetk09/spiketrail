from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from src.pattern_config import PatternConfig, load_pattern_config


TRAIN_PATH = Path("data/train/train.csv")
TRAIN_FEATURE_PATH = Path("data/train/train_features.csv")
FEATURE_STATE_PATH = Path("models/feature_state.joblib")
FEATURE_COLUMNS = [
    "tx_velocity",
    "amount_deviation",
    "time_of_day_sin",
    "time_of_day_cos",
    "device_reuse_count",
    "burst_ratio",
]


@dataclass(frozen=True)
class FeatureState:
    customer_amount_mean: dict[str, float]
    customer_amount_std: dict[str, float]
    global_amount_mean: float
    global_amount_std: float
    velocity_window_minutes: int
    reuse_window_minutes: int
    ratio_context_window_minutes: int


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "transaction_id",
        "sequence_id",
        "customer_id",
        "device_id",
        "ip",
        "amount",
        "timestamp",
        "label",
    }
    _require_columns(frame, required)
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["amount"] = prepared["amount"].astype(float)
    return prepared.sort_values("timestamp").reset_index(drop=True)


def fit_feature_state(
    train_frame: pd.DataFrame,
    config: PatternConfig | None = None,
) -> FeatureState:
    config = config or load_pattern_config()
    train = _prepare_frame(train_frame)

    grouped = train.groupby("customer_id")["amount"]
    customer_mean = grouped.mean().to_dict()
    customer_std = grouped.std(ddof=0).replace(0, np.nan).to_dict()
    global_mean = float(train["amount"].mean())
    global_std = float(train["amount"].std(ddof=0)) or 1.0

    clean_customer_std = {
        customer_id: float(std) if not pd.isna(std) and std > 0 else global_std
        for customer_id, std in customer_std.items()
    }

    return FeatureState(
        customer_amount_mean={
            customer_id: float(mean) for customer_id, mean in customer_mean.items()
        },
        customer_amount_std=clean_customer_std,
        global_amount_mean=global_mean,
        global_amount_std=global_std,
        velocity_window_minutes=config.burst_window_minutes,
        reuse_window_minutes=24 * 60,
        ratio_context_window_minutes=24 * 60,
    )


def add_tx_velocity(frame: pd.DataFrame, window_minutes: int) -> pd.Series:
    window = pd.Timedelta(minutes=window_minutes)
    values: list[int] = []
    for _, row in frame.iterrows():
        prior = frame[
            (frame["timestamp"] < row["timestamp"])
            & (frame["timestamp"] >= row["timestamp"] - window)
            & (
                (frame["device_id"] == row["device_id"])
                | (frame["ip"] == row["ip"])
            )
        ]
        values.append(int(len(prior)))
    return pd.Series(values, index=frame.index, dtype="int64")


def add_amount_deviation(frame: pd.DataFrame, state: FeatureState) -> pd.Series:
    means = frame["customer_id"].map(state.customer_amount_mean).fillna(
        state.global_amount_mean
    )
    stds = frame["customer_id"].map(state.customer_amount_std).fillna(
        state.global_amount_std
    )
    stds = stds.replace(0, state.global_amount_std)
    return ((frame["amount"] - means) / stds).astype(float)


def add_time_of_day(frame: pd.DataFrame) -> pd.DataFrame:
    seconds = (
        frame["timestamp"].dt.hour * 3600
        + frame["timestamp"].dt.minute * 60
        + frame["timestamp"].dt.second
    )
    radians = 2 * np.pi * seconds / (24 * 3600)
    return pd.DataFrame(
        {
            "time_of_day_sin": np.sin(radians),
            "time_of_day_cos": np.cos(radians),
        },
        index=frame.index,
    )


def add_device_reuse_count(frame: pd.DataFrame, window_minutes: int) -> pd.Series:
    window = pd.Timedelta(minutes=window_minutes)
    values: list[int] = []
    for _, row in frame.iterrows():
        prior = frame[
            (frame["timestamp"] < row["timestamp"])
            & (frame["timestamp"] >= row["timestamp"] - window)
            & (
                (frame["device_id"] == row["device_id"])
                | (frame["ip"] == row["ip"])
            )
            & (frame["customer_id"] != row["customer_id"])
        ]
        values.append(int(prior["customer_id"].nunique()))
    return pd.Series(values, index=frame.index, dtype="int64")


def add_burst_ratio(frame: pd.DataFrame, window_minutes: int) -> pd.Series:
    window = pd.Timedelta(minutes=window_minutes)
    values: list[float] = []
    for _, row in frame.iterrows():
        prior = frame[
            (frame["timestamp"] < row["timestamp"])
            & (frame["timestamp"] >= row["timestamp"] - window)
            & (
                (frame["device_id"] == row["device_id"])
                | (frame["ip"] == row["ip"])
            )
        ]
        if prior.empty:
            values.append(0.0)
            continue
        prior_average = float(prior["amount"].mean())
        raw_ratio = float(row["amount"] / prior_average) if prior_average else 0.0
        values.append(min(float(np.log1p(raw_ratio)), 8.0))
    return pd.Series(values, index=frame.index, dtype="float64")


def transform_features(frame: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    prepared = _prepare_frame(frame)
    features = pd.DataFrame(index=prepared.index)
    features["transaction_id"] = prepared["transaction_id"]
    features["label"] = prepared["label"].astype(int)
    features["tx_velocity"] = add_tx_velocity(
        prepared, state.velocity_window_minutes
    )
    features["amount_deviation"] = add_amount_deviation(prepared, state)
    time_features = add_time_of_day(prepared)
    features["time_of_day_sin"] = time_features["time_of_day_sin"]
    features["time_of_day_cos"] = time_features["time_of_day_cos"]
    features["device_reuse_count"] = add_device_reuse_count(
        prepared, state.reuse_window_minutes
    )
    features["burst_ratio"] = add_burst_ratio(
        prepared, state.ratio_context_window_minutes
    )
    return features[["transaction_id", *FEATURE_COLUMNS, "label"]]


def fit_transform_train(
    train_frame: pd.DataFrame,
    config: PatternConfig | None = None,
) -> tuple[pd.DataFrame, FeatureState]:
    state = fit_feature_state(train_frame, config)
    return transform_features(train_frame, state), state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit SpikeTrail train features.")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=TRAIN_FEATURE_PATH)
    parser.add_argument("--state", type=Path, default=FEATURE_STATE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    features, state = fit_transform_train(train)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    joblib.dump(asdict(state), args.state)

    print(f"Wrote {args.output} ({len(features)} rows)")
    print(f"Wrote {args.state}")
    print("Feature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")


if __name__ == "__main__":
    main()
