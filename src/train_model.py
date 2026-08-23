from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED
from src.features import FEATURE_COLUMNS


TRAIN_FEATURE_PATH = Path("data/train/train_features.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")
SCALER_PATH = Path("models/standard_scaler.joblib")


def train_model(
    train_feature_path: Path = TRAIN_FEATURE_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> tuple[LogisticRegression, StandardScaler, pd.Series]:
    if not train_feature_path.exists():
        raise FileNotFoundError(
            f"Missing train features at {train_feature_path}. "
            "Run python -m src.features first."
        )

    train = pd.read_csv(train_feature_path)
    missing = set(FEATURE_COLUMNS + ["label"]) - set(train.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"].astype(int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=1_000,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    coefficients = pd.Series(model.coef_[0], index=FEATURE_COLUMNS)
    return model, scaler, coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpikeTrail logistic model.")
    parser.add_argument("--train-features", type=Path, default=TRAIN_FEATURE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--scaler", type=Path, default=SCALER_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, coefficients = train_model(args.train_features, args.model, args.scaler)

    print(f"Wrote {args.model}")
    print(f"Wrote {args.scaler}")
    print(f"Training converged in {int(model.n_iter_[0])} iterations")
    print("Coefficients:")
    for feature, coefficient in coefficients.items():
        print(f"{feature}: {coefficient:.6f}")


if __name__ == "__main__":
    main()
