from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import SEED


RAW_PATH = Path("data/raw/transactions.csv")
TRAIN_PATH = Path("data/train/train.csv")
TEST_PATH = Path("data/test/test.csv")
MANIFEST_PATH = Path("docs/split_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_summary(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    label_counts = frame["label"].value_counts().sort_index()
    clean_rows = int(label_counts.get(0, 0))
    fraud_rows = int(label_counts.get(1, 0))
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "clean_rows": clean_rows,
        "fraud_rows": fraud_rows,
        "fraud_ratio": round(float(frame["label"].mean()), 6),
        "sha256": sha256_file(path),
    }


def write_split_manifest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw_path: Path,
    train_path: Path,
    test_path: Path,
    manifest_path: Path,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "seed": SEED,
        "split": {
            "method": "sklearn.model_selection.GroupShuffleSplit",
            "train_size": 0.8,
            "test_size": 0.2,
            "group_key": "sequence_id",
            "shuffle": True,
        },
        "data_visibility": {
            "raw_csv": "gitignored",
            "train_csv": "gitignored",
            "test_csv": "gitignored",
            "reason": (
                "Generated labeled transaction rows are not committed because "
                "they could reveal private fraud-pattern cutoffs."
            ),
        },
        "config_source_for_reported_results": (
            "config/pattern.local.json, gitignored private config"
        ),
        "raw": {
            "path": str(raw_path),
            "rows": int(len(train) + len(test)),
            "sha256": sha256_file(raw_path),
        },
        "train": split_summary(train, train_path),
        "test": split_summary(test, test_path),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def split_data(
    raw_path: Path = RAW_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw data at {raw_path}. Run python -m src.generate_data first."
        )

    raw = pd.read_csv(raw_path)
    
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=SEED)
    train_indices, test_indices = next(splitter.split(raw, groups=raw["sequence_id"]))
    
    train = raw.iloc[train_indices].copy()
    test = raw.iloc[test_indices].copy()

    train = train.sort_values("timestamp").reset_index(drop=True)
    test = test.sort_values("timestamp").reset_index(drop=True)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    write_split_manifest(train, test, raw_path, train_path, test_path, manifest_path)

    print(f"Wrote {train_path} ({len(train)} rows)")
    print(f"Wrote {test_path} ({len(test)} rows)")
    print(f"Wrote {manifest_path}")
    print(f"Train fraud ratio: {train['label'].mean():.3%}")
    print(f"Test fraud ratio: {test['label'].mean():.3%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SpikeTrail held-out split.")
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_data(args.raw, args.train, args.test, args.manifest)


if __name__ == "__main__":
    main()
