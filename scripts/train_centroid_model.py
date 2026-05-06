from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple centroid gesture classifier.")
    parser.add_argument("--input", default="data/gesture_samples.csv", help="Recorded landmark samples CSV.")
    parser.add_argument("--output", default="data/gesture_centroids.csv", help="Output centroid CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing sample file: {input_path}")

    data = pd.read_csv(input_path)
    if "label" not in data.columns:
        raise ValueError("Input CSV must contain a label column.")

    feature_columns = [column for column in data.columns if column.startswith("f")]
    centroids = data.groupby("label", as_index=False)[feature_columns].mean()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(output_path, index=False)
    print(f"Saved {len(centroids)} gesture centroids to {output_path}")


if __name__ == "__main__":
    main()
