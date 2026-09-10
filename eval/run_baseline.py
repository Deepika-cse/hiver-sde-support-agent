from __future__ import annotations

import argparse
import pandas as pd

from src.baselines import (
    majority_baseline,
    keyword_baseline,
    tfidf_baseline,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="Training/reference data CSV",
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Golden evaluation CSV",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype=str).fillna("")
    gold = pd.read_csv(args.gold, dtype=str).fillna("")

    print(f"Training/reference cases: {len(train)}")
    print(f"Golden evaluation cases: {len(gold)}")

    # B0: majority class
    print("\nRunning majority baseline...")
    majority = majority_baseline(train, gold)

    pd.DataFrame(majority).to_csv(
        f"{args.out_dir}/majority_predictions.csv",
        index=False,
    )

    # B1: keyword rules
    print("Running keyword baseline...")
    keyword = keyword_baseline(
        gold.to_dict("records")
    )

    pd.DataFrame(keyword).to_csv(
        f"{args.out_dir}/keyword_predictions.csv",
        index=False,
    )

    # B2: TF-IDF + Logistic Regression
    print("Running TF-IDF baseline...")
    tfidf = tfidf_baseline(train, gold)

    pd.DataFrame(tfidf).to_csv(
        f"{args.out_dir}/tfidf_predictions.csv",
        index=False,
    )

    print("\nBaselines completed successfully.")


if __name__ == "__main__":
    main()