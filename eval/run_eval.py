from __future__ import annotations

import argparse
import json

import pandas as pd

from src.utils import ensure_parent
from eval.metrics import (
    classification_metrics,
    escalation_metrics,
    expected_calibration_error,
)


def parse_bool(series: pd.Series) -> pd.Series:
    """Safely convert CSV boolean values to actual booleans."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    pred = pd.read_csv(args.predictions, dtype=str).fillna("")
    gold = pd.read_csv(args.gold, dtype=str).fillna("")

    merged = gold.merge(
        pred,
        on="case_id",
        suffixes=("_gold", "_pred"),
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            "No matching case_id values were found between predictions and gold."
        )

    result = {
        "n": len(merged),
        "intent": classification_metrics(
            merged["gold_intent"],
            merged["intent"],
        ),
        "escalation": escalation_metrics(
            parse_bool(merged["gold_should_escalate"]),
            parse_bool(merged["should_escalate"]),
        ),
    }

    # Intent confidence calibration.
    if "intent_confidence" in merged.columns:
        confidence = pd.to_numeric(
            merged["intent_confidence"],
            errors="coerce",
        )

        valid = confidence.notna()

        if valid.any():
            correct = (
                merged.loc[valid, "gold_intent"]
                == merged.loc[valid, "intent"]
            ).astype(int)

            result["intent_ece"] = expected_calibration_error(
                correct,
                confidence.loc[valid],
            )

    out_dir = args.out_dir
    ensure_parent(f"{out_dir}/metrics.json")

    with open(
        f"{out_dir}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()