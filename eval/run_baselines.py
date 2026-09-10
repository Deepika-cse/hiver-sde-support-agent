from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.baselines import (
    majority_baseline,
    keyword_baseline,
    tfidf_baseline,
)
from eval.metrics import classification_metrics, escalation_metrics


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def evaluate(name, predictions, gold):
    pred = pd.DataFrame(predictions)

    merged = gold.merge(
        pred,
        on="case_id",
        suffixes=("_gold", "_pred"),
    )

    intent_metrics = classification_metrics(
        merged["gold_intent"],
        merged["intent"],
    )

    escalation = escalation_metrics(
        parse_bool(merged["gold_should_escalate"]),
        parse_bool(merged["should_escalate"]),
    )

    return {
        "baseline": name,
        "n": len(merged),
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "escalation_precision": escalation["precision"],
        "escalation_recall": escalation["recall"],
        "escalation_f1": escalation["f1"],
        "false_auto_handle_count": escalation["false_auto_handle_count"],
        "false_auto_handle_rate": escalation["false_auto_handle_rate"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        required=True,
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gold = pd.read_csv(
        args.gold,
        dtype=str,
    ).fillna("")

    print(f"Total labelled cases: {len(gold)}")

    # Split the labelled data into baseline training and holdout test data.
    train, test = train_test_split(
        gold,
        test_size=0.20,
        random_state=42,
        stratify=gold["gold_intent"],
    )

    print(f"Baseline training cases: {len(train)}")
    print(f"Baseline holdout cases: {len(test)}")

    # ---------------------------------------------------------
    # B0: Majority baseline
    # ---------------------------------------------------------
    print("\nRunning majority baseline...")

    majority_predictions = majority_baseline(
        train,
        test,
    )

    pd.DataFrame(majority_predictions).to_csv(
        f"{args.out_dir}/majority_predictions.csv",
        index=False,
    )

    majority_result = evaluate(
        "majority",
        majority_predictions,
        test,
    )

    # ---------------------------------------------------------
    # B1: Keyword baseline
    # ---------------------------------------------------------
    print("Running keyword baseline...")

    keyword_predictions = keyword_baseline(
        test.to_dict("records"),
    )

    pd.DataFrame(keyword_predictions).to_csv(
        f"{args.out_dir}/keyword_predictions.csv",
        index=False,
    )

    keyword_result = evaluate(
        "keyword",
        keyword_predictions,
        test,
    )

    # ---------------------------------------------------------
    # B2: TF-IDF + Logistic Regression
    # ---------------------------------------------------------
    print("Running TF-IDF + Logistic Regression baseline...")

    tfidf_predictions = tfidf_baseline(
        train,
        test,
    )

    pd.DataFrame(tfidf_predictions).to_csv(
        f"{args.out_dir}/tfidf_predictions.csv",
        index=False,
    )

    tfidf_result = evaluate(
        "tfidf_logistic_regression",
        tfidf_predictions,
        test,
    )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------
    results = pd.DataFrame(
        [
            majority_result,
            keyword_result,
            tfidf_result,
        ]
    )

    results.to_csv(
        f"{args.out_dir}/baseline_comparison.csv",
        index=False,
    )

    print("\n================ BASELINE RESULTS ================")
    print(results.to_string(index=False))

    print(
        f"\nSaved baseline results to "
        f"{args.out_dir}/baseline_comparison.csv"
    )


if __name__ == "__main__":
    main()