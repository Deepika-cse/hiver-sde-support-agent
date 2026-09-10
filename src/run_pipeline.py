from __future__ import annotations

import argparse

import pandas as pd

from src.agent.agent import SupportAgent
from src.agent.llm import GeminiAgent
from src.agent.groq import GroqAgent
from src.config import llm_provider
from src.retrieval.tfidf import HistoricalRetriever
from src.utils import ensure_parent


def load_cases(path: str) -> pd.DataFrame:
    """
    Load CSV data while preserving text and IDs.
    """

    return (
        pd.read_csv(
            path,
            dtype=str,
        )
        .fillna("")
    )


def select_evaluation_cases(
    threads: pd.DataFrame,
    case_ids_csv: str | None,
    n: int | None,
) -> pd.DataFrame:
    """
    Select cases for evaluation.

    If a golden/case-ID CSV is supplied, select those cases.

    If --n is also supplied, limit the selected
    cases to n.
    """

    if case_ids_csv:

        golden = load_cases(
            case_ids_csv
        )

        if "case_id" not in golden.columns:
            raise ValueError(
                f"{case_ids_csv} must contain "
                "a case_id column."
            )

        case_ids = golden[
            "case_id"
        ].tolist()

        indexed = threads.set_index(
            "case_id"
        )

        missing = [
            case_id
            for case_id in case_ids
            if case_id not in indexed.index
        ]

        if missing:
            raise ValueError(
                f"{len(missing)} case IDs were not found. "
                f"Examples: {missing[:5]}"
            )

        selected = (
            indexed
            .loc[case_ids]
            .reset_index()
        )

        if n is not None:
            selected = selected.head(
                n
            ).copy()

        return selected

    if n is not None:
        return threads.head(
            n
        ).copy()

    return threads.copy()


def build_retrieval_corpus(
    threads: pd.DataFrame,
    evaluation_cases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build retrieval corpus while excluding
    evaluation cases.

    This prevents the retriever from returning
    the exact test case as evidence.
    """

    evaluation_ids = set(
        evaluation_cases[
            "case_id"
        ].tolist()
    )

    return threads[
        ~threads[
            "case_id"
        ].isin(evaluation_ids)
    ].copy()


def create_llm():
    """
    Create the configured LLM provider.

    Supported providers:
        - gemini
        - groq
    """

    provider = llm_provider()

    if provider == "groq":

        print(
            "LLM: Groq / GPT-OSS 20B"
        )

        return GroqAgent(
max_tokens=1000,
    max_retries=2,
)

    if provider == "gemini":

        print(
            "LLM: Gemini"
        )

        return GeminiAgent(
            max_tokens=1000,
            max_retries=2,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Use 'groq' or 'gemini'."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the Hiver customer-support "
            "agent pipeline."
        )
    )

    parser.add_argument(
        "--threads_csv",
        required=True,
        help="Path to processed threads CSV.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of cases to process.",
    )

    parser.add_argument(
        "--case_ids_csv",
        default=None,
        help=(
            "CSV containing the case IDs "
            "to evaluate."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output predictions CSV.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without calling an LLM.",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    threads = load_cases(
        args.threads_csv
    )

    evaluation_cases = (
        select_evaluation_cases(
            threads=threads,
            case_ids_csv=args.case_ids_csv,
            n=args.n,
        )
    )

    print(
        f"Running pipeline on "
        f"{len(evaluation_cases)} cases..."
    )

    # --------------------------------------------------
    # Build retrieval corpus
    # --------------------------------------------------

    retrieval_corpus = (
        build_retrieval_corpus(
            threads=threads,
            evaluation_cases=evaluation_cases,
        )
    )

    print(
        f"Historical retrieval corpus: "
        f"{len(retrieval_corpus)} cases"
    )

    retriever = HistoricalRetriever(
        retrieval_corpus
    )

    # --------------------------------------------------
    # Create LLM
    # --------------------------------------------------

    if args.dry_run:

        llm = None

        print(
            "LLM: disabled (dry run)"
        )

    else:

        llm = create_llm()

    # --------------------------------------------------
    # Create agent
    # --------------------------------------------------

    agent = SupportAgent(
        retriever=retriever,
        llm=llm,
    )

    # --------------------------------------------------
    # Resume support
    # --------------------------------------------------

    completed = {}

    try:

        existing = pd.read_csv(
            args.out,
            dtype=str,
        ).fillna("")

        if "case_id" in existing.columns:

            for _, row in (
                existing.iterrows()
            ):

                completed[
                    str(row["case_id"])
                ] = row.to_dict()

            print(
                "Resuming from existing output: "
                f"{len(completed)} cases "
                "already completed."
            )

    except FileNotFoundError:

        pass

    rows = list(
        completed.values()
    )

    # --------------------------------------------------
    # Process cases
    # --------------------------------------------------

    for i, (_, row) in enumerate(
        evaluation_cases.iterrows(),
        start=1,
    ):

        case_id = str(
            row["case_id"]
        )

        if case_id in completed:

            print(
                f"[{i}/{len(evaluation_cases)}] "
                f"{case_id} -> "
                "already completed"
            )

            continue

        try:

            prediction = agent.predict(
                row.to_dict()
            )

            rows.append(
                prediction
            )

            completed[
                case_id
            ] = prediction

            print(
                f"[{i}/{len(evaluation_cases)}] "
                f"{case_id} -> "
                f"{prediction['intent']} / "
                f"{prediction['decision']}"
            )

            # Save after every successful case.
            output = pd.DataFrame(
                rows
            )

            ensure_parent(
                args.out
            )

            output.to_csv(
                args.out,
                index=False,
                encoding="utf-8-sig",
            )

        except Exception as exc:

            print(
                f"\nERROR on {case_id}:"
            )

            print(exc)

            print(
                "\nAlready completed predictions "
                "have been saved."
            )

            print(
                f"Output file: {args.out}"
            )

            break

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    output = pd.DataFrame(
        rows
    )

    ensure_parent(
        args.out
    )

    output.to_csv(
        args.out,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        "\n================ "
        "PIPELINE SUMMARY "
        "================"
    )

    print(
        f"Requested cases: "
        f"{len(evaluation_cases)}"
    )

    print(
        f"Completed cases: "
        f"{len(output)}"
    )

    if len(output) > 0:

        columns_to_show = [
            "case_id",
            "intent",
            "decision",
            "trust_score",
            "retrieval_top_score",
            "escalation_reason",
        ]

        available_columns = [
            column
            for column in columns_to_show
            if column in output.columns
        ]

        print(
            output[
                available_columns
            ].to_string(index=False)
        )

    print(
        f"\nSaved predictions to "
        f"{args.out}"
    )


if __name__ == "__main__":
    main()
