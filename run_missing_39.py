import pandas as pd

from src.agent.groq import GroqAgent
from src.agent.prompts import system_prompt, user_prompt
from src.config import taxonomy
from src.retrieval.tfidf import HistoricalRetriever


GOLD_PATH = "data/golden/golden_set.csv"
THREADS_PATH = "data/processed/threads.csv"
EXISTING_PATH = "results/groq_predictions_180.csv"
OUTPUT_PATH = "results/groq_predictions_missing_39.csv"


def main():

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    gold = pd.read_csv(
        GOLD_PATH,
        dtype=str,
    ).fillna("")

    threads = pd.read_csv(
        THREADS_PATH,
        dtype=str,
    ).fillna("")

    existing = pd.read_csv(
        EXISTING_PATH,
        dtype=str,
    ).fillna("")

    # ---------------------------------------------------------
    # Identify cases that previously failed because of
    # provider rate limiting.
    # ---------------------------------------------------------

    rate_limit_mask = (
        existing["escalation_reason"]
        == "LLM rate limit reached; human review required."
    )

    missing_ids = set(
        existing.loc[
            rate_limit_mask,
            "case_id",
        ]
    )

    print(
        "Rate-limit cases found:",
        len(missing_ids),
    )

    if not missing_ids:
        print(
            "No missing LLM cases found."
        )
        return

    # ---------------------------------------------------------
    # Select the 39 cases from the golden set
    # ---------------------------------------------------------

    cases = gold[
        gold["case_id"].isin(missing_ids)
    ].copy()

    print(
        "Cases selected:",
        len(cases),
    )

    # ---------------------------------------------------------
    # Build retrieval corpus.
    #
    # Exclude all golden cases to prevent evaluation leakage.
    # ---------------------------------------------------------

    golden_ids = set(
        gold["case_id"]
    )

    retrieval_corpus = threads[
        ~threads["case_id"].isin(golden_ids)
    ].copy()

    print(
        "Retrieval corpus:",
        len(retrieval_corpus),
    )

    retriever = HistoricalRetriever(
        retrieval_corpus
    )

    # ---------------------------------------------------------
    # Create Groq LLM client
    # ---------------------------------------------------------

    llm = GroqAgent()

    # ---------------------------------------------------------
    # Run only missing cases
    # ---------------------------------------------------------

    results = []

    intents = taxonomy()

    for i, (_, case) in enumerate(
        cases.iterrows(),
        start=1,
    ):

        case_id = case["case_id"]

        print(
            f"[{i}/{len(cases)}] {case_id}"
        )

        evidence = retriever.search(
            case["latest_customer_text"],
            k=5,
            exclude_case_id=case_id,
        )

        raw = llm.run(
            system_prompt(intents),
            user_prompt(
                case["latest_customer_text"],
                case["conversation_context"],
                evidence,
            ),
        )

        evidence_ids = raw.get(
            "evidence_case_ids",
            [],
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            evidence_ids = []

        row = {
            "case_id": case_id,

            "customer_text":
                case["latest_customer_text"],

            "intent":
                raw.get(
                    "intent",
                    "other",
                ),

            "intent_confidence":
                raw.get(
                    "intent_confidence",
                    0.0,
                ),

            "draft_reply":
                raw.get(
                    "draft_reply",
                    "",
                ),

            "should_escalate":
                raw.get(
                    "should_escalate",
                    True,
                ),

            "decision":
                (
                    "escalate"
                    if raw.get(
                        "should_escalate",
                        True,
                    )
                    else "auto_handle"
                ),

            "escalation_reason":
                raw.get(
                    "escalation_reason",
                    "",
                ),

            "retrieval_top_score":
                (
                    evidence[0]["score"]
                    if evidence
                    else 0.0
                ),

            "evidence_consistency":
                0.0,

            "grounding_confidence":
                raw.get(
                    "grounding_confidence",
                    0.0,
                ),

            "risk_score":
                0.0,

            "trust_score":
                0.0,

            "evidence_case_ids":
                "|".join(
                    map(
                        str,
                        evidence_ids[:5],
                    )
                ),
        }

        results.append(row)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Rows:",
        len(output),
    )


if __name__ == "__main__":
    main()