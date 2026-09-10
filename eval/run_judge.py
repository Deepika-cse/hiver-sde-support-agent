from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.config import groq_key, groq_model
from src.utils import ensure_parent

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# SHORT LLM-AS-JUDGE RUBRIC
# ============================================================

RUBRIC = """
Evaluate the AI support reply using only the supplied customer message,
context, and historical evidence.

Score 1-5:
groundedness = supported by historical evidence
helpfulness = useful response to the customer
correctness = accurate and avoids unsupported claims
tone = professional, concise, empathetic
overall = overall quality

If evidence is insufficient, do not reward unsupported claims.

Return ONLY this JSON object:
{
  "groundedness": 1,
  "helpfulness": 1,
  "correctness": 1,
  "tone": 1,
  "overall": 1,
  "reason": "short reason"
}
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text: str) -> dict:

    text = (text or "").strip()

    if not text:
        raise ValueError(
            "Judge returned an empty response."
        )

    # Remove markdown fences if the model accidentally adds them.
    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
            1,
        )

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

    # Some models may return extra text around JSON.
    # Try the complete response first.
    try:
        result = json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Judge response did not contain valid JSON."
            )

        result = json.loads(
            text[start:end + 1]
        )

    if not isinstance(result, dict):
        raise ValueError(
            "Judge response is not a JSON object."
        )

    required = [
        "groundedness",
        "helpfulness",
        "correctness",
        "tone",
        "overall",
        "reason",
    ]

    for key in required:

        if key not in result:
            raise ValueError(
                f"Missing judge field: {key}"
            )

    for key in required[:-1]:

        try:
            value = int(
                result[key]
            )
        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid score for {key}: "
                f"{result[key]}"
            ) from exc

        if value < 1 or value > 5:
            raise ValueError(
                f"Score for {key} must be 1-5."
            )

        result[key] = value

    result["reason"] = str(
        result["reason"]
    ).strip()

    return result


# ============================================================
# GROQ JUDGE
# ============================================================

class GroqJudge:

    def __init__(
        self,
        max_tokens: int = 180,
        max_retries: int = 1,
    ):

        key = groq_key()

        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        if Groq is None:
            raise RuntimeError(
                "groq package is not installed."
            )

        self.client = Groq(
            api_key=key
        )

        self.model = groq_model()

        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def run(
        self,
        user_prompt: str,
    ) -> dict:

        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):

            try:

                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict "
                                    "support-quality "
                                    "evaluator. "
                                    "Return ONLY valid "
                                    "JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        temperature=0,
                        max_tokens=self.max_tokens,
                    )
                )

                # ------------------------------------------------
                # Safely extract response content
                # ------------------------------------------------

                if not response.choices:
                    raise ValueError(
                        "Judge returned no choices."
                    )

                message = (
                    response
                    .choices[0]
                    .message
                )

                text = (
                    message.content
                    if message is not None
                    else ""
                )

                # ------------------------------------------------
                # Empty response
                # ------------------------------------------------

                if not text:

                    finish_reason = getattr(
                        response.choices[0],
                        "finish_reason",
                        "",
                    )

                    raise ValueError(
                        "Judge returned an empty "
                        f"response. finish_reason="
                        f"{finish_reason}"
                    )

                return parse_json(
                    text
                )

            except Exception as exc:

                last_error = exc

                error_text = str(
                    exc
                )

                lower = (
                    error_text.lower()
                )

                # --------------------------------------------
                # Retry only temporary failures.
                # --------------------------------------------

                retryable = (
                    "429" in error_text
                    or "rate limit" in lower
                    or "503" in error_text
                    or "timeout" in lower
                    or "temporarily" in lower
                    or "empty response" in lower
                    or "no choices" in lower
                )

                if (
                    attempt < self.max_retries
                    and retryable
                ):

                    wait = 4 * (
                        attempt + 1
                    )

                    print(
                        "Judge temporary/empty "
                        f"response. Retrying in {wait}s..."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                raise RuntimeError(
                    "Groq judge failed: "
                    f"{last_error}"
                )


# ============================================================
# BUILD SMALL JUDGE PROMPT
# ============================================================

def build_prompt(
    customer_text: str,
    context: str,
    draft_reply: str,
    historical_examples: list[dict],
) -> str:

    # Keep evidence deliberately small to reduce token usage.
    compact = []

    for item in historical_examples[:2]:

        compact.append(
            {
                "customer":
                    item.get(
                        "latest_customer_text",
                        "",
                    )[:500],

                "reply":
                    item.get(
                        "historical_brand_reply",
                        "",
                    )[:500],
            }
        )

    evidence_text = json.dumps(
        compact,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )

    prompt = f"""
CUSTOMER:
{customer_text[:1000]}

CONTEXT:
{context[:1000]}

DRAFT:
{draft_reply[:700]}

HISTORICAL EVIDENCE:
{evidence_text}

{RUBRIC}
"""

    return prompt


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--threads",
        required=True,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    if not groq_key():

        raise SystemExit(
            "GROQ_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (
        pd.read_csv(
            args.predictions,
            dtype=str,
        )
        .fillna("")
    )

    required_prediction_columns = [
        "case_id",
        "customer_text",
        "draft_reply",
        "evidence_case_ids",
    ]

    missing = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing:

        raise SystemExit(
            "Missing prediction columns: "
            + ", ".join(missing)
        )

    # Limit to requested number.
    predictions = predictions.head(
        args.n
    )

    # --------------------------------------------------------
    # Threads
    # --------------------------------------------------------

    threads = (
        pd.read_csv(
            args.threads,
            dtype=str,
        )
        .fillna("")
    )

    required_thread_columns = [
        "case_id",
        "latest_customer_text",
        "historical_brand_reply",
        "conversation_context",
    ]

    missing = [
        column
        for column in required_thread_columns
        if column not in threads.columns
    ]

    if missing:

        raise SystemExit(
            "Missing thread columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Evidence lookup
    # --------------------------------------------------------

    evidence_by_id = (
        threads[
            required_thread_columns
        ]
        .set_index("case_id")
        .to_dict("index")
    )

    # --------------------------------------------------------
    # Resume existing judge output
    # --------------------------------------------------------

    existing_rows = []

    try:

        existing = pd.read_csv(
            args.out,
            dtype=str,
        ).fillna("")

        if "case_id" in existing.columns:

            existing_rows = (
                existing
                .to_dict("records")
            )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
    ):

        existing_rows = []

    already_done = {
        str(row["case_id"])
        for row in existing_rows
        if str(
            row.get("case_id", "")
        ).strip()
    }

    if already_done:

        print(
            f"Existing judge results: "
            f"{len(already_done)}"
        )

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------
    judge = GroqJudge(
    max_tokens=400,
    max_retries=1,
)

    results = list(
        existing_rows
    )

    total = len(
        predictions
    )

    for i, (_, row) in enumerate(
        predictions.iterrows(),
        start=1,
    ):

        case_id = str(
            row["case_id"]
        )

        # Resume support.
        if case_id in already_done:

            print(
                f"[{i}/{total}] "
                f"{case_id} already judged"
            )

            continue

        print(
            f"[{i}/{total}] "
            f"Judging {case_id}..."
        )

        # ----------------------------------------------------
        # Evidence IDs
        # ----------------------------------------------------

        evidence_ids = [
            x.strip()
            for x in str(
                row.get(
                    "evidence_case_ids",
                    "",
                )
            ).split("|")
            if x.strip()
        ]

        historical_examples = []

        for evidence_id in evidence_ids:

            item = evidence_by_id.get(
                evidence_id
            )

            if item is None:
                continue

            historical_examples.append(
                item
            )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            customer_text=str(
                row.get(
                    "customer_text",
                    "",
                )
            ),
            context=str(
                row.get(
                    "conversation_context",
                    "",
                )
            ),
            draft_reply=str(
                row.get(
                    "draft_reply",
                    "",
                )
            ),
            historical_examples=historical_examples,
        )

        # ----------------------------------------------------
        # Call judge
        # ----------------------------------------------------

        try:

            score = judge.run(
                prompt
            )

        except Exception as exc:

            print(
                f"ERROR judging {case_id}: "
                f"{exc}"
            )

            # Do NOT fabricate scores.
            # Stop so we do not silently produce invalid
            # judge data.
            raise

        score["case_id"] = case_id

        results.append(
            score
        )

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        output = pd.DataFrame(
            results
        )

        ensure_parent(
            args.out
        )

        output.to_csv(
            args.out,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Saved {case_id}"
        )

    print()
    print(
        f"Judge results saved to: {args.out}"
    )

    print(
        f"Total completed: {len(results)}"
    )


if __name__ == "__main__":
    main()