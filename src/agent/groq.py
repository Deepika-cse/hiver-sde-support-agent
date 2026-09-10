
from __future__ import annotations

import json
import time

from groq import Groq

from src.config import groq_key, groq_model


ALLOWED_INTENTS = [
    "order_issue",
    "delivery_issue",
    "package_not_received",
    "return_cancellation",
    "refund_payment",
    "prime_subscription",
    "account_issue",
    "product_technical_issue",
    "complaint_followup",
    "other",
]


class GroqAgent:
    """
    Groq-backed LLM agent using GPT-OSS 20B.

    Uses JSON object mode with local validation.
    The implementation fails closed when the provider returns
    an incomplete/invalid response or a rate-limit error.
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        max_retries: int = 1,
    ):
        key = groq_key()

        if not key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=key)
        self.model = groq_model()
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _fallback(self, reason: str) -> dict:
        """
        Safe failure mode.

        Never invent a customer reply when the LLM response
        cannot be trusted.
        """
        return {
            "intent": "other",
            "intent_confidence": 0.0,
            "draft_reply": "",
            "should_escalate": True,
            "escalation_reason": reason,
            "grounding_confidence": 0.0,
            "evidence_case_ids": [],
        }

    def _validate(self, result: dict) -> dict:
        required_fields = [
            "intent",
            "intent_confidence",
            "draft_reply",
            "should_escalate",
            "escalation_reason",
            "grounding_confidence",
            "evidence_case_ids",
        ]

        missing = [
            field
            for field in required_fields
            if field not in result
        ]

        if missing:
            raise RuntimeError(
                "Groq response missing required fields: "
                + ", ".join(missing)
            )

        # Intent
        if result["intent"] not in ALLOWED_INTENTS:
            result["intent"] = "other"

        # Intent confidence
        try:
            result["intent_confidence"] = max(
                0.0,
                min(
                    1.0,
                    float(result["intent_confidence"]),
                ),
            )
        except (TypeError, ValueError):
            result["intent_confidence"] = 0.0

        # Grounding confidence
        try:
            result["grounding_confidence"] = max(
                0.0,
                min(
                    1.0,
                    float(result["grounding_confidence"]),
                ),
            )
        except (TypeError, ValueError):
            result["grounding_confidence"] = 0.0

        # Escalation
        value = result["should_escalate"]

        if isinstance(value, bool):
            result["should_escalate"] = value

        elif isinstance(value, str):
            result["should_escalate"] = (
                value.strip().lower()
                in {"true", "yes", "1"}
            )

        else:
            result["should_escalate"] = bool(value)

        # Reply
        if not isinstance(result["draft_reply"], str):
            result["draft_reply"] = ""

        # Escalation reason
        if not isinstance(result["escalation_reason"], str):
            result["escalation_reason"] = ""

        # Evidence
        if not isinstance(result["evidence_case_ids"], list):
            result["evidence_case_ids"] = []

        result["evidence_case_ids"] = [
            str(case_id)
            for case_id in result["evidence_case_ids"]
        ]

        return result

    def run(
        self,
        system: str,
        user: str,
    ) -> dict:

        last_error = None

        for attempt in range(self.max_retries + 1):

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    temperature=0,
                    max_completion_tokens=self.max_tokens,
                    response_format={
                        "type": "json_object"
                    },
                )

                content = response.choices[0].message.content

                if not content:
                    raise RuntimeError(
                        "Groq returned an empty response."
                    )

                result = json.loads(content)

                if not isinstance(result, dict):
                    raise RuntimeError(
                        "Groq response was not a JSON object."
                    )

                return self._validate(result)

            except Exception as exc:

                last_error = exc
                error_text = str(exc).lower()

                # ----------------------------------------------
                # Rate limit
                # ----------------------------------------------
                # Do NOT repeatedly hammer Groq when the
                # organization has exhausted its daily quota.
                if (
                    "429" in error_text
                    or "rate limit" in error_text
                    or "tokens per day" in error_text
                    or "rate_limit_exceeded" in error_text
                ):
                    return self._fallback(
                        "LLM rate limit reached; human review required."
                    )

                # ----------------------------------------------
                # Temporary service errors
                # ----------------------------------------------
                temporary = (
                    "503" in error_text
                    or "service unavailable" in error_text
                    or "timeout" in error_text
                )

                if temporary and attempt < self.max_retries:
                    wait_seconds = 3 * (attempt + 1)

                    print(
                        "Groq temporary error. "
                        f"Retrying in {wait_seconds}s..."
                    )

                    time.sleep(wait_seconds)
                    continue

                # ----------------------------------------------
                # JSON generation/validation failure
                # ----------------------------------------------
                json_failure = (
                    "json_validate_failed" in error_text
                    or "failed to generate json" in error_text
                    or "failed to validate json" in error_text
                    or "missing required fields" in error_text
                    or "empty response" in error_text
                )

                if json_failure and attempt < self.max_retries:
                    print(
                        "Groq JSON generation failed. "
                        f"Retrying in 3s..."
                    )

                    time.sleep(3)
                    continue

                # ----------------------------------------------
                # Final safe failure
                # ----------------------------------------------
                return self._fallback(
                    "LLM response was incomplete or invalid; "
                    "human review required."
                )

        return self._fallback(
            f"LLM request failed: {last_error}"
        )
