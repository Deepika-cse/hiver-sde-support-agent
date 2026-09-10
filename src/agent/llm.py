from __future__ import annotations

import json
import time

from google import genai

from src.config import gemini_key, gemini_model
from src.utils import safe_json


class GeminiAgent:
    """
    Gemini-backed LLM wrapper.

    The rest of the support agent only needs:
        llm.run(system, user)

    This class handles:
    - Gemini API calls
    - JSON parsing
    - temporary 429/503 errors
    """

    def __init__(
        self,
        max_tokens: int = 700,
        max_retries: int = 2,
    ):
        key = gemini_key()

        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(api_key=key)
        self.model = gemini_model()
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def _parse_json(self, text: str) -> dict:
        """
        Parse Gemini's JSON response safely.
        """

        text = (text or "").strip()

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        # Remove accidental markdown fences.
        if text.startswith("```"):
            text = text.replace("```json", "", 1)
            text = text.replace("```", "", 1)
            text = text.strip()

        try:
            result = json.loads(text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Try existing project parser.
        try:
            result = safe_json(text)

            if isinstance(result, dict):
                return result

        except Exception:
            pass

        raise RuntimeError(
            "Gemini returned a response that could not be parsed as JSON."
        )

    def run(self, system: str, user: str) -> dict:
        """
        Call Gemini and return a JSON dictionary.

        Temporary 429/503 failures are retried.
        """

        prompt = f"""
SYSTEM INSTRUCTIONS:
{system}

CUSTOMER REQUEST:
{user}

IMPORTANT OUTPUT RULE:
Return ONLY one valid JSON object.
Do not use Markdown.
Do not use ```json.
Do not include explanations outside the JSON.
"""

        last_error = None

        for attempt in range(self.max_retries + 1):

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": self.max_tokens,
                        "response_mime_type": "application/json",
                    },
                )

                return self._parse_json(response.text)

            except Exception as exc:
                last_error = exc
                error_text = str(exc)

                retryable = (
                    "429" in error_text
                    or "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "high demand" in error_text.lower()
                    or "quota" in error_text.lower()
                )

                if not retryable:
                    raise

                if attempt < self.max_retries:
                    wait_seconds = 5 * (attempt + 1)

                    print(
                        f"Gemini temporary error. "
                        f"Waiting {wait_seconds}s before retry..."
                    )

                    time.sleep(wait_seconds)

        raise RuntimeError(
            "Gemini request failed after retries: "
            f"{last_error}"
        )