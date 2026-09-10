from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def load_yaml(name: str) -> dict:
    with open(
        ROOT / "configs" / name,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def taxonomy() -> list[dict]:
    return load_yaml("taxonomy.yaml")["intents"]


def thresholds() -> dict:
    return load_yaml("thresholds.yaml")


# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

def gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def gemini_model() -> str:
    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    )


# --------------------------------------------------
# Groq configuration
# --------------------------------------------------

def groq_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def groq_model() -> str:
    return os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )
def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()