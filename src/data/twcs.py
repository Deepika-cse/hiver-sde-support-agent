from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from src.utils import parse_bool


REQUIRED = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id"
]


def _normalize_id(value) -> str:
    """Normalize IDs consistently when pandas reads integer IDs as floats."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    # CSVs with blank ID cells can cause pandas to parse IDs such as 1 as 1.0.
    # TWCS tweet IDs are integer-like strings, so remove only this artifact.
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"TWCS schema is missing columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df["inbound"] = df["inbound"].map(parse_bool)
    for c in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[c] = df[c].map(_normalize_id)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["created_at"] = df["created_at"].fillna("").astype(str)
    return df


def read_twcs(path: str | Path) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(path, low_memory=False))


def split_ids(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_cases(df: pd.DataFrame, brand_id: str, brand_name: str, max_context_turns: int = 6) -> pd.DataFrame:
    df = normalize_columns(df)
    by_id = {r.tweet_id: r for r in df.itertuples(index=False)}
    child_map: dict[str, list[str]] = {}
    for r in df.itertuples(index=False):
        for parent in split_ids(r.in_response_to_tweet_id):
            child_map.setdefault(parent, []).append(r.tweet_id)

    brand_id = str(brand_id)
    brand_rows = df[(~df["inbound"]) & (df["author_id"] == brand_id)]
    cases = []

    for r in brand_rows.itertuples(index=False):
        parent_id = str(r.in_response_to_tweet_id)
        customer = by_id.get(parent_id)
        if customer is None or not customer.inbound:
            continue

        # Walk backwards through customer/brand context.
        chain = [customer]
        current_parent = str(customer.in_response_to_tweet_id)
        seen = {customer.tweet_id}
        while current_parent and len(chain) < max_context_turns:
            prev = by_id.get(current_parent)
            if prev is None or prev.tweet_id in seen:
                break
            chain.append(prev)
            seen.add(prev.tweet_id)
            current_parent = str(prev.in_response_to_tweet_id)

        chain = list(reversed(chain))
        context_lines = []
        for turn in chain:
            speaker = "CUSTOMER" if turn.inbound else "BRAND"
            context_lines.append(f"{speaker}: {turn.text}")

        cases.append({
            "case_id": f"twcs_{r.tweet_id}",
            "brand_name": brand_name,
            "brand_author_id": brand_id,
            "latest_customer_tweet_id": customer.tweet_id,
            "brand_reply_tweet_id": r.tweet_id,
            "created_at": customer.created_at,
            "latest_customer_text": customer.text,
            "historical_brand_reply": r.text,
            "conversation_context": "\n".join(context_lines),
            "source_tweet_ids": "|".join(x.tweet_id for x in chain + [r]),
            "context_turn_count": len(chain),
        })

    out = pd.DataFrame(cases).drop_duplicates("case_id")
    return out


def discover_support_accounts(df: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    df = normalize_columns(df)
    outbound = df[~df["inbound"]].copy()
    rows = []
    for author_id, group in outbound.groupby("author_id"):
        replies = len(group)
        parents = group["in_response_to_tweet_id"].astype(str).tolist()
        parent_texts = []
        for pid in parents:
            if pid:
                match = df[df["tweet_id"] == pid]
                if not match.empty:
                    parent_texts.append(str(match.iloc[0]["text"]))
        handles = {}
        import re
        for text in parent_texts:
            for h in re.findall(r"@[A-Za-z0-9_]+", text):
                handles[h] = handles.get(h, 0) + 1
        likely = sorted(handles.items(), key=lambda x: x[1], reverse=True)[:5]
        rows.append({
            "author_id": str(author_id),
            "outbound_reply_count": replies,
            "inbound_parent_count": len(parent_texts),
            "likely_handles": ", ".join(f"{h}:{n}" for h, n in likely),
        })
    return pd.DataFrame(rows).sort_values("outbound_reply_count", ascending=False).head(top_k)
