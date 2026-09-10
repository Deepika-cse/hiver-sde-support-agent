from __future__ import annotations

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.utils import ensure_parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads_csv", required=True)
    p.add_argument("--target_n", type=int, default=180)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.threads_csv).fillna("")
    n = min(args.target_n, len(df))
    if len(df) <= n:
        chosen = df.copy()
    else:
        text = df["latest_customer_text"].astype(str)
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(text)
        clusters = min(max(10, n // 15), len(df))
        km = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        selected = []
        for c in range(clusters):
            idx = df.index[labels == c].tolist()
            if idx:
                selected.append(idx[0])
        # Fill remaining slots with deterministic spaced rows.
        remaining = [i for i in df.index if i not in selected]
        selected.extend(remaining[: max(0, n - len(selected))])
        chosen = df.loc[selected[:n]].copy()

    chosen["gold_intent"] = ""
    chosen["gold_secondary_intent"] = ""
    chosen["gold_should_escalate"] = ""
    chosen["gold_escalation_reason"] = ""
    chosen["gold_reply_notes"] = ""
    chosen["labeler"] = ""
    chosen["label_version"] = "v1"
    ensure_parent(args.out)
    chosen.to_csv(args.out, index=False)
    print(f"Created {len(chosen)} rows for manual labeling: {args.out}")


if __name__ == "__main__":
    main()
