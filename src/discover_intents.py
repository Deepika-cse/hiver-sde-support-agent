from __future__ import annotations

import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.utils import ensure_parent


def clean(s: str) -> str:
    s = re.sub(r"http\S+", " ", str(s))
    s = re.sub(r"@\w+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clusters", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(args.threads_csv).fillna("")
    texts = df["latest_customer_text"].map(clean)
    n_clusters = min(args.clusters, max(2, len(df)))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()

    rows = []
    for cluster in range(n_clusters):
        centroid = km.cluster_centers_[cluster]
        top = centroid.argsort()[-12:][::-1]
        examples = df.loc[labels == cluster, "latest_customer_text"].head(5).tolist()
        rows.append({
            "cluster": cluster,
            "size": int((labels == cluster).sum()),
            "keywords": ", ".join(terms[i] for i in top),
            "example_1": examples[0] if len(examples) > 0 else "",
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else "",
        })

    out = pd.DataFrame(rows).sort_values("size", ascending=False)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
