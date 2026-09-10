from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:
    """TF-IDF baseline/retriever. No external model download required."""

    def __init__(self, cases: pd.DataFrame):
        self.cases = cases.reset_index(drop=True).copy()
        texts = (
            self.cases["latest_customer_text"].fillna("")
            + " "
            + self.cases["conversation_context"].fillna("")
        ).tolist()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=60000,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 5, exclude_case_id: str | None = None):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)

        results = []
        for idx in order:
            row = self.cases.iloc[int(idx)]
            if exclude_case_id and row["case_id"] == exclude_case_id:
                continue
            results.append({
                "case_id": row["case_id"],
                "score": float(sims[idx]),
                "customer_text": row["latest_customer_text"],
                "historical_reply": row["historical_brand_reply"],
                "created_at": row["created_at"],
            })
            if len(results) >= k:
                break
        return results
