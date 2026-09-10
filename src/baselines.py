from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INTENTS = [
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


def majority_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> List[Dict]:
    """
    B0: Majority-class baseline.

    The majority intent is calculated from labelled training data,
    not from the evaluation set.
    """
    counts = Counter(train_df["gold_intent"].astype(str))

    majority_intent = counts.most_common(1)[0][0]

    return [
        {
            "case_id": row["case_id"],
            "intent": majority_intent,
            "should_escalate": False,
            "draft_reply": (
                "Thanks for reaching out. Please contact support "
                "for further assistance."
            ),
            "baseline": "majority",
        }
        for _, row in eval_df.iterrows()
    ]


KEYWORDS = {
    "order_issue": [
        "order",
        "ordered",
        "wrong order",
        "order issue",
    ],
    "delivery_issue": [
        "delivery",
        "shipping",
        "shipment",
        "tracking",
        "late",
        "delayed",
        "arrive",
        "arrived",
    ],
    "package_not_received": [
        "delivered",
        "not received",
        "didn't receive",
        "did not receive",
        "missing package",
        "package missing",
    ],
    "return_cancellation": [
        "return",
        "cancel",
        "cancellation",
        "send back",
    ],
    "refund_payment": [
        "refund",
        "refunded",
        "charged",
        "charge",
        "payment",
        "billing",
        "card",
        "money",
    ],
    "prime_subscription": [
        "prime",
        "prime membership",
        "prime trial",
        "membership",
        "subscription",
    ],
    "account_issue": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "locked",
        "verification",
    ],
    "product_technical_issue": [
        "not working",
        "doesn't work",
        "does not work",
        "broken",
        "device",
        "fire tv",
        "echo",
        "kindle",
        "technical",
    ],
    "complaint_followup": [
        "still",
        "again",
        "follow up",
        "follow-up",
        "case",
        "previous",
        "already contacted",
        "no response",
    ],
}


def keyword_baseline(cases) -> List[Dict]:
    """
    B1: Simple keyword-based intent classifier.

    Uses the same final 10-intent taxonomy as the AI agent.
    """
    out = []

    for r in cases:
        text = str(r["latest_customer_text"]).lower()

        scores = {
            intent: sum(
                1 for keyword in keywords
                if keyword in text
            )
            for intent, keywords in KEYWORDS.items()
        }

        best_intent = max(scores, key=scores.get)

        if scores[best_intent] == 0:
            best_intent = "other"

        should_escalate = any(
            keyword in text
            for keyword in [
                "hacked",
                "fraud",
                "lawsuit",
                "legal",
                "police",
                "danger",
                "threat",
            ]
        )

        out.append(
            {
                "case_id": r["case_id"],
                "intent": best_intent,
                "should_escalate": should_escalate,
                "draft_reply": (
                    "Thanks for contacting us. "
                    "Please share more details so we can help."
                ),
                "baseline": "keyword",
            }
        )

    return out


def tfidf_baseline(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> List[Dict]:
    """
    B2: Simple TF-IDF + Logistic Regression classifier.

    This is intentionally a conventional non-LLM baseline.
    """

    train_text = train_df["latest_customer_text"].astype(str)
    eval_text = eval_df["latest_customer_text"].astype(str)

    train_labels = train_df["gold_intent"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_eval = vectorizer.transform(eval_text)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(X_train, train_labels)

    predictions = classifier.predict(X_eval)

    out = []

    for row, prediction in zip(
        eval_df.to_dict("records"),
        predictions,
    ):
        text = str(row["latest_customer_text"]).lower()

        should_escalate = any(
            keyword in text
            for keyword in [
                "hacked",
                "fraud",
                "lawsuit",
                "legal",
                "police",
                "danger",
                "threat",
            ]
        )

        out.append(
            {
                "case_id": row["case_id"],
                "intent": prediction,
                "should_escalate": should_escalate,
                "draft_reply": (
                    "Thanks for contacting us. "
                    "Please share more details so we can help."
                ),
                "baseline": "tfidf_logistic_regression",
            }
        )

    return out