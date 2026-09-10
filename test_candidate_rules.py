import pandas as pd
from src.agent.agent import SupportAgent

p = pd.read_csv(
    "results/groq_predictions_141_base.csv",
    dtype=str
).fillna("")

g = pd.read_csv(
    "data/golden/golden_set.csv",
    dtype=str
).fillna("")

a = SupportAgent(None)


def candidate_rule(text, intent):
    text = text.lower().strip()

    # 1. Paid delivery/service not received
    paid_delivery = (
        ("paid for" in text or "paid enough" in text)
        and (
            "delivery" in text
            or "deliver" in text
            or "arrive" in text
        )
        and (
            "didn't arrive" in text
            or "did not arrive" in text
            or "wasn't delivered" in text
            or "was not delivered" in text
        )
    )

    # 2. Marked delivered but customer says not received
    delivered_not_received = (
        (
            "says delivered" in text
            or "marked as delivered" in text
            or "been delivered" in text
            or "was delivered" in text
            or "dlvd" in text
        )
        and (
            "not delivered" in text
            or "can't see" in text
            or "washed their hands" in text
            or "not received" in text
        )
    )

    # 3. Refund/transaction money missing
    refund_missing = (
        (
            "refund issued" in text
            or "redund issued" in text
            or "refund" in text
        )
        and (
            "no sign" in text
            or "no sing" in text
            or "no signof" in text
            or "money" in text
            or "billing department" in text
        )
    )

    # 4. Explicit product supportability question
    supportability = (
        "no longer supported" in text
        or "still supported" in text
        or "supported anymore" in text
    )

    return (
        paid_delivery
        or delivered_not_received
        or refund_missing
        or supportability
    )


x = p.merge(
    g[["case_id", "gold_should_escalate"]],
    on="case_id"
)

x["existing_rules"] = x.apply(
    lambda r: (
        a._account_sensitive(r["customer_text"], r["intent"])
        or a._persistent_issue(r["customer_text"])
        or a._insufficient_information(r["customer_text"])
        or a._payment_or_transaction_issue(r["customer_text"])
        or a._high_risk_transaction_or_order(r["customer_text"])
        or a._followup_request(r["customer_text"])
    ),
    axis=1
)

x["candidate"] = x.apply(
    lambda r: candidate_rule(
        r["customer_text"],
        r["intent"]
    ),
    axis=1
)

new_cases = x[
    (x["existing_rules"] == False)
    & (x["candidate"] == True)
].copy()

correct = new_cases[
    new_cases["gold_should_escalate"].astype(str).str.lower() == "true"
]

incorrect = new_cases[
    new_cases["gold_should_escalate"].astype(str).str.lower() != "true"
]

print("Candidate additional escalations:", len(new_cases))
print("Correct:", len(correct))
print("Incorrect:", len(incorrect))

print("\nCORRECT:")
print(
    correct[
        ["case_id", "customer_text"]
    ].to_string(index=False)
)

print("\nINCORRECT:")
print(
    incorrect[
        ["case_id", "customer_text"]
    ].to_string(index=False)
)
