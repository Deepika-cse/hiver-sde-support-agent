import pandas as pd
from src.agent.agent import SupportAgent

p = pd.read_csv(
    "results/groq_predictions_141_base.csv",
    dtype=str
).fillna("")

a = SupportAgent(None)

rows = []

for _, row in p.iterrows():
    text = row["customer_text"]
    intent = row["intent"]

    rule_escalate = (
        a._account_sensitive(text, intent)
        or a._persistent_issue(text)
        or a._insufficient_information(text)
        or a._payment_or_transaction_issue(text)
        or a._high_risk_transaction_or_order(text)
        or a._followup_request(text)
    )

    new_row = row.to_dict()

    if rule_escalate:
        new_row["should_escalate"] = "True"
        new_row["decision"] = "escalate"

    rows.append(new_row)

out = pd.DataFrame(rows)

out.to_csv(
    "results/groq_predictions_141_rerouted.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Rows:", len(out))
print(
    "Escalations:",
    (out["should_escalate"] == "True").sum()
)
print(
    "Auto-handle:",
    (out["should_escalate"] == "False").sum()
)
