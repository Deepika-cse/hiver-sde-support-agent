import pandas as pd

from src.agent.agent import SupportAgent


INPUT_PATH = "results/groq_predictions_180_llm.csv"
OUTPUT_PATH = "results/groq_predictions_180_final.csv"


def main():

    predictions = pd.read_csv(
        INPUT_PATH,
        dtype=str,
    ).fillna("")

    agent = SupportAgent(None)

    rows = []

    for _, row in predictions.iterrows():

        text = row["customer_text"]
        intent = row["intent"]

        account_sensitive = agent._account_sensitive(
            text,
            intent,
        )

        persistent_issue = agent._persistent_issue(
            text,
        )

        insufficient_information = (
            agent._insufficient_information(
                text,
            )
        )

        payment_or_transaction_issue = (
            agent._payment_or_transaction_issue(
                text,
            )
        )

        high_risk_transaction_or_order = (
            agent._high_risk_transaction_or_order(
                text,
            )
        )

        followup_request = (
            agent._followup_request(
                text,
            )
        )

        additional_review_signal = (
            agent._additional_review_signal(
                text,
            )
        )

        rule_escalate = (
            account_sensitive
            or persistent_issue
            or insufficient_information
            or payment_or_transaction_issue
            or high_risk_transaction_or_order
            or followup_request
            or additional_review_signal
        )

        new_row = row.to_dict()

        if rule_escalate:

            new_row["should_escalate"] = "True"
            new_row["decision"] = "escalate"

            if account_sensitive:
                reason = (
                    "Account or security-sensitive "
                    "request requires human support."
                )

            elif persistent_issue:
                reason = (
                    "Repeated or unresolved customer "
                    "issue requires human support."
                )

            elif insufficient_information:
                reason = (
                    "Insufficient customer information "
                    "to safely resolve the request."
                )

            elif payment_or_transaction_issue:
                reason = (
                    "Payment or refund discrepancy "
                    "requires human review."
                )

            elif high_risk_transaction_or_order:
                reason = (
                    "High-risk order or account transaction "
                    "requires human review."
                )

            elif followup_request:
                reason = (
                    "Customer has already provided information "
                    "and requires follow-up support."
                )

            else:
                reason = (
                    "Customer request involves a supportability, "
                    "refund-processing, or delivery-receipt issue "
                    "requiring human review."
                )

            new_row["escalation_reason"] = reason

        rows.append(new_row)

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Rows:",
        len(output),
    )

    print(
        "Unique case IDs:",
        output["case_id"].nunique(),
    )

    print(
        "Escalations:",
        (
            output["should_escalate"]
            == "True"
        ).sum(),
    )

    print(
        "Auto-handle:",
        (
            output["should_escalate"]
            == "False"
        ).sum(),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()