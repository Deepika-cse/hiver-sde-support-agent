from pathlib import Path
import pandas as pd

GOLD_PATH = Path("data/golden/golden_set.csv")

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


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def save(df):
    df.to_csv(GOLD_PATH, index=False)


def main():
    if not GOLD_PATH.exists():
        print(f"ERROR: Could not find {GOLD_PATH}")
        return

    df = pd.read_csv(GOLD_PATH, dtype=str).fillna("")

    required_columns = [
        "gold_intent",
        "gold_secondary_intent",
        "gold_should_escalate",
        "gold_escalation_reason",
        "gold_reply_notes",
        "labeler",
        "label_version",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    # Make every labeling column explicitly string.
    for column in required_columns:
        df[column] = df[column].astype(str)

    unlabeled = [
        i for i in df.index
        if clean(df.at[i, "gold_intent"]) == ""
    ]

    if not unlabeled:
        print("\nAll Golden Set examples are already labeled!")
        return

    print("\n==========================================")
    print(" AMAZONHELP GOLDEN SET LABELING TOOL")
    print("==========================================")
    print(f"Total rows: {len(df)}")
    print(f"Already labeled: {len(df) - len(unlabeled)}")
    print(f"Remaining: {len(unlabeled)}")
    print()
    print("Enter q at the intent prompt to save and quit.")
    print("Progress is saved after EVERY completed case.")
    print()

    for row_index in unlabeled:

        row = df.loc[row_index]

        completed = len(df) - len(unlabeled) + (
            list(unlabeled).index(row_index)
        )

        print("\n" + "=" * 80)
        print(f"CASE {completed + 1} / {len(df)}")
        print(f"Case ID: {clean(row.get('case_id'))}")
        print("=" * 80)

        print("\nLATEST CUSTOMER MESSAGE:\n")
        print(clean(row.get("latest_customer_text")))

        context = clean(row.get("conversation_context"))

        if context:
            print("\nCONVERSATION CONTEXT:\n")
            print(context)

        historical_reply = clean(row.get("historical_brand_reply"))

        if historical_reply:
            print("\nHISTORICAL AMAZONHELP REPLY:\n")
            print(historical_reply)

        print("\nINTENTS:")

        for number, intent in enumerate(INTENTS, start=1):
            print(f"{number}. {intent}")

        while True:

            choice = input(
                "\nChoose primary intent [1-10] or q to quit: "
            ).strip().lower()

            if choice == "q":
                save(df)
                print("\nProgress saved.")
                print("You can run this program again to continue.")
                return

            if choice.isdigit():

                number = int(choice)

                if 1 <= number <= len(INTENTS):
                    primary_intent = INTENTS[number - 1]
                    break

            print("Invalid choice. Enter a number from 1 to 10.")

        print(f"\nSelected: {primary_intent}")

        while True:

            secondary = input(
                "Secondary intent [1-10, Enter for none]: "
            ).strip()

            if secondary == "":
                secondary_intent = ""
                break

            if secondary.isdigit():

                number = int(secondary)

                if 1 <= number <= len(INTENTS):

                    candidate = INTENTS[number - 1]

                    if candidate == primary_intent:
                        print(
                            "Secondary intent must be different "
                            "from primary intent."
                        )
                        continue

                    secondary_intent = candidate
                    break

            print("Invalid choice. Enter 1-10 or press Enter for none.")

        while True:

            escalation = input(
                "Should this case escalate to a human? [y/n]: "
            ).strip().lower()

            if escalation in {"y", "yes"}:
                should_escalate = True
                break

            if escalation in {"n", "no"}:
                should_escalate = False
                break

            print("Please enter y or n.")

        escalation_reason = ""

        if should_escalate:

            while not escalation_reason:

                escalation_reason = input(
                    "Escalation reason: "
                ).strip()

                if not escalation_reason:
                    print("Please provide a short, specific reason.")

        reply_note = ""

        while not reply_note:

            reply_note = input(
                "What should a good grounded reply contain/avoid? "
            ).strip()

            if not reply_note:
                print("Please enter a short reply note.")

        df.at[row_index, "gold_intent"] = primary_intent
        df.at[row_index, "gold_secondary_intent"] = secondary_intent
        df.at[row_index, "gold_should_escalate"] = (
            "true" if should_escalate else "false"
        )
        df.at[row_index, "gold_escalation_reason"] = escalation_reason
        df.at[row_index, "gold_reply_notes"] = reply_note
        df.at[row_index, "labeler"] = "human"
        df.at[row_index, "label_version"] = "v1"

        save(df)

        print("\n✓ Case saved successfully.")

    print("\n==========================================")
    print(" ALL 180 GOLDEN SET CASES ARE LABELED!")
    print("==========================================")
    print(f"Saved to: {GOLD_PATH}")


if __name__ == "__main__":
    main()