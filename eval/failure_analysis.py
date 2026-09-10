from __future__ import annotations

import argparse
import pandas as pd
from src.utils import ensure_parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pdt = pd.read_csv(args.predictions).fillna("")
    gold = pd.read_csv(args.gold).fillna("")
    m = gold.merge(pdt, on="case_id", suffixes=("_gold", "_pred"))

    rows = []
    for _, r in m.iterrows():
        intent_error = r["gold_intent"] != r["intent"]
        esc_gold = str(r["gold_should_escalate"]).lower() in {"true", "1", "yes"}
        esc_pred = bool(r["should_escalate"])
        if intent_error or esc_gold != esc_pred:
            if intent_error:
                category = f"intent:{r['gold_intent']}→{r['intent']}"
            else:
                category = "escalation_mismatch"
            rows.append({
                "case_id": r["case_id"],
                "category": category,
                "customer_text": r["customer_text"],
                "gold_intent": r["gold_intent"],
                "pred_intent": r["intent"],
                "gold_escalate": esc_gold,
                "pred_escalate": esc_pred,
                "trust_score": r.get("trust_score", ""),
                "retrieval_top_score": r.get("retrieval_top_score", ""),
                "escalation_reason": r.get("escalation_reason", ""),
                "failure_hypothesis": "",
            })

    out = pd.DataFrame(rows)
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)
    print(out.head(20).to_string(index=False))
    print(f"\nTotal failures/mismatches: {len(out)}")


if __name__ == "__main__":
    main()
