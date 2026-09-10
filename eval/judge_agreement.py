from __future__ import annotations

import argparse, json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from src.utils import ensure_parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    j = pd.read_csv(args.judge).fillna("")
    h = pd.read_csv(args.human).fillna("")
    m = j.merge(h, on="case_id", suffixes=("_judge", "_human"))

    result = {"n": len(m), "kappa": {}}
    for dim in ["groundedness", "helpfulness", "correctness", "tone", "overall"]:
        jc, hc = dim, f"{dim}_human"
        if jc in m and hc in m:
            result["kappa"][dim] = float(
                cohen_kappa_score(m[jc].astype(int), m[hc].astype(int), weights="quadratic")
            )

    ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
