from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.twcs import read_twcs, build_cases
from src.utils import ensure_parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv", required=True)
    p.add_argument("--brand", default=None, help="Friendly brand name for a sample dataset.")
    p.add_argument("--brand_id", default=None, help="Actual support account author_id.")
    p.add_argument("--brand_name", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.raw_csv, nrows=args.max_rows, low_memory=False)
    brand_id = args.brand_id or args.brand
    brand_name = args.brand_name or args.brand or brand_id
    if not brand_id:
        raise SystemExit("Provide --brand_id for the real TWCS dataset.")

    cases = build_cases(df, str(brand_id), str(brand_name))
    if cases.empty:
        raise SystemExit(
            "No customer→selected-brand cases found. "
            "Run src.discover_brands first and verify the author_id."
        )

    ensure_parent(args.out)
    cases.to_csv(args.out, index=False)
    print(f"Saved {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
