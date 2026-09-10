from pathlib import Path
import argparse
from collections import Counter

import pandas as pd


def discover_support_accounts(raw_csv: str, top_k: int = 30, chunksize: int = 100_000):
    """
    Discover likely brand/support accounts from TWCS without loading
    the entire 500+ MB CSV into memory.
    """

    usecols = [
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    author_outbound = Counter()
    author_total = Counter()

    print("Reading TWCS in chunks...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            raw_csv,
            usecols=usecols,
            chunksize=chunksize,
            dtype={
                "tweet_id": "string",
                "author_id": "string",
                "inbound": "string",
                "text": "string",
                "response_tweet_id": "string",
                "in_response_to_tweet_id": "string",
            },
            keep_default_na=False,
            low_memory=True,
        ),
        start=1,
    ):
        # Normalize inbound values.
        inbound = (
            df["inbound"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "t"])
        )

        authors = df["author_id"].astype("string").fillna("").str.strip()

        # Count all tweets by author.
        for author in authors[authors != ""]:
            author_total[author] += 1

        # Outbound tweets are likely brand/support replies.
        for author in authors[~inbound]:
            if author:
                author_outbound[author] += 1

        if chunk_number % 5 == 0:
            print(f"Processed approximately {chunk_number * chunksize:,} rows...")

    print("Finished reading dataset.")
    print("Ranking likely support accounts...")

    rows = []

    for author, outbound_count in author_outbound.most_common():
        total_count = author_total.get(author, 0)

        if total_count == 0:
            continue

        outbound_ratio = outbound_count / total_count

        rows.append(
            {
                "author_id": author,
                "outbound_tweets": outbound_count,
                "total_tweets": total_count,
                "outbound_ratio": round(outbound_ratio, 4),
            }
        )

        if len(rows) >= top_k:
            break

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_csv",
        required=True,
        help="Path to twcs.csv",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    output = discover_support_accounts(
        args.raw_csv,
        top_k=args.top_k,
        chunksize=args.chunksize,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)

    print()
    print(f"Saved {len(output)} candidates to: {args.out}")
    print()
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()