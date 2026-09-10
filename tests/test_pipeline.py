from pathlib import Path
import pandas as pd
from src.retrieval.tfidf import HistoricalRetriever
from src.agent.agent import SupportAgent


def test_dry_agent():
    p = Path("data/raw/sample_raw.csv")
    df = pd.read_csv(p).fillna("")
    from src.data.twcs import build_cases
    cases = build_cases(df, "AmazonHelp", "AmazonHelp")
    assert len(cases) > 0
    r = HistoricalRetriever(cases)
    a = SupportAgent(r, llm=None)
    out = a.predict(cases.iloc[0].to_dict())
    assert "decision" in out


def test_sample_case_ids_are_normalized():
    p = Path("data/raw/sample_raw.csv")
    df = pd.read_csv(p).fillna("")
    from src.data.twcs import build_cases
    cases = build_cases(df, "AmazonHelp", "AmazonHelp")
    assert cases.iloc[0]["latest_customer_tweet_id"] == "1"
    assert cases.iloc[0]["brand_reply_tweet_id"] == "2"
