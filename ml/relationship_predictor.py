import pandas as pd

def predict_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts relationship outcome based on heuristics:
    - Mostly negative sentiment + many arguments → breakup
    - Mostly positive sentiment + increasing interest → proposal
    - Long response times or fading interest → ghosting
    - Otherwise → stagnant
    """

    # Map sentiment to numbers for calculations
    sentiment_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
    df["sentiment_numeric"] = df["sentiment"].map(sentiment_map)

    avg_sentiment = df["sentiment_numeric"].mean()
    argument_count = df["argument"].sum()
    interest_trend = df["interest_trend"].value_counts().get("fading", 0)
    total_messages = len(df)

    # Simple heuristic rules
    if avg_sentiment < -0.2 and argument_count / total_messages > 0.2:
        outcome = "Breakup"
    elif avg_sentiment > 0.5 and df["interest_trend"].value_counts().get("increasing",0) > 0:
        outcome = "Proposal / Deepening"
    elif interest_trend / total_messages > 0.3:
        outcome = "Ghosting / Fading"
    else:
        outcome = "Stagnant"

    # Add outcome column to all messages for display
    df["predicted_relationship"] = outcome
    return df