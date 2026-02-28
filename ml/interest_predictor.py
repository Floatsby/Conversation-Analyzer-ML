import pandas as pd

def predict_interest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict interest trend using rolling sentiment.
    """
    # Convert sentiment labels to numbers
    sentiment_map = {"POSITIVE":1, "NEGATIVE":-1, "NEUTRAL":0}
    df["sentiment_numeric"] = df["sentiment"].map(sentiment_map)
    
    # Rolling sentiment mean
    df["rolling_sentiment"] = df["sentiment_numeric"].rolling(3, min_periods=1).mean()
    
    # Interest trend
    df["interest_trend"] = df["rolling_sentiment"].apply(lambda x: "fading" if x < 0 else "increasing")
    return df