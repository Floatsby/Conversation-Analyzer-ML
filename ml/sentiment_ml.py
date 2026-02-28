from transformers import pipeline
import pandas as pd

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def add_sentiment_ml(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["message"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
    return df