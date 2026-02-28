from transformers import pipeline
import pandas as pd

# Emotion detection pipeline
emotion_pipeline = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    return_all_scores=False
)

def add_emotion(df: pd.DataFrame) -> pd.DataFrame:
    df["emotion"] = df["message"].apply(lambda x: emotion_pipeline(x)[0]["label"])
    return df