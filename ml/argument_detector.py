import pandas as pd
import random

# Dummy argument detection (replace with fine-tuned model later)
def detect_arguments(df: pd.DataFrame) -> pd.DataFrame:
    df["argument"] = df["message"].apply(lambda x: random.choice([True, False]))
    return df