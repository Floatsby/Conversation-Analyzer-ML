import streamlit as st
import pandas as pd

from ml.utils import parse_chat
from ml.sentiment_ml import add_sentiment_ml
from ml.emotion_ml import add_emotion
from ml.argument_detector import detect_arguments
from ml.interest_predictor import predict_interest
from ml.relationship_predictor import predict_relationship

st.title("Conversation Analyzer: ML Version with Relationship Prediction")

# Upload chat file
uploaded_file = st.file_uploader("Upload your chat file (txt)", type=["txt"])
if uploaded_file:
    df = parse_chat(uploaded_file)
    
    # ML analysis
    df = add_sentiment_ml(df)
    df = add_emotion(df)
    df = detect_arguments(df)
    df = predict_interest(df)
    df = predict_relationship(df)
    
    st.subheader("Chat Analysis")
    st.dataframe(df)
    
    st.subheader("Metrics")
    st.write("Sentiment distribution:", df["sentiment"].value_counts(normalize=True))
    st.write("Emotion distribution:", df["emotion"].value_counts(normalize=True))
    st.write("Argument count:", df["argument"].sum())
    st.write("Interest trend:", df["interest_trend"].value_counts())
    st.write("Predicted Relationship Outcome:", df["predicted_relationship"].iloc[0])