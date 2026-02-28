# Conversation Analyzer ML

**Conversation Analyzer ML** is a Streamlit-powered application that analyzes chat conversations using Machine Learning techniques. It predicts conversation dynamics, emotions, arguments, engagement trends, and even estimates the likely relationship outcome (Breakup, Proposal, Ghosting, Stagnant) based on heuristics.

---

## Features

- **Sentiment Analysis**: Detects POSITIVE, NEGATIVE, or NEUTRAL messages using Hugging Face DistilBERT.  
- **Emotion Detection**: Identifies emotions such as joy, sadness, anger, and more.  
- **Argument Detection**: Flags messages that may indicate disputes (dummy classifier for now).  
- **Interest Trend Analysis**: Determines if a conversation shows increasing or fading interest.  
- **Relationship Outcome Prediction**: Uses simple heuristics to estimate whether the conversation is heading towards a breakup, proposal, ghosting, or stagnancy.  
- **Interactive Dashboard**: Upload a chat file and visualize analysis live.

---

## What I have done and next Steps

- For Sentiment Analysis -Transformer - distilbert-base-uncased-finetuned-sst-2-english
- For Emotion detection -Transformer - j-hartmann/emotion-english-distilroberta-base
- For Argument / Dispute Detection - Transformer Classifier - Fine-tune roberta-base on labeled chat data
- For Interest Trend / Fading Interest - Sequence model - LSTM or BERT on conversation history
- For Relationship Outcome Prediction - Classifier Heuristic - Custom classifier or heuristics (TBD)

---

##Next Steps 

- Instead of a dummy heuristic use trained and labeled data
- Collect data for training

