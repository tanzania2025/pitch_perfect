#!/usr/bin/env python3
"""
DistilRoBERTa-only comparison to avoid TensorFlow mutex issues
Usage: python distilroberta_compare.py --data_dir data/external/
"""

import os
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# MELD emotion mapping
MELD_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MELD_LABEL2ID = {c: i for i, c in enumerate(MELD_CLASSES)}

def load_meld_test_data(data_dir):
    """Load MELD test data."""
    test_csv = os.path.join(data_dir, "test_sent_emo.csv")

    try:
        test_df = pd.read_csv(test_csv)[["Utterance", "Emotion"]].dropna()
        test_df["label"] = test_df["Emotion"].map(MELD_LABEL2ID)
        test_df = test_df.dropna(subset=["label"])
        test_df.rename(columns={"Utterance": "text"}, inplace=True)
        test_df["label"] = test_df["label"].astype(int)

        print(f"Loaded MELD test data: {len(test_df)} samples")
        return test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def evaluate_sentiment_distilroberta(test_df):
    """Evaluate standard DistilRoBERTa sentiment model."""
    print("Evaluating DistilRoBERTa (Sentiment)...")

    classifier = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english")

    # Simple mapping: sentiment -> MELD emotion
    sentiment_to_meld = {"NEGATIVE": "sadness", "POSITIVE": "joy"}

    texts = test_df["text"].tolist()
    y_true = test_df["label"].values
    y_pred = []

    for text in texts:
        result = classifier(text)[0]
        sentiment = result['label']

        if sentiment in sentiment_to_meld:
            emotion = sentiment_to_meld[sentiment]
            label_id = MELD_LABEL2ID[emotion]
        else:
            label_id = MELD_LABEL2ID["neutral"]

        y_pred.append(label_id)

    return y_true, np.array(y_pred)

def evaluate_emotion_distilroberta(test_df):
    """Evaluate emotion-specific DistilRoBERTa."""
    print("Evaluating DistilRoBERTa (Emotion)...")

    classifier = pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base")

    texts = test_df["text"].tolist()
    y_true = test_df["label"].values
    y_pred = []

    emotion_mapping = {
        "anger": "anger", "disgust": "disgust", "fear": "fear",
        "joy": "joy", "neutral": "neutral", "sadness": "sadness",
        "surprise": "surprise"
    }

    for text in texts:
        result = classifier(text)[0]
        emotion = result['label'].lower()

        if emotion in emotion_mapping:
            meld_emotion = emotion_mapping[emotion]
            label_id = MELD_LABEL2ID[meld_emotion]
        else:
            label_id = MELD_LABEL2ID["neutral"]

        y_pred.append(label_id)

    return y_true, np.array(y_pred)

def print_results(y_true, y_pred, model_name):
    """Print evaluation results."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"{model_name} Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nPer-class Report:")
    print(classification_report(y_true, y_pred, target_names=MELD_CLASSES, digits=4))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare DistilRoBERTa models on MELD")
    parser.add_argument("--data_dir", required=True, help="Directory with MELD data")
    args = parser.parse_args()

    # Load test data
    test_df = load_meld_test_data(args.data_dir)
    if test_df is None:
        return

    # Test both DistilRoBERTa approaches
    y_true_sent, y_pred_sent = evaluate_sentiment_distilroberta(test_df)
    print_results(y_true_sent, y_pred_sent, "DistilRoBERTa (Sentiment)")

    y_true_emo, y_pred_emo = evaluate_emotion_distilroberta(test_df)
    print_results(y_true_emo, y_pred_emo, "DistilRoBERTa (Emotion)")

    # Simple comparison
    acc_sent = accuracy_score(y_true_sent, y_pred_sent)
    acc_emo = accuracy_score(y_true_emo, y_pred_emo)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Sentiment DistilRoBERTa: {acc_sent:.4f}")
    print(f"Emotion DistilRoBERTa:   {acc_emo:.4f}")
    print(f"Best Model: {'Emotion' if acc_emo > acc_sent else 'Sentiment'} DistilRoBERTa")

if __name__ == "__main__":
    main()
