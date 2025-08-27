# 🎭 MELD Emotion Classification (BiLSTM)

This project implements an **end-to-end text emotion classification pipeline** using the [MELD dataset](https://affective-meld.github.io/).  
It classifies dialogue utterances into **7 emotions**: anger, disgust, fear, joy, neutral, sadness, surprise

The model is based on a **Bidirectional LSTM (BiLSTM)** with a dense "funnel" architecture.

---

## 📂 Project Structure
text_sentiment_analysis/
│
├── init.py # Exports constants (CLASSES, label2id, id2label) and key functions
├── preprocessing.py # Data loading, cleaning, tokenization, padding
├── models.py # Model architecture & training utilities
├── analyzer.py # Evaluation, classification reports, confusion matrix
├── text_sentiment_full.py # End-to-end training script (CLI entrypoint)
└── README.md # Project documentation

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas

Install dependencies:

```bash
pip install tensorflow pandas numpy
