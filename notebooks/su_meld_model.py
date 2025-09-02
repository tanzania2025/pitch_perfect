#!/usr/bin/env python3
"""
Dedicated MELD Model Evaluation Script
Test the robust_meld_model.pth performance on MELD test set and cross-validation datasets
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
import tempfile
from tqdm import tqdm
from google.cloud import storage
import warnings

warnings.filterwarnings("ignore")


class SimpleEmotionNet(nn.Module):
    """Identical architecture to training script"""

    def __init__(self, input_size=50, num_classes=7):
        super(SimpleEmotionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class MELDModelEvaluator:
    """Dedicated evaluator for the trained MELD model"""

    def __init__(self, model_path="robust_meld_model.pth", use_gcs=True):
        """
        Initialize evaluator with the specific MELD model

        Args:
            model_path: Path to robust_meld_model.pth
            use_gcs: Whether to use Google Cloud Storage for data
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotions = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotions)
        self.use_gcs = use_gcs

        # GCS setup if needed
        if use_gcs:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(
                    "pp-pitchperfect-lewagon-raw-data"
                )
                print("Connected to GCS bucket: pp-pitchperfect-lewagon-raw-data")
            except:
                print("Warning: Could not connect to GCS. Will use local files only.")
                self.use_gcs = False

        self.load_model(model_path)
        print(f"MELD Model Evaluator initialized on {self.device}")

    def load_model(self, model_path):
        """Load the trained MELD model"""
        try:
            # Try loading local file first
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
            elif self.use_gcs:
                # Try GCS
                print("Loading model from Google Cloud Storage...")
                model_blob = self.bucket.blob("models/robust_meld_model.pth")
                with tempfile.NamedTemporaryFile() as temp_file:
                    model_blob.download_to_filename(temp_file.name)
                    checkpoint = torch.load(
                        temp_file.name, map_location=self.device, weights_only=False
                    )
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model
            self.model = SimpleEmotionNet(input_size=50, num_classes=7)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Print model info
            training_accuracy = checkpoint.get("accuracy", "Unknown")
            print(f"✓ Model loaded successfully!")
            print(f"  Training accuracy: {training_accuracy}")
            print(
                f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_features(self, audio_path, target_sr=16000, max_length=5.0):
        """Extract features using EXACT same method as training"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=target_sr, duration=max_length)
            if len(y) == 0:
                return np.zeros(50, dtype=np.float32)

            # Ensure consistent length (SAME AS TRAINING)
            target_length = int(max_length * target_sr)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            features = []

            # 1. Basic MFCC (17 features) - IDENTICAL to training
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features.extend(
                    [
                        float(np.mean(mfcc)),
                        float(np.std(mfcc)),
                        float(np.max(mfcc)),
                        float(np.min(mfcc)),
                    ]
                )
                mfcc_means = np.mean(mfcc, axis=1)
                features.extend([float(x) for x in mfcc_means[:13]])
            except:
                features.extend([0.0] * 17)

            # 2. Spectral features (6 features) - IDENTICAL to training
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features.extend(
                    [
                        float(np.mean(spectral_centroids)),
                        float(np.std(spectral_centroids)),
                    ]
                )
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features.extend(
                    [float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))]
                )
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features.extend([float(np.mean(zcr)), float(np.std(zcr))])
            except:
                features.extend([0.0] * 6)

            # 3. Energy features (4 features) - IDENTICAL to training
            try:
                rms = librosa.feature.rms(y=y)[0]
                features.extend(
                    [
                        float(np.mean(rms)),
                        float(np.std(rms)),
                        float(np.max(rms)),
                        float(np.min(rms)),
                    ]
                )
            except:
                features.extend([0.0] * 4)

            # 4. Chroma features (12 features) - IDENTICAL to training
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend([float(x) for x in chroma_mean[:12]])
            except:
                features.extend([0.0] * 12)

            # 5. Pitch and tempo (4 features) - IDENTICAL to training
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                pitch_values = []
                for t in range(min(10, pitches.shape[1])):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    features.extend(
                        [
                            float(np.mean(pitch_values)),
                            float(np.std(pitch_values)),
                            float(len(pitch_values)),
                        ]
                    )
                else:
                    features.extend([0.0, 0.0, 0.0])

                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))
            except:
                features.extend([0.0] * 4)

            # Additional features to reach 50 (7 features)
            try:
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                features.extend([float(np.mean(spec_bw)), float(np.std(spec_bw))])
                spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features.extend(
                    [float(np.mean(spec_contrast)), float(np.std(spec_contrast))]
                )
                features.extend(
                    [float(np.sum(y**2)), float(np.mean(np.abs(y))), float(len(y) / sr)]
                )
            except:
                features.extend([0.0] * 7)

            # Ensure exactly 50 features
            features = features[:50]
            while len(features) < 50:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(50, dtype=np.float32)

    def download_audio_from_gcs(self, gcs_path):
        """Download audio from GCS (same as training)"""
        try:
            blob = self.bucket.blob(gcs_path)
            if not blob.exists():
                return None
            return blob.download_as_bytes()
        except:
            return None

    def evaluate_meld_test_set(self, test_csv_path=None, test_audio_dir=None):
        """
        Evaluate on MELD test set - the DEFINITIVE performance test

        Args:
            test_csv_path: Path to test_sent_emo.csv (or None for GCS)
            test_audio_dir: Path to test audio directory (or None for GCS)
        """
        test_csv_path = "data/external/test_sent_emo.csv"
        print("\n" + "=" * 60)
        print("EVALUATING ON MELD TEST SET")
        print("=" * 60)

        # Load test data
        if test_csv_path and os.path.exists(test_csv_path):
            print(f"Loading test CSV from: {test_csv_path}")
            test_df = pd.read_csv(test_csv_path)
        elif self.use_gcs:
            print("Loading test CSV from GCS...")
            blob = self.bucket.blob("data/external/MELD.Raw/test_sent_emo.csv")
            csv_bytes = blob.download_as_bytes()
            from io import StringIO

            test_df = pd.read_csv(StringIO(csv_bytes.decode("utf-8")))
        else:
            raise FileNotFoundError("Test CSV not found locally or in GCS")

        print(f"Test set size: {len(test_df)} samples")
        print(f"Emotion distribution:")
        for emotion in self.emotions:
            count = (test_df["Emotion"] == emotion).sum()
            print(f"  {emotion}: {count} ({count/len(test_df)*100:.1f}%)")

        # Evaluate samples
        predictions = []
        true_labels = []
        confidences = []
        successful_predictions = 0

        print("\nProcessing test samples...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"

            # Get audio features
            features = None
            if test_audio_dir and os.path.exists(
                os.path.join(test_audio_dir, audio_filename)
            ):
                # Local file
                audio_path = os.path.join(test_audio_dir, audio_filename)
                features = self.extract_features(audio_path)
            elif self.use_gcs:
                # GCS file
                gcs_path = f"datasets/meld/test_splits_complete/{audio_filename}"
                audio_bytes = self.download_audio_from_gcs(gcs_path)
                if audio_bytes:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_file:
                        tmp_file.write(audio_bytes)
                        features = self.extract_features(tmp_file.name)
                        os.unlink(tmp_file.name)

            # Make prediction
            if features is not None and not np.all(features == 0):
                with torch.no_grad():
                    features_tensor = (
                        torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    )
                    output = self.model(features_tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

                    predicted_idx = np.argmax(probabilities)
                    predicted_emotion = self.emotions[predicted_idx]
                    confidence = float(probabilities[predicted_idx])

                    predictions.append(predicted_emotion)
                    confidences.append(confidence)
                    successful_predictions += 1
            else:
                # Fallback for missing/failed audio
                predictions.append("neutral")  # Most common emotion
                confidences.append(0.0)

            true_labels.append(row["Emotion"])

        print(
            f"Successfully processed: {successful_predictions}/{len(test_df)} samples"
        )

        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(
            true_labels, predictions, confidences
        )
        self.print_detailed_results(metrics, "MELD Test Set")

        # Plot results
        self.plot_confusion_matrix(true_labels, predictions, "MELD Test Set")
        self.plot_emotion_performance(metrics["per_emotion_metrics"])

        return metrics

    def calculate_comprehensive_metrics(self, true_labels, predictions, confidences):
        """Calculate all performance metrics"""

        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=self.emotions,
            average=None,
            zero_division=0,
        )

        # Weighted metrics (accounts for class imbalance)
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted", zero_division=0
        )

        # Macro metrics (treats all classes equally)
        m_precision, m_recall, m_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro", zero_division=0
        )

        # Additional metrics
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(true_labels, predictions)

        # Per-emotion analysis
        emotion_metrics = {}
        for i, emotion in enumerate(self.emotions):
            emotion_metrics[emotion] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": support[i],
            }

        return {
            "accuracy": accuracy,
            "weighted_precision": w_precision,
            "weighted_recall": w_recall,
            "weighted_f1": w_f1,
            "macro_precision": m_precision,
            "macro_recall": m_recall,
            "macro_f1": m_f1,
            "cohen_kappa": kappa,
            "average_confidence": np.mean(confidences),
            "per_emotion_metrics": emotion_metrics,
            "confusion_matrix": confusion_matrix(
                true_labels, predictions, labels=self.emotions
            ),
        }

    def print_detailed_results(self, metrics, dataset_name):
        """Print comprehensive performance report"""

        print(f"\n{'='*60}")
        print(f"PERFORMANCE RESULTS: {dataset_name}")
        print(f"{'='*60}")

        print(f"\nOVERALL PERFORMANCE:")
        print(
            f"  🎯 Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
        )
        print(f"  📊 Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        print(f"  📊 Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"  📏 Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"  🔮 Average Confidence: {metrics['average_confidence']:.4f}")

        print(f"\n📋 DETAILED PER-EMOTION PERFORMANCE:")
        print(
            f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}"
        )
        print("-" * 52)

        best_f1 = 0
        worst_f1 = 1
        best_emotion = ""
        worst_emotion = ""

        for emotion, scores in metrics["per_emotion_metrics"].items():
            precision = scores["precision"]
            recall = scores["recall"]
            f1 = scores["f1_score"]
            support = scores["support"]

            print(
                f"{emotion.capitalize():<12} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<8}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_emotion = emotion
            if f1 < worst_f1:
                worst_f1 = f1
                worst_emotion = emotion

        print(f"\n🏆 Best performing: {best_emotion} (F1: {best_f1:.3f})")
        print(f"⚠️  Worst performing: {worst_emotion} (F1: {worst_f1:.3f})")

        # Performance assessment
        if metrics["accuracy"] >= 0.70:
            assessment = "🌟 EXCELLENT - Model performs very well!"
        elif metrics["accuracy"] >= 0.60:
            assessment = "✅ GOOD - Model performs well on most emotions"
        elif metrics["accuracy"] >= 0.50:
            assessment = "⚡ FAIR - Model has room for improvement"
        else:
            assessment = "🔧 NEEDS WORK - Consider retraining or feature engineering"

        print(f"\n{assessment}")

        return metrics

    def plot_confusion_matrix(self, true_labels, predictions, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions, labels=self.emotions)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[e.capitalize() for e in self.emotions],
            yticklabels=[e.capitalize() for e in self.emotions],
        )
        plt.title(f"{title} - Confusion Matrix (Normalized)", fontsize=16)
        plt.xlabel("Predicted Emotion", fontsize=14)
        plt.ylabel("True Emotion", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Also plot raw counts
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=[e.capitalize() for e in self.emotions],
            yticklabels=[e.capitalize() for e in self.emotions],
        )
        plt.title(f"{title} - Confusion Matrix (Raw Counts)", fontsize=16)
        plt.xlabel("Predicted Emotion", fontsize=14)
        plt.ylabel("True Emotion", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_emotion_performance(self, emotion_metrics):
        """Plot per-emotion performance metrics"""
        emotions = list(emotion_metrics.keys())
        precisions = [emotion_metrics[e]["precision"] for e in emotions]
        recalls = [emotion_metrics[e]["recall"] for e in emotions]
        f1_scores = [emotion_metrics[e]["f1_score"] for e in emotions]

        x = np.arange(len(emotions))
        width = 0.25

        plt.figure(figsize=(14, 8))
        plt.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        plt.bar(x, recalls, width, label="Recall", alpha=0.8)
        plt.bar(x + width, f1_scores, width, label="F1-Score", alpha=0.8)

        plt.xlabel("Emotions", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Per-Emotion Performance Metrics", fontsize=14)
        plt.xticks(x, [e.capitalize() for e in emotions], rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """Main evaluation function"""
    print("🎭 MELD MODEL PERFORMANCE EVALUATION")
    print("=" * 60)

    # Initialize evaluator
    evaluator = MELDModelEvaluator("models/robust_meld_model.pth", use_gcs=False)

    if evaluator.model is None:
        print("❌ Failed to load model. Please ensure 'robust_meld_model.pth' exists.")
        return

    print("\n🚀 Starting comprehensive evaluation...")

    # Main evaluation on MELD test set
    try:
        metrics = evaluator.evaluate_meld_test_set()

        print(f"\n🏁 FINAL EVALUATION SUMMARY")
        print(f"Model Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
        print(f"Macro F1: {metrics['macro_f1']:.3f}")

        # Save results
        results_summary = f"""
MELD Model Evaluation Results
============================
Overall Accuracy: {metrics['accuracy']*100:.2f}%
Weighted F1-Score: {metrics['weighted_f1']:.3f}
Macro F1-Score: {metrics['macro_f1']:.3f}
Cohen's Kappa: {metrics['cohen_kappa']:.3f}

Per-Emotion F1 Scores:
{chr(10).join([f"  {emotion}: {scores['f1_score']:.3f}" for emotion, scores in metrics['per_emotion_metrics'].items()])}
        """

        with open("meld_evaluation_results.txt", "w") as f:
            f.write(results_summary)

        print("\n📄 Results saved to: meld_evaluation_results.txt")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure 'robust_meld_model.pth' exists in current directory")
        print("2. Check Google Cloud Storage credentials if using GCS")
        print("3. Verify MELD test data is accessible")


if __name__ == "__main__":
    main()
