#!/usr/bin/env python3
"""
Tonal Analysis Model - Load trained model and predict emotions from audio files
Always shows probabilities by default
"""
import torch
import torch.nn as nn
import librosa
import numpy as np
import pickle
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

class OptimizedSimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class OptimizedMediumNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class OptimizedLargeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class EmotionPredictor:
    def __init__(self, model_path="best_with_augmentation_model.pth"):
        """Initialize emotion predictor with trained model (defaults to best model)"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.models = None  # For ensemble
        self.label_encoder = None
        self.feature_scaler = None
        self.is_ensemble = False

        self.load_model()

    def load_model(self):
        """Load trained model and preprocessing components"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Extract model info
        self.label_encoder = checkpoint['label_encoder']
        self.feature_scaler = checkpoint['feature_scaler']

        # Check if ensemble or single model
        if 'models' in checkpoint:
            # Ensemble model
            self.is_ensemble = True
            self.models = []
            model_states = checkpoint['models']
            architecture = checkpoint['model_architecture']
            num_classes = checkpoint['num_classes']
            input_size = checkpoint['input_size']

            print(f"Loading ensemble of {len(model_states)} models...")

            for i, state_dict in enumerate(model_states):
                model = self.create_model(architecture, input_size, num_classes)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)

            print(f"✅ Loaded ensemble with accuracy: {checkpoint.get('ensemble_accuracy', 'N/A'):.1f}%")

        else:
            # Single model
            self.is_ensemble = False
            architecture = checkpoint['model_architecture']
            num_classes = checkpoint['num_classes']
            input_size = checkpoint['input_size']

            self.model = self.create_model(architecture, input_size, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Loaded single model with accuracy: {checkpoint.get('accuracy', 'N/A'):.1f}%")

        print(f"Emotions: {list(self.label_encoder.classes_)}")

    def create_model(self, architecture, input_size, num_classes):
        """Create model based on architecture type"""
        if architecture == "simple":
            return OptimizedSimpleNet(input_size, num_classes)
        elif architecture == "medium":
            return OptimizedMediumNet(input_size, num_classes)
        else:
            return OptimizedLargeNet(input_size, num_classes)

    def extract_features(self, file_path, target_sr=16000, max_length=6.0):
        """Extract the same optimized features used during training"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=target_sr, duration=max_length)

            if len(y) == 0 or np.max(np.abs(y)) == 0:
                print("⚠️  Warning: Empty or silent audio file")
                return np.zeros(65, dtype=np.float32)

            # Normalize audio
            y = y / np.max(np.abs(y))

            # Ensure consistent length
            target_length = int(max_length * target_sr)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            features = []

            # 1. Enhanced MFCC features (39 features total)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                # Original MFCCs (13 features)
                mfcc_means = np.mean(mfcc, axis=1)
                features.extend([float(x) for x in mfcc_means])

                # Delta MFCCs (13 features)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta_means = np.mean(mfcc_delta, axis=1)
                features.extend([float(x) for x in mfcc_delta_means])

                # Delta-delta MFCCs (13 features)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_delta2_means = np.mean(mfcc_delta2, axis=1)
                features.extend([float(x) for x in mfcc_delta2_means])

            except:
                features.extend([0.0] * 39)

            # 2. Spectral features (8 features)
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                zcr = librosa.feature.zero_crossing_rate(y)[0]

                features.extend([
                    float(np.mean(spectral_centroids) / 8000),
                    float(np.std(spectral_centroids) / 8000),
                    float(np.mean(spectral_rolloff) / 8000),
                    float(np.std(spectral_rolloff) / 8000),
                    float(np.mean(spectral_bandwidth) / 8000),
                    float(np.std(spectral_bandwidth) / 8000),
                    float(np.mean(zcr)),
                    float(np.std(zcr))
                ])
            except:
                features.extend([0.0] * 8)

            # 3. Energy features (4 features)
            try:
                rms = librosa.feature.rms(y=y)[0]
                features.extend([
                    float(np.mean(rms)),
                    float(np.std(rms)),
                    float(np.max(rms)),
                    float(np.min(rms))
                ])
            except:
                features.extend([0.0] * 4)

            # 4. Spectral contrast (7 features)
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_means = np.mean(spectral_contrast, axis=1)
                features.extend([float(x) for x in contrast_means])
            except:
                features.extend([0.0] * 7)

            # 5. Tonnetz (6 features)
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_means = np.mean(tonnetz, axis=1)
                features.extend([float(x) for x in tonnetz_means])
            except:
                features.extend([0.0] * 6)

            # 6. Tempo feature (1 feature)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo / 200))
            except:
                features.append(0.0)

            # Total: 65 features
            features = features[:65]
            while len(features) < 65:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return np.zeros(65, dtype=np.float32)

    def predict_emotion(self, audio_file):
        """Predict emotion from audio file - always returns probabilities"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        print(f"🎵 Analyzing audio: {os.path.basename(audio_file)}")

        # Extract features
        features = self.extract_features(audio_file)

        # Scale features
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        # Predict
        with torch.no_grad():
            if self.is_ensemble:
                # Ensemble prediction
                all_predictions = []
                for model in self.models:
                    outputs = model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.append(probs.cpu().numpy())

                # Average ensemble predictions
                ensemble_probs = np.mean(all_predictions, axis=0)
                predicted_class = np.argmax(ensemble_probs)
                confidence = float(ensemble_probs[0][predicted_class])
                all_probs = {self.label_encoder.classes_[i]: float(ensemble_probs[0][i])
                            for i in range(len(self.label_encoder.classes_))}

            else:
                # Single model prediction
                outputs = self.model(features_tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = float(probs[0][predicted_class])
                all_probs = {self.label_encoder.classes_[i]: float(probs[0][i])
                            for i in range(len(self.label_encoder.classes_))}

        # Get emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            'emotion': emotion,
            'confidence': confidence,
            'audio_file': audio_file,
            'all_probabilities': all_probs
        }

def main():
    """Command line interface for emotion prediction with probabilities always shown"""
    parser = argparse.ArgumentParser(description='Predict emotion from audio file using best model (68.9% accuracy)')
    parser.add_argument('audio_file', help='Path to audio file (.wav, .mp3, etc.)')
    parser.add_argument('--model', '-m', default='best_with_augmentation_model.pth',
                       help='Path to trained model file (default: best_with_augmentation_model.pth)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only output the predicted emotion (no probabilities)')

    args = parser.parse_args()

    try:
        # Load predictor with best model by default
        predictor = EmotionPredictor(args.model)

        # Make prediction - always get probabilities
        result = predictor.predict_emotion(args.audio_file)

        if args.quiet:
            print(result['emotion'])
        else:
            print(f"\n🎯 PREDICTION RESULT (Using Best Model - 68.9% Accuracy):")
            print(f"📁 File: {os.path.basename(result['audio_file'])}")
            print(f"😊 Emotion: {result['emotion'].upper()}")
            print(f"🔥 Confidence: {result['confidence']:.1%}")

            # Always show probabilities (unless quiet mode)
            print(f"\n📊 ALL PROBABILITIES:")
            sorted_probs = sorted(result['all_probabilities'].items(),
                                key=lambda x: x[1], reverse=True)
            for emotion, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                print(f"  {emotion:10s}: {prob:.1%} {bar}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
