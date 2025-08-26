#!/usr/bin/env python3
"""
Robust MELD Trainer with simplified, error-resistant feature extraction
"""

import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RobustMELDDataset(Dataset):
    """Robust dataset with simplified feature extraction"""

    def __init__(self, csv_data, audio_base_path, target_sr=16000, max_length=5.0):
        self.data = csv_data
        self.audio_base_path = audio_base_path
        self.target_sr = target_sr
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Construct filename
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        audio_path = os.path.join(self.audio_base_path, audio_filename)

        # Extract features
        features = self._extract_robust_features(audio_path)
        emotion = row['Emotion']

        return torch.FloatTensor(features), emotion

    def _extract_robust_features(self, audio_path):
        """Simplified, robust feature extraction"""
        try:
            if not os.path.exists(audio_path):
                return np.zeros(50, dtype=np.float32)

            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, duration=self.max_length)

            if len(y) == 0:
                return np.zeros(50, dtype=np.float32)

            # Ensure consistent length
            target_length = int(self.max_length * self.target_sr)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            features = []

            # 1. Basic MFCC (most important for emotion)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features.extend([
                    float(np.mean(mfcc)),
                    float(np.std(mfcc)),
                    float(np.max(mfcc)),
                    float(np.min(mfcc))
                ])

                # Individual MFCC coefficients means
                mfcc_means = np.mean(mfcc, axis=1)
                features.extend([float(x) for x in mfcc_means[:13]])

            except:
                features.extend([0.0] * 17)  # 4 + 13

            # 2. Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features.extend([
                    float(np.mean(spectral_centroids)),
                    float(np.std(spectral_centroids))
                ])

                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features.extend([
                    float(np.mean(spectral_rolloff)),
                    float(np.std(spectral_rolloff))
                ])

                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features.extend([
                    float(np.mean(zcr)),
                    float(np.std(zcr))
                ])
            except:
                features.extend([0.0] * 6)

            # 3. Energy features
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

            # 4. Chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend([float(x) for x in chroma_mean[:12]])
            except:
                features.extend([0.0] * 12)

            # 5. Pitch and tempo
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                pitch_values = []
                for t in range(min(10, pitches.shape[1])):  # Only check first 10 frames
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    features.extend([
                        float(np.mean(pitch_values)),
                        float(np.std(pitch_values)),
                        float(len(pitch_values))
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])

                # Tempo
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))

            except:
                features.extend([0.0] * 4)

            # Ensure exactly 50 features
            features = features[:50]  # Truncate if too many
            while len(features) < 50:  # Pad if too few
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            # If anything fails, return zero features
            return np.zeros(50, dtype=np.float32)

class SimpleEmotionNet(nn.Module):
    """Simplified neural network"""

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

            nn.Linear(64, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

class RobustMELDTrainer:
    """Simplified, robust MELD trainer"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        print(f"Using device: {self.device}")

    def load_data(self):
        """Load and prepare data"""
        print("Loading MELD dataset...")

        train_df = pd.read_csv('train_sent_emo.csv')
        dev_df = pd.read_csv('dev_sent_emo.csv')
        test_df = pd.read_csv('test_sent_emo.csv')

        print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

        # Encode emotions
        all_emotions = pd.concat([train_df['Emotion'], dev_df['Emotion'], test_df['Emotion']])
        self.label_encoder.fit(all_emotions)
        self.emotions = list(self.label_encoder.classes_)
        self.num_classes = len(self.emotions)

        print(f"Emotions: {self.emotions}")

        return train_df, dev_df, test_df

    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        # Load data
        train_df, dev_df, test_df = self.load_data()

        # Create datasets
        train_dataset = RobustMELDDataset(train_df, './train')
        dev_dataset = RobustMELDDataset(dev_df, './dev_splits_complete')

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create model
        model = SimpleEmotionNet(input_size=50, num_classes=self.num_classes).to(self.device)

        # Calculate class weights
        train_emotions_encoded = self.label_encoder.transform(train_df['Emotion'])
        class_weights = compute_class_weight('balanced', classes=np.unique(train_emotions_encoded), y=train_emotions_encoded)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_acc = 0
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_features, batch_emotions in train_pbar:
                # Convert emotions to indices
                batch_labels = torch.LongTensor([
                    self.label_encoder.transform([emotion])[0] for emotion in batch_emotions
                ]).to(self.device)

                batch_features = batch_features.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*train_correct/train_total:.1f}%'
                })

            # Validation
            model.eval()
            dev_loss = 0
            dev_correct = 0
            dev_total = 0

            with torch.no_grad():
                for batch_features, batch_emotions in dev_loader:
                    batch_labels = torch.LongTensor([
                        self.label_encoder.transform([emotion])[0] for emotion in batch_emotions
                    ]).to(self.device)

                    batch_features = batch_features.to(self.device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    dev_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    dev_total += batch_labels.size(0)
                    dev_correct += (predicted == batch_labels).sum().item()

            # Calculate accuracies
            train_acc = 100 * train_correct / train_total
            dev_acc = 100 * dev_correct / dev_total
            avg_train_loss = train_loss / len(train_loader)
            avg_dev_loss = dev_loss / len(dev_loader)

            # Update learning rate
            scheduler.step(avg_dev_loss)

            print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Dev Acc: {dev_acc:.2f}%')

            # Save best model
            if dev_acc > best_acc:
                best_acc = dev_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoder': self.label_encoder,
                    'emotions': self.emotions,
                    'accuracy': dev_acc
                }, 'robust_meld_model.pth')
                print(f'  New best model saved! Dev Acc: {dev_acc:.2f}%')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
        return model

def main():
    """Main training function"""
    print("Robust MELD Emotion Recognition Training")
    print("=" * 50)

    trainer = RobustMELDTrainer()
    model = trainer.train(epochs=30, batch_size=32)

    print("\nTraining completed successfully!")
    print("Model saved as: robust_meld_model.pth")

if __name__ == "__main__":
    main()
