#!/usr/bin/env python3
"""
Custom Emotion Detection Training for MELD Dataset
Train your own emotion recognition model using the MELD dataset
"""

import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MELDDataset(Dataset):
    """
    Custom dataset for MELD emotion recognition
    """
    def __init__(self, csv_data, audio_base_path, transform=None, target_sr=16000, max_length=8.0):
        self.data = csv_data
        self.audio_base_path = audio_base_path
        self.transform = transform
        self.target_sr = target_sr
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # MELD audio file naming convention
        utterance_id = row['Utterance_ID']
        split = row.get('Split', 'train')  # train, dev, or test

        # Different naming patterns for different splits
        possible_names = [
            f"dia{row.get('Dialogue_ID', 0)}_utt{utterance_id}.wav",
            f"{split}_sent_{utterance_id}.wav",
            f"sent_{utterance_id}.wav",
            f"{utterance_id}.wav"
        ]

        # Try to find the audio file
        audio_path = None
        split_path = self.audio_base_path if isinstance(self.audio_base_path, str) else self.audio_base_path.get(split, '.')

        for name in possible_names:
            potential_path = os.path.join(split_path, name)
            if os.path.exists(potential_path):
                audio_path = potential_path
                break

        # If not found, try without subdirectories
        if audio_path is None:
            for name in possible_names:
                potential_path = name
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break

        if audio_path is None:
            print(f"Warning: Audio file not found for utterance {utterance_id} in split {split}")
            # Create a dummy path for now
            audio_path = os.path.join(split_path, f"dummy_{utterance_id}.wav")

        emotion = row['Emotion']

        # Load and process audio
        try:
            features = self._extract_features(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Return zero features if file can't be loaded
            features = np.zeros(200, dtype=np.float32)

        return torch.FloatTensor(features), emotion

    def _extract_features(self, audio_path):
        """
        Extract comprehensive audio features optimized for MELD emotion recognition
        """
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=self.target_sr)

        # Ensure consistent length (pad or trim)
        max_samples = int(self.max_length * self.target_sr)
        if len(y) > max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - len(y)), mode='constant')

        features = []

        # 1. MFCC features (most important for speech emotion)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend([
            np.mean(mfcc, axis=1),      # 13 features
            np.std(mfcc, axis=1),       # 13 features
            np.max(mfcc, axis=1),       # 13 features
            np.min(mfcc, axis=1),       # 13 features
        ])

        # 2. Delta MFCC (first and second derivatives)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend([
            np.mean(delta_mfcc, axis=1),    # 13 features
            np.mean(delta2_mfcc, axis=1),   # 13 features
        ])

        # 3. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

        features.extend([
            [np.mean(spectral_centroids), np.std(spectral_centroids), np.max(spectral_centroids), np.min(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.max(spectral_bandwidth), np.min(spectral_bandwidth)],
            np.mean(spectral_contrast, axis=1),  # 7 features
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate), np.max(zero_crossing_rate), np.min(zero_crossing_rate)]
        ])

        # 4. Energy and rhythm features
        rms_energy = librosa.feature.rms(y=y)[0]
        features.append([np.mean(rms_energy), np.std(rms_energy), np.max(rms_energy), np.min(rms_energy)])

        # 5. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([
            np.mean(chroma, axis=1),    # 12 features
            np.std(chroma, axis=1),     # 12 features
        ])

        # 6. Tonnetz features (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features.extend([
            np.mean(tonnetz, axis=1),   # 6 features
            np.std(tonnetz, axis=1),    # 6 features
        ])

        # 7. Mel-scale spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
        features.extend([
            np.mean(mel_spec, axis=1),  # 13 features
            np.std(mel_spec, axis=1),   # 13 features
        ])

        # 8. Pitch features
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features.append([
                    np.mean(pitch_values), np.std(pitch_values),
                    np.max(pitch_values), np.min(pitch_values),
                    np.median(pitch_values)
                ])
            else:
                features.append([0, 0, 0, 0, 0])
        except:
            features.append([0, 0, 0, 0, 0])

        # 9. Tempo and beat features
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append([tempo, len(beats), np.std(np.diff(beats)) if len(beats) > 1 else 0])
        except:
            features.append([0, 0, 0])

        flat_features = []
        for feature_group in features:
            if isinstance(feature_group, np.ndarray):
                flat_features.extend(feature_group.flatten().tolist())
            elif isinstance(feature_group, list):
                flat_features.extend(np.array(feature_group).flatten().tolist())
            else:
                flat_features.append(float(feature_group))

        # Convert to numpy array and ensure fixed size
        flat_features = np.array(flat_features, dtype=np.float32)

        # Force fixed size of 200 features
        if len(flat_features) < 200:
            padded = np.zeros(200, dtype=np.float32)
            padded[:len(flat_features)] = flat_features
            return padded
        else:
            return flat_features[:200]



class MELDEmotionNet(nn.Module):
    """
    Deep neural network for MELD emotion classification
    """
    def __init__(self, input_size, num_classes, hidden_sizes=[1024, 512, 256, 128], dropout=0.3):
        super(MELDEmotionNet, self).__init__()

        layers = []
        prev_size = input_size

        # Input layer with batch norm
        layers.extend([
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        prev_size = hidden_sizes[0]

        # Hidden layers
        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

class MELDTrainer:
    """
    Main trainer class for MELD emotion recognition
    """
    def __init__(self, train_csv='train_sent_emo.csv', dev_csv='dev_sent_emo.csv',
                 test_csv='test_sent_emo.csv', audio_paths=None):
        self.train_csv = train_csv
        self.dev_csv = dev_csv
        self.test_csv = test_csv

        # Default audio paths based on your extracted structure
        if audio_paths is None:
            self.audio_paths = {
                'train': './train',
                'dev': './dev_splits_complete',
                'test': './output_repeated_splits_test'
            }
        else:
            self.audio_paths = audio_paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MELD has 7 emotions
        self.meld_emotions = [
            'neutral', 'surprise', 'fear', 'sadness',
            'joy', 'disgust', 'anger'
        ]

        print(f"Initializing MELD Trainer on {self.device}")
        print(f"Expected emotions: {self.meld_emotions}")

    def load_and_prepare_data(self, balance_dataset=False):
        """
        Load and prepare MELD dataset
        """
        print("Loading MELD dataset...")

        # Load train, dev, test CSVs
        train_df = pd.read_csv(self.train_csv)
        dev_df = pd.read_csv(self.dev_csv)
        test_df = pd.read_csv(self.test_csv)

        # Add split information
        train_df['Split'] = 'train'
        dev_df['Split'] = 'dev'
        test_df['Split'] = 'test'

        print(f"Loaded datasets:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Dev: {len(dev_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Total: {len(train_df) + len(dev_df) + len(test_df)} samples")

        # Check emotion distribution in training set
        print(f"\nEmotion distribution in training set:")
        emotion_counts = train_df['Emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count / len(train_df)) * 100
            print(f"  {emotion}: {count} samples ({percentage:.1f}%)")

        # Balance dataset if requested
        if balance_dataset:
            min_samples = emotion_counts.min()
            balanced_dfs = []
            for emotion in emotion_counts.index:
                emotion_df = train_df[train_df['Emotion'] == emotion].sample(n=min_samples, random_state=42)
                balanced_dfs.append(emotion_df)
            train_df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"\nAfter balancing training set: {len(train_df)} samples ({min_samples} per emotion)")

        # Encode labels for all datasets
        all_emotions = pd.concat([train_df['Emotion'], dev_df['Emotion'], test_df['Emotion']]).unique()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_emotions)

        # Apply encoding
        train_df['Emotion_Encoded'] = self.label_encoder.transform(train_df['Emotion'])
        dev_df['Emotion_Encoded'] = self.label_encoder.transform(dev_df['Emotion'])
        test_df['Emotion_Encoded'] = self.label_encoder.transform(test_df['Emotion'])

        # Store emotion labels
        self.emotion_labels = list(self.label_encoder.classes_)
        self.num_classes = len(self.emotion_labels)

        print(f"\nFinal emotion classes: {self.emotion_labels}")
        print(f"Number of classes: {self.num_classes}")

        return train_df, dev_df, test_df

    def create_model(self, input_size):
        """Create the MELD emotion recognition model"""
        model = MELDEmotionNet(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden_sizes=[1024, 512, 256, 128],
            dropout=0.4
        ).to(self.device)

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        return model

    def train_model(self, train_df, dev_df, epochs=150, batch_size=32, learning_rate=0.001):
        """
        Train the MELD emotion recognition model
        """
        print(f"\nStarting training...")

        # Create datasets
        train_dataset = MELDDataset(train_df, self.audio_paths.get('train', './train'))
        dev_dataset = MELDDataset(dev_df, self.audio_paths.get('dev', './dev_splits_complete'))

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Get input size from first sample
        sample_features, _ = train_dataset[0]
        input_size = len(sample_features)
        print(f"Feature vector size: {input_size}")

        # Create model
        model = self.create_model(input_size)

        # Calculate class weights for imbalanced dataset
        y_train = train_df['Emotion_Encoded'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        print(f"\nClass weights: {dict(zip(self.emotion_labels, class_weights))}")

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


        # Training history
        history = {
            'train_loss': [], 'dev_loss': [],
            'train_acc': [], 'dev_acc': []
        }

        best_dev_acc = 0
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_features, batch_emotions in train_pbar:
                # Convert string emotions to encoded labels
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
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*train_correct/train_total:.2f}%'
                })

            # Development phase
            model.eval()
            dev_loss = 0
            dev_correct = 0
            dev_total = 0

            with torch.no_grad():
                dev_pbar = tqdm(dev_loader, desc=f'Epoch {epoch+1}/{epochs} [Dev]')
                for batch_features, batch_emotions in dev_pbar:
                    batch_labels = torch.LongTensor([
                        self.label_encoder.transform([emotion])[0] for emotion in batch_emotions
                    ]).to(self.device)

                    batch_features = batch_features.to(self.device)

                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    dev_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    dev_total += batch_labels.size(0)
                    dev_correct += (predicted == batch_labels).sum().item()

                    dev_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100*dev_correct/dev_total:.2f}%'
                    })

            # Calculate epoch metrics
            train_acc = 100 * train_correct / train_total
            dev_acc = 100 * dev_correct / dev_total
            avg_train_loss = train_loss / len(train_loader)
            avg_dev_loss = dev_loss / len(dev_loader)

            # Update learning rate
            scheduler.step(avg_dev_loss)

            # Store history
            history['train_loss'].append(avg_train_loss)
            history['dev_loss'].append(avg_dev_loss)
            history['train_acc'].append(train_acc)
            history['dev_acc'].append(dev_acc)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Dev Loss: {avg_dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%')

            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'label_encoder': self.label_encoder,
                    'emotion_labels': self.emotion_labels,
                    'dev_accuracy': dev_acc,
                    'input_size': input_size
                }, 'best_meld_emotion_model.pth')
                print(f'  💾 New best model saved! Dev Acc: {dev_acc:.2f}%')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            print('-' * 60)

        self.model = model
        self.plot_training_history(history)

        return model, history

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Plot losses
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['dev_loss'], 'r-', label='Development Loss', linewidth=2)
        ax1.set_title('MELD Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracies
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['dev_acc'], 'r-', label='Development Accuracy', linewidth=2)
        ax2.set_title('MELD Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('meld_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, test_df):
        """Evaluate the trained model on test set"""
        print("\n🎯 Evaluating model on test set...")

        # Load best model
        checkpoint = torch.load('best_meld_emotion_model.pth', map_location=self.device)

        # Recreate model
        model = MELDEmotionNet(
            input_size=checkpoint['input_size'],
            num_classes=len(checkpoint['emotion_labels'])
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create test dataset
        test_dataset = MELDDataset(test_df, self.audio_paths.get('test', './output_repeated_splits_test'))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_features, batch_emotions in tqdm(test_loader, desc='Testing'):
                batch_labels = torch.LongTensor([
                    self.label_encoder.transform([emotion])[0] for emotion in batch_emotions
                ]).to(self.device)

                batch_features = batch_features.to(self.device)

                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(batch_labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate metrics
        test_accuracy = accuracy_score(y_true, y_pred)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.emotion_labels)

        print(f"\n🎉 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"\n📊 Detailed Classification Report:")
        print(report)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.emotion_labels, yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix - MELD Emotion Recognition', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('meld_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return test_accuracy, report

def main():
    """
    Main training pipeline for MELD emotion recognition
    """
    print("🎭 MELD Custom Emotion Recognition Training")
    print("=" * 60)
    print("Dataset: Multimodal EmotionLines Dataset")
    print("Emotions: neutral, surprise, fear, sadness, joy, disgust, anger")
    print("=" * 60)

    # Initialize trainer
    # Update these paths to match your MELD dataset structure
    trainer = MELDTrainer(
        train_csv='train_sent_emo.csv',
        dev_csv='dev_sent_emo.csv',
        test_csv='test_sent_emo.csv',
        audio_paths={
            'train': './train',
            'dev': './dev_splits_complete',
            'test': './output_repeated_splits_test'
        }
    )

    # Load and prepare data
    train_df, dev_df, test_df = trainer.load_and_prepare_data(
        balance_dataset=False  # Set to True if you want balanced classes
    )

    # Train the model
    model, history = trainer.train_model(
        train_df, dev_df,
        epochs=100,  # Reduce if training takes too long
        batch_size=32,  # Reduce if you run out of memory
        learning_rate=0.001
    )

    # Evaluate the model
    test_accuracy, report = trainer.evaluate_model(test_df)

    print(f"\n🎉 MELD Training Complete!")
    print(f"📊 Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"💾 Best model saved as: best_meld_emotion_model.pth")
    print(f"📈 Training plots saved as: meld_training_history.png")
    print(f"🔥 Confusion matrix saved as: meld_confusion_matrix.png")

    print(f"\n🔧 To use your trained model:")
    print(f"from meld_trainer import MELDTrainer")
    print(f"# Load and predict emotions on new audio files")

if __name__ == "__main__":
    main()
