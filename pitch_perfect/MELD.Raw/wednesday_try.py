#!/usr/bin/env python3
"""
Enhanced MELD Trainer integrating comprehensive voice analysis features
Target: 40-50% accuracy through better features, text analysis, and data improvements
"""

import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from textblob import TextBlob
import os
from tqdm import tqdm
import warnings
import random
from collections import Counter
warnings.filterwarnings('ignore')

class EnhancedMELDDataset(Dataset):
    """Enhanced dataset using comprehensive voice analysis features"""

    def __init__(self, csv_data, audio_base_path, target_sr=16000, max_length=8.0, use_augmentation=False):
        self.data = csv_data
        self.audio_base_path = audio_base_path
        self.target_sr = target_sr
        self.max_length = max_length
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Construct filename
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        audio_path = os.path.join(self.audio_base_path, audio_filename)

        # Extract comprehensive audio features
        audio_features = self._extract_voice_analyzer_features(audio_path)

        # Extract text features from transcript
        text = row['Utterance']
        text_features = self._extract_text_features(text)

        # Combine features
        combined_features = np.concatenate([audio_features, text_features])

        emotion = row['Emotion']
        return torch.FloatTensor(combined_features), emotion

    def _extract_voice_analyzer_features(self, audio_path):
        """Extract comprehensive audio features based on voice analyzer"""
        try:
            if not os.path.exists(audio_path):
                return np.zeros(120, dtype=np.float32)

            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, duration=self.max_length)

            if len(y) == 0:
                return np.zeros(120, dtype=np.float32)

            # Apply augmentation if enabled
            if self.use_augmentation:
                y = self._apply_augmentation(y, sr)

            # Ensure consistent length
            target_length = int(self.max_length * self.target_sr)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            features = []

            # 1. Pitch Features (12 features)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)

                if pitch_values:
                    features.extend([
                        float(np.mean(pitch_values)),      # pitch_mean_hz
                        float(np.std(pitch_values)),       # pitch_std_hz
                        float(max(pitch_values) - min(pitch_values)),  # pitch_range_hz
                        float(np.median(pitch_values)),    # pitch_median_hz
                        float(np.std(pitch_values) / np.mean(pitch_values)),  # pitch_coefficient_variation
                        float(np.percentile(pitch_values, 25)),  # 25th percentile
                        float(np.percentile(pitch_values, 75)),  # 75th percentile
                        float(len([p for p in pitch_values if p > np.mean(pitch_values)])),  # high_pitch_count
                        float(len([p for p in pitch_values if p < np.mean(pitch_values)])),  # low_pitch_count
                        float(np.var(pitch_values)),       # pitch_variance
                        float(len(pitch_values) / len(y) * sr),  # voiced_ratio
                        float(np.mean(np.abs(np.diff(pitch_values))))  # pitch_jitter_approx
                    ])
                else:
                    features.extend([0.0] * 12)
            except:
                features.extend([0.0] * 12)

            # 2. Volume/Energy Features (15 features)
            try:
                rms = librosa.feature.rms(y=y)[0]
                rms_db = librosa.amplitude_to_db(rms, ref=np.max)

                features.extend([
                    float(np.mean(rms)),               # volume_mean
                    float(np.std(rms)),                # volume_std
                    float(np.max(rms)),                # volume_max
                    float(np.min(rms)),                # volume_min
                    float(np.max(rms) - np.min(rms)),  # dynamic_range
                    float(np.mean(rms_db)),            # volume_mean_db
                    float(np.std(rms_db)),             # volume_std_db
                    float(np.median(rms)),             # volume_median
                    float(np.percentile(rms, 25)),     # volume_25th
                    float(np.percentile(rms, 75)),     # volume_75th
                    float(np.mean(np.abs(np.diff(rms)))),  # energy_variation
                    float(np.std(rms) / np.mean(rms)), # shimmer_approx
                    float(len([r for r in rms if r > np.mean(rms)])),  # high_energy_frames
                    float(np.sum(rms > np.percentile(rms, 90))),  # peak_energy_frames
                    float(np.mean(rms**2))             # power_mean
                ])
            except:
                features.extend([0.0] * 15)

            # 3. Spectral Features (20 features)
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

                features.extend([
                    float(np.mean(spectral_centroids)),   # spectral_centroid_mean
                    float(np.std(spectral_centroids)),    # spectral_centroid_std
                    float(np.mean(spectral_rolloff)),     # spectral_rolloff_mean
                    float(np.std(spectral_rolloff)),      # spectral_rolloff_std
                    float(np.mean(spectral_bandwidth)),   # spectral_bandwidth_mean
                    float(np.std(spectral_bandwidth)),    # spectral_bandwidth_std
                    float(np.mean(zcr)),                  # zero_crossing_rate_mean
                    float(np.std(zcr)),                   # zero_crossing_rate_std
                    float(np.mean(spectral_flatness)),    # spectral_flatness_mean
                    float(np.std(spectral_flatness)),     # spectral_flatness_std
                    float(np.max(spectral_centroids)),    # spectral_centroid_max
                    float(np.min(spectral_centroids)),    # spectral_centroid_min
                    float(np.median(spectral_centroids)), # spectral_centroid_median
                    float(np.percentile(spectral_centroids, 75) - np.percentile(spectral_centroids, 25)),  # spectral_centroid_iqr
                    float(np.mean(spectral_rolloff / (spectral_centroids + 1e-8))),  # rolloff_centroid_ratio
                    float(np.mean(spectral_bandwidth / (spectral_centroids + 1e-8))),  # bandwidth_centroid_ratio
                    float(np.mean(zcr * sr / 2)),         # zcr_frequency
                    float(np.std(spectral_rolloff / (spectral_centroids + 1e-8))),  # spectral_variability
                    float(np.mean(np.abs(np.diff(spectral_centroids)))),  # spectral_centroid_variation
                    float(np.mean(np.abs(np.diff(spectral_rolloff))))     # spectral_rolloff_variation
                ])
            except:
                features.extend([0.0] * 20)

            # 4. MFCC Features (26 features)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                delta_mfcc = librosa.feature.delta(mfcc)

                # MFCC statistics
                features.extend([float(np.mean(mfcc[i])) for i in range(13)])  # 13 features
                features.extend([float(np.mean(delta_mfcc[i])) for i in range(13)])  # 13 features
            except:
                features.extend([0.0] * 26)

            # 5. Harmonic and Rhythm Features (15 features)
            try:
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_means = np.mean(chroma, axis=1)
                features.extend([float(x) for x in chroma_means])  # 12 features

                # Tempo and beat features
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))  # tempo_bpm

                # Rhythm regularity
                if len(beats) > 1:
                    beat_times = librosa.frames_to_time(beats, sr=sr)
                    inter_beat_intervals = np.diff(beat_times)
                    rhythm_regularity = np.std(inter_beat_intervals) / np.mean(inter_beat_intervals)
                    features.append(float(rhythm_regularity))
                else:
                    features.append(0.0)

                features.append(float(len(beats)))  # num_beats
            except:
                features.extend([0.0] * 15)

            # 6. Advanced Features (32 features)
            try:
                # Tonnetz features
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_means = np.mean(tonnetz, axis=1)
                features.extend([float(x) for x in tonnetz_means])  # 6 features

                # Mel spectrogram features
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
                mel_means = np.mean(mel_spec, axis=1)
                features.extend([float(x) for x in mel_means])  # 13 features

                # Spectral contrast
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_means = np.mean(spectral_contrast, axis=1)
                features.extend([float(x) for x in contrast_means])  # 7 features

                # Harmonic-percussive separation
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                harmonic_rms = np.mean(librosa.feature.rms(y=y_harmonic)[0])
                percussive_rms = np.mean(librosa.feature.rms(y=y_percussive)[0])
                hpr_ratio = harmonic_rms / (percussive_rms + 1e-8)

                features.extend([
                    float(harmonic_rms),    # harmonic_energy
                    float(percussive_rms),  # percussive_energy
                    float(hpr_ratio),       # harmonic_percussive_ratio
                    float(np.std(librosa.feature.rms(y=y_harmonic)[0])),  # harmonic_variability
                    float(np.std(librosa.feature.rms(y=y_percussive)[0])), # percussive_variability
                    float(np.mean(y_harmonic**2) / np.mean(y**2))  # harmonic_ratio
                ])  # 6 features
            except:
                features.extend([0.0] * 32)

            # Ensure exactly 120 audio features
            features = features[:120]
            while len(features) < 120:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return np.zeros(120, dtype=np.float32)

    def _apply_augmentation(self, y, sr):
        """Apply data augmentation"""
        if random.random() < 0.3:  # 30% chance
            # Pitch shift
            n_steps = random.choice([-2, -1, 1, 2])
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        if random.random() < 0.3:  # 30% chance
            # Time stretch
            rate = random.choice([0.9, 1.1])
            y = librosa.effects.time_stretch(y, rate=rate)

        if random.random() < 0.3:  # 30% chance
            # Add noise
            noise = np.random.normal(0, 0.005, len(y))
            y = y + noise

        return y

    def _extract_text_features(self, text):
        """Extract comprehensive text features"""
        features = []

        # Basic text statistics (10 features)
        words = text.split()
        features.extend([
            float(len(text)),                          # text_length
            float(len(words)),                         # word_count
            float(np.mean([len(w) for w in words])) if words else 0,  # avg_word_length
            float(len(set(words)) / len(words)) if words else 0,      # vocabulary_richness
            float(len([w for w in words if len(w) > 6])),             # long_words
            float(len([w for w in words if w.isupper()])),            # uppercase_words
            float(len([w for w in words if w.islower()])),            # lowercase_words
            float(text.count(' ') + 1),                               # sentence_count_approx
            float(sum(1 for c in text if c.isalpha())),               # letter_count
            float(sum(1 for c in text if c.isdigit()))                # digit_count
        ])

        # Punctuation features (10 features)
        features.extend([
            float(text.count('!')),      # exclamation_marks
            float(text.count('?')),      # question_marks
            float(text.count('.')),      # periods
            float(text.count(',')),      # commas
            float(text.count(';')),      # semicolons
            float(text.count(':')),      # colons
            float(text.count('-')),      # hyphens
            float(text.count('"')),      # quotes
            float(text.count("'")),      # apostrophes
            float(text.count('(')),      # parentheses
        ])

        # Sentiment and emotion indicators (10 features)
        try:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
        except:
            sentiment_polarity = 0.0
            sentiment_subjectivity = 0.0

        # Emotion word lists
        positive_words = ['good', 'great', 'happy', 'love', 'wonderful', 'excellent', 'amazing', 'fantastic', 'awesome', 'perfect']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'sad', 'angry', 'disgusting', 'annoying', 'stupid']
        excitement_words = ['wow', 'omg', 'amazing', 'incredible', 'unbelievable', 'awesome', 'fantastic', 'brilliant']
        calm_words = ['okay', 'fine', 'alright', 'normal', 'usual', 'regular', 'standard', 'typical']

        text_lower = text.lower()
        features.extend([
            float(sentiment_polarity),                                           # sentiment_polarity
            float(sentiment_subjectivity),                                       # sentiment_subjectivity
            float(sum(1 for word in positive_words if word in text_lower)),      # positive_word_count
            float(sum(1 for word in negative_words if word in text_lower)),      # negative_word_count
            float(sum(1 for word in excitement_words if word in text_lower)),    # excitement_word_count
            float(sum(1 for word in calm_words if word in text_lower)),          # calm_word_count
            float(len([w for w in words if w.endswith('ing')])),                 # progressive_verbs
            float(len([w for w in words if w.endswith('ed')])),                  # past_tense_verbs
            float(len([w for w in words if w.endswith('ly')])),                  # adverbs
            float(text.count('...'))                                             # ellipses
        ])

        # Ensure exactly 30 text features
        features = features[:30]
        while len(features) < 30:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

class EnhancedEmotionNet(nn.Module):
    """Enhanced neural network architecture"""

    def __init__(self, audio_features=120, text_features=30, num_classes=7, dropout=0.4):
        super(EnhancedEmotionNet, self).__init__()

        # Separate processing for audio and text features
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout/2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/4)
        )

        self.text_branch = nn.Sequential(
            nn.Linear(text_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )

        # Combined processing
        combined_size = 64 + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),

            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Split features
        audio_features = x[:, :120]
        text_features = x[:, 120:]

        # Process through branches
        audio_out = self.audio_branch(audio_features)
        text_out = self.text_branch(text_features)

        # Combine and classify
        combined = torch.cat([audio_out, text_out], dim=1)
        output = self.classifier(combined)

        return output

class EnhancedMELDTrainer:
    """Enhanced trainer with data balancing and improved features"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        print(f"Using device: {self.device}")

    def load_and_balance_data(self, balance_method='oversample'):
        """Load data with balancing"""
        print("Loading MELD dataset...")

        train_df = pd.read_csv('train_sent_emo.csv')
        dev_df = pd.read_csv('dev_sent_emo.csv')
        test_df = pd.read_csv('test_sent_emo.csv')

        print("Original emotion distribution:")
        original_counts = train_df['Emotion'].value_counts()
        print(original_counts)

        if balance_method == 'oversample':
            # Oversample minority classes
            emotion_counts = Counter(train_df['Emotion'])
            target_count = max(emotion_counts.values())  # Match largest class

            balanced_dfs = []
            for emotion in emotion_counts.keys():
                emotion_df = train_df[train_df['Emotion'] == emotion]
                if len(emotion_df) < target_count:
                    # Oversample with replacement
                    sampled_df = emotion_df.sample(n=target_count, replace=True, random_state=42)
                else:
                    sampled_df = emotion_df
                balanced_dfs.append(sampled_df)

            train_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)

        elif balance_method == 'undersample':
            # Undersample majority classes
            emotion_counts = Counter(train_df['Emotion'])
            target_count = min(emotion_counts.values())  # Match smallest class

            balanced_dfs = []
            for emotion in emotion_counts.keys():
                emotion_df = train_df[train_df['Emotion'] == emotion]
                sampled_df = emotion_df.sample(n=min(target_count, len(emotion_df)), random_state=42)
                balanced_dfs.append(sampled_df)

            train_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)

        print(f"\nAfter {balance_method}:")
        print(f"Original size: {len(pd.read_csv('train_sent_emo.csv'))}")
        print(f"Balanced size: {len(train_df)}")
        print(train_df['Emotion'].value_counts())

        # Encode emotions
        all_emotions = pd.concat([train_df['Emotion'], dev_df['Emotion'], test_df['Emotion']])
        self.label_encoder.fit(all_emotions)
        self.emotions = list(self.label_encoder.classes_)
        self.num_classes = len(self.emotions)

        return train_df, dev_df, test_df

    def train_enhanced(self, epochs=60, batch_size=24, learning_rate=0.001):
        """Enhanced training with all improvements"""
        # Load balanced data
        train_df, dev_df, test_df = self.load_and_balance_data(balance_method='oversample')

        # Create datasets with augmentation for training
        train_dataset = EnhancedMELDDataset(train_df, './train', use_augmentation=True)
        dev_dataset = EnhancedMELDDataset(dev_df, './dev_splits_complete', use_augmentation=False)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create enhanced model
        model = EnhancedEmotionNet(
            audio_features=120,
            text_features=30,
            num_classes=self.num_classes
        ).to(self.device)

        print(f"Enhanced model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Calculate class weights
        train_emotions_encoded = self.label_encoder.transform(train_df['Emotion'])
        class_weights = compute_class_weight('balanced', classes=np.unique(train_emotions_encoded), y=train_emotions_encoded)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

        best_acc = 0
        patience_counter = 0

        print(f"\nStarting enhanced training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_features, batch_emotions in train_pbar:
                batch_labels = torch.LongTensor([
                    self.label_encoder.transform([emotion])[0] for emotion in batch_emotions
                ]).to(self.device)

                batch_features = batch_features.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()

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
            avg_dev_loss = dev_loss / len(dev_loader)

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
                    'accuracy': dev_acc,
                    'audio_features': 120,
                    'text_features': 30
                }, 'enhanced_meld_model.pth')
                print(f'  New best model saved! Dev Acc: {dev_acc:.2f}%')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= 15:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f"\nEnhanced training complete! Best accuracy: {best_acc:.2f}%")
        return model, best_acc

def main():
    """Main function for enhanced training"""
    print("Enhanced MELD Emotion Recognition Training")
    print("=" * 60)
    print("Improvements:")
    print("- 120 comprehensive voice analyzer features")
    print("- 30 text analysis features")
    print("- Data augmentation (pitch, time, noise)")
    print("- Class balancing via oversampling")
    print("- Enhanced neural architecture")
    print("- Advanced training techniques")
    print("=" * 60)

    trainer = EnhancedMELDTrainer()
    model, best_accuracy = trainer.train_enhanced(epochs=60, batch_size=24)

    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

    if best_accuracy >= 45.0:
        print("EXCELLENT: Target accuracy range achieved!")
    elif best_accuracy >= 35.0:
        print("GOOD: Strong improvement over baseline")
    elif best_accuracy >= 25.0:
        print("MODERATE: Decent improvement")
    else:
        print("NEEDS WORK: Consider further optimizations")

if __name__ == "__main__":
    main()
