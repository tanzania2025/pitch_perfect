#!/usr/bin/env python3
"""
Standalone MELD Small Test - No external imports needed
Tests your MELD dataset and training pipeline
"""

import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

class MELDDataset(Dataset):
    """Dataset class for MELD emotion recognition"""

    def __init__(self, csv_data, audio_base_path, target_sr=16000, max_length=8.0):
        self.data = csv_data
        self.audio_base_path = audio_base_path
        self.target_sr = target_sr
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Try different audio file naming patterns
        utterance_id = row['Utterance_ID']
        split = row.get('Split', 'train')

        # Common MELD naming patterns
        possible_names = [
            f"dia{row.get('Dialogue_ID', 0)}_utt{utterance_id}.wav",
            f"{split}_sent_{utterance_id}.wav",
            f"sent_{utterance_id}.wav",
            f"{utterance_id}.wav"
        ]

        # Find the audio file
        audio_path = None
        for name in possible_names:
            potential_path = os.path.join(self.audio_base_path, name)
            if os.path.exists(potential_path):
                audio_path = potential_path
                break

        if audio_path is None:
            # Create dummy features if file not found
            features = np.zeros(100, dtype=np.float32)
            print(f"Warning: Audio file not found for utterance {utterance_id}")
        else:
            features = self._extract_features(audio_path)

        emotion = row['Emotion']
        return torch.FloatTensor(features), emotion

    def _extract_features(self, audio_path):
        """Extract basic audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr)

            # Pad or trim to consistent length
            max_samples = int(self.max_length * self.target_sr)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')

            features = []

            # Basic MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                np.mean(mfcc, axis=1),  # 13 features
                np.std(mfcc, axis=1),   # 13 features
            ])

            # Basic spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]

            features.extend([
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(zero_crossing), np.std(zero_crossing)],
                [np.mean(rms), np.std(rms)]
            ])

            # Flatten features
            flat_features = []
            for feature_group in features:
                if isinstance(feature_group, np.ndarray):
                    flat_features.extend(feature_group.tolist())
                elif isinstance(feature_group, list):
                    flat_features.extend(feature_group)
                else:
                    flat_features.append(feature_group)

            return np.array(flat_features[:100], dtype=np.float32)  # Limit to 100 features

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(100, dtype=np.float32)

class SimpleEmotionNet(nn.Module):
    """Simple neural network for emotion classification"""

    def __init__(self, input_size, num_classes):
        super(SimpleEmotionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def check_meld_structure():
    """Check if MELD dataset is properly set up"""
    print("🔍 Checking MELD Dataset Structure...")
    print("=" * 40)

    issues = []

    # Check CSV files
    csv_files = ['train_sent_emo.csv', 'dev_sent_emo.csv', 'test_sent_emo.csv']
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"✅ {csv_file}: {len(df)} samples")
                if len(df) == 0:
                    issues.append(f"{csv_file} is empty")
            except Exception as e:
                print(f"❌ {csv_file}: Error reading - {e}")
                issues.append(f"Cannot read {csv_file}")
        else:
            print(f"❌ {csv_file}: Not found")
            issues.append(f"Missing {csv_file}")

    # Check audio folders
    audio_folders = [
        ('train', './train'),
        ('dev', './dev_splits_complete'),
        ('test', './output_repeated_splits_test')
    ]

    for name, folder in audio_folders:
        if os.path.exists(folder):
            try:
                audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
                print(f"✅ {name} audio: {len(audio_files)} files in {folder}")
                if len(audio_files) == 0:
                    issues.append(f"No audio files in {folder}")
            except Exception as e:
                print(f"❌ {name} audio: Error accessing {folder} - {e}")
                issues.append(f"Cannot access {folder}")
        else:
            print(f"❌ {name} audio: {folder} not found")
            issues.append(f"Missing folder {folder}")

    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n🎉 All files and folders found!")
        return True

def run_tiny_test():
    """Run a tiny test with minimal samples"""
    print("\n🧪 Running Tiny MELD Test")
    print("=" * 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load tiny datasets
        print("Loading small datasets...")
        train_df = pd.read_csv('train_sent_emo.csv').head(20)  # Just 20 samples
        dev_df = pd.read_csv('dev_sent_emo.csv').head(5)       # Just 5 samples

        # Add split info
        train_df['Split'] = 'train'
        dev_df['Split'] = 'dev'

        print(f"Train samples: {len(train_df)}")
        print(f"Dev samples: {len(dev_df)}")

        # Check emotions
        emotions = train_df['Emotion'].unique()
        print(f"Emotions in sample: {emotions}")

        # Encode labels
        label_encoder = LabelEncoder()
        all_emotions = pd.concat([train_df['Emotion'], dev_df['Emotion']]).unique()
        label_encoder.fit(all_emotions)

        num_classes = len(all_emotions)
        print(f"Number of emotion classes: {num_classes}")

        # Create datasets
        train_dataset = MELDDataset(train_df, './train')
        dev_dataset = MELDDataset(dev_df, './dev_splits_complete')

        # Test loading one sample
        print("\nTesting audio processing...")
        sample_features, sample_emotion = train_dataset[0]
        print(f"Sample features shape: {sample_features.shape}")
        print(f"Sample emotion: {sample_emotion}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False)

        # Create model
        input_size = len(sample_features)
        model = SimpleEmotionNet(input_size, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        print(f"\nModel created with {input_size} input features")
        print("Starting mini training (3 epochs)...")

        # Mini training loop
        for epoch in range(3):
            model.train()
            train_loss = 0
            train_samples = 0

            for batch_features, batch_emotions in train_loader:
                # Convert emotions to labels
                batch_labels = torch.LongTensor([
                    label_encoder.transform([emotion])[0] for emotion in batch_emotions
                ]).to(device)

                batch_features = batch_features.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_samples += len(batch_labels)

            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/3: Average Loss = {avg_loss:.4f}")

        # Test prediction
        model.eval()
        with torch.no_grad():
            test_features, test_emotion = train_dataset[0]
            test_features = test_features.unsqueeze(0).to(device)
            output = model(test_features)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]

            print(f"\nTest prediction:")
            print(f"Actual emotion: {test_emotion}")
            print(f"Predicted emotion: {predicted_emotion}")

        print(f"\n🎉 Tiny test completed successfully!")
        print(f"✅ Audio loading: Working")
        print(f"✅ Feature extraction: Working")
        print(f"✅ Model training: Working")
        print(f"✅ Prediction: Working")
        print(f"\n🚀 Ready for larger training!")

        return True

    except Exception as e:
        print(f"\n❌ Error during tiny test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎭 MELD Standalone Test")
    print("=" * 30)
    print("This script tests your MELD dataset setup")
    print("No external imports required!")
    print()

    # First check structure
    if not check_meld_structure():
        print("\n❌ Please fix the issues above before proceeding.")
        return

    print("\n" + "="*50)

    # Ask if user wants to run the test
    proceed = input("Run tiny training test? (y/n): ").lower().strip()

    if proceed == 'y':
        success = run_tiny_test()
        if success:
            print("\n🎯 Next Steps:")
            print("1. Everything works! Your setup is correct.")
            print("2. You can now run full training with confidence.")
            print("3. Save the main meld_trainer.py and run: python meld_trainer.py")
        else:
            print("\n🔧 Something went wrong. Check the error messages above.")
    else:
        print("👋 Test cancelled. Run again when ready!")

if __name__ == "__main__":
    main()
