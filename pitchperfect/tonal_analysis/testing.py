#!/usr/bin/env python3
"""
Optimized RAVDESS Trainer - Final version with all improvements
No CREMA-D combination, enhanced features, data augmentation, longer training
"""
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from tqdm import tqdm
import time
import glob
import random
import warnings
warnings.filterwarnings('ignore')

class OptimizedRAVDESSTrainer:
    """Final optimized trainer for maximum RAVDESS performance"""

    def __init__(self, dataset_path=".", cache_dir="./feature_cache"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using device: {self.device}")

    def load_ravdess_data(self):
        """Load RAVDESS dataset"""
        print("Loading RAVDESS dataset...")

        audio_files = glob.glob(os.path.join(self.dataset_path, "**/*.wav"), recursive=True)

        if not audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.dataset_path}")

        data = []
        emotion_mapping = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }

        for file_path in audio_files:
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')

                if len(parts) >= 3 and parts[0] == '03':
                    emotion_code = int(parts[2])
                    emotion = emotion_mapping.get(emotion_code, 'unknown')

                    if emotion != 'unknown':
                        data.append({
                            'file_path': file_path,
                            'emotion': emotion,
                            'filename': filename
                        })
            except:
                continue

        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} RAVDESS samples")
        print(f"Emotions: {sorted(df['emotion'].unique())}")

        return df

    def augment_audio(self, y, sr, augmentation_strength=0.3):
        """Apply data augmentation to increase training variety"""
        try:
            # Time stretching (change speed without changing pitch)
            if random.random() < augmentation_strength:
                stretch_factor = random.uniform(0.9, 1.1)
                y = librosa.effects.time_stretch(y, rate=stretch_factor)

            # Pitch shifting (change pitch without changing speed)
            if random.random() < augmentation_strength:
                n_steps = random.uniform(-1.0, 1.0)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

            # Add subtle noise
            if random.random() < augmentation_strength:
                noise_factor = random.uniform(0.001, 0.005)
                noise = np.random.normal(0, noise_factor, len(y))
                y = y + noise

            # Amplitude scaling
            if random.random() < augmentation_strength:
                scale_factor = random.uniform(0.8, 1.2)
                y = y * scale_factor

            return y

        except Exception as e:
            # If augmentation fails, return original
            return y

    def extract_optimized_features(self, file_path, target_sr=16000, max_length=6.0, use_augmentation=False):
        """Extract optimized features with delta MFCCs and better processing"""
        try:
            y, sr = librosa.load(file_path, sr=target_sr, duration=max_length)

            if len(y) == 0 or np.max(np.abs(y)) == 0:
                return np.zeros(65, dtype=np.float32)

            # Apply augmentation during training
            if use_augmentation:
                y = self.augment_audio(y, sr)

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

                # Delta MFCCs (13 features) - velocity of MFCC changes
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta_means = np.mean(mfcc_delta, axis=1)
                features.extend([float(x) for x in mfcc_delta_means])

                # Delta-delta MFCCs (13 features) - acceleration of MFCC changes
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

            # 4. Spectral contrast (7 features) - NEW
            try:
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_means = np.mean(spectral_contrast, axis=1)
                features.extend([float(x) for x in contrast_means])
            except:
                features.extend([0.0] * 7)

            # 5. Tonnetz (6 features) - NEW
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_means = np.mean(tonnetz, axis=1)
                features.extend([float(x) for x in tonnetz_means])
            except:
                features.extend([0.0] * 6)

            # 6. Advanced pitch features (1 feature)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo / 200))
            except:
                features.append(0.0)

            # Total: 39 + 8 + 4 + 7 + 6 + 1 = 65 features
            features = features[:65]
            while len(features) < 65:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return np.zeros(65, dtype=np.float32)

    def preprocess_with_augmentation(self, df, cache_name, augment_factor=2):
        """Preprocess with data augmentation to increase dataset size"""
        cache_file = os.path.join(self.cache_dir, f"{cache_name}_augmented.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached augmented features...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Processing {len(df)} files with {augment_factor}x augmentation...")
        processed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            file_path = row['file_path']
            emotion = row['emotion']

            if os.path.exists(file_path):
                # Original file
                features_orig = self.extract_optimized_features(file_path, use_augmentation=False)
                processed_data.append({
                    'file_path': file_path,
                    'features': features_orig,
                    'emotion': emotion,
                    'augmented': False
                })

                # Augmented versions
                for aug_id in range(augment_factor - 1):
                    features_aug = self.extract_optimized_features(file_path, use_augmentation=True)
                    processed_data.append({
                        'file_path': f"{file_path}_aug{aug_id}",
                        'features': features_aug,
                        'emotion': emotion,
                        'augmented': True
                    })

        print(f"Created {len(processed_data)} samples (original + augmented)")

        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)

        return processed_data

    def train_optimized_model(self, X_train, X_val, y_train, y_val, config, model_id=0):
        """Train single model with optimized settings"""
        torch.manual_seed(42 + model_id)
        np.random.seed(42 + model_id)

        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Create model (65 input features now)
        model = self.create_optimized_model(config['architecture']).to(self.device)

        # Loss and optimizer with better settings
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_acc = 0
        patience_counter = 0

        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == config['epochs']-1:
                print(f'  Model {model_id+1} Epoch {epoch+1}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%, Best: {best_acc:.1f}%')

            if patience_counter >= 15:  # More patience for longer training
                break

        return model, best_acc

    def train_optimized_ensemble(self, X_train, X_val, y_train, y_val, config, n_models=5):
        """Train optimized ensemble"""
        print(f"\nTraining optimized ensemble of {n_models} models...")

        models = []
        accuracies = []

        for i in range(n_models):
            print(f"\nTraining model {i+1}/{n_models}:")
            model, accuracy = self.train_optimized_model(X_train, X_val, y_train, y_val, config, i)
            models.append(model)
            accuracies.append(accuracy)

        print(f"\nIndividual model accuracies: {[f'{acc:.1f}%' for acc in accuracies]}")

        # Test ensemble
        ensemble_accuracy = self.evaluate_ensemble(models, X_val, y_val)
        print(f"Ensemble accuracy: {ensemble_accuracy:.1f}%")

        # Save ensemble
        torch.save({
            'models': [model.state_dict() for model in models],
            'label_encoder': self.label_encoder,
            'feature_scaler': self.feature_scaler,
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': accuracies,
            'config': config
        }, 'final_optimized_ensemble.pth')

        return models, ensemble_accuracy

    def evaluate_ensemble(self, models, X_val, y_val):
        """Evaluate ensemble with weighted voting"""
        all_predictions = []

        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_predictions.append(probs.cpu().numpy())

        # Weighted average (could weight by individual accuracy)
        ensemble_probs = np.mean(all_predictions, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        accuracy = np.mean(ensemble_preds == y_val) * 100
        return accuracy

    def run_optimized_experiment(self, config):
        """Run optimized experiment"""
        print(f"\n{'='*50}")
        print(f"OPTIMIZED EXPERIMENT: {config['name']}")
        print(f"{'='*50}")

        # Load RAVDESS only (no CREMA-D combination)
        df = self.load_ravdess_data()

        # Process with augmentation if specified
        if config.get('use_augmentation', False):
            processed_data = self.preprocess_with_augmentation(
                df, config['name'], augment_factor=config.get('augment_factor', 2)
            )
        else:
            processed_data = self.preprocess_standard(df, config['name'])

        if not processed_data:
            return 0

        # Prepare data
        X = np.array([item['features'] for item in processed_data])
        y = [item['emotion'] for item in processed_data]

        print(f"Final dataset size: {len(processed_data)} samples")

        self.label_encoder.fit(y)
        self.num_classes = len(self.label_encoder.classes_)
        y_encoded = self.label_encoder.transform(y)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        # Feature scaling
        if config.get('use_scaling', True):
            X_train = self.feature_scaler.fit_transform(X_train)
            X_val = self.feature_scaler.transform(X_val)

        # Random Forest baseline
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
        rf.fit(X_train, y_train)
        rf_accuracy = rf.score(X_val, y_val) * 100
        print(f"Random Forest accuracy: {rf_accuracy:.1f}%")

        # Train model(s)
        if config.get('use_ensemble', False):
            models, final_acc = self.train_optimized_ensemble(
                X_train, X_val, y_train, y_val, config,
                n_models=config.get('n_models', 5)
            )
        else:
            model, final_acc = self.train_optimized_model(X_train, X_val, y_train, y_val, config)

        return final_acc

    def preprocess_standard(self, df, cache_name):
        """Standard preprocessing without augmentation"""
        cache_file = os.path.join(self.cache_dir, f"{cache_name}_optimized.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached optimized features...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Extracting optimized features for {len(df)} files...")
        processed_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            if os.path.exists(row['file_path']):
                features = self.extract_optimized_features(row['file_path'], use_augmentation=False)
                processed_data.append({
                    'file_path': row['file_path'],
                    'features': features,
                    'emotion': row['emotion']
                })

        with open(cache_file, 'wb') as f:
            pickle.dump(processed_data, f)

        return processed_data

    def create_optimized_model(self, architecture):
        """Create optimized model architectures for 65 features"""
        if architecture == "simple":
            return OptimizedSimpleNet(65, self.num_classes)
        elif architecture == "medium":
            return OptimizedMediumNet(65, self.num_classes)
        else:
            return OptimizedLargeNet(65, self.num_classes)

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

def main():
    """Run final optimized experiments"""
    print("Final Optimized RAVDESS Emotion Recognition")
    print("=" * 50)

    trainer = OptimizedRAVDESSTrainer(
        dataset_path="/Users/nehirbektas/data/ravdess/Audio_Speech_Actors_01-24"
    )

    # Progressive optimization experiments
    experiments = [
        {
            "name": "optimized_features",
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 32,
            "architecture": "medium",
            "use_scaling": True,
            "weight_decay": 1e-4,
            "use_augmentation": False,
            "use_ensemble": False
        },
        {
            "name": "with_augmentation",
            "epochs": 40,
            "lr": 0.001,
            "batch_size": 32,
            "architecture": "medium",
            "use_scaling": True,
            "weight_decay": 1e-4,
            "use_augmentation": True,
            "augment_factor": 3,
            "use_ensemble": False
        },
        {
            "name": "final_ensemble",
            "epochs": 35,
            "lr": 0.001,
            "batch_size": 32,
            "architecture": "medium",
            "use_scaling": True,
            "weight_decay": 1e-4,
            "use_augmentation": True,
            "augment_factor": 2,
            "use_ensemble": True,
            "n_models": 5
        }
    ]

    results = {}
    baseline_acc = 64.6  # Your previous ensemble result

    for config in experiments:
        start_time = time.time()
        accuracy = trainer.run_optimized_experiment(config)
        duration = (time.time() - start_time) / 60

        results[config['name']] = {
            'accuracy': accuracy,
            'time': duration,
            'improvement': accuracy - baseline_acc
        }

        print(f"\nResult: {config['name']} = {accuracy:.1f}% (+{accuracy-baseline_acc:+.1f}%) in {duration:.1f} min")

    # Final summary
    print("\n" + "=" * 50)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 50)

    print(f"{'Configuration':25s} {'Accuracy':>9s} {'Improvement':>12s} {'Time':>8s}")
    print("-" * 56)

    for name, result in results.items():
        print(f"{name:25s} {result['accuracy']:8.1f}% {result['improvement']:+11.1f}% {result['time']:7.1f}m")

    best_name, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])

    print(f"\nFinal best result: {best_name}")
    print(f"Best accuracy: {best_result['accuracy']:.1f}%")
    print(f"Total improvement: +{best_result['improvement']:+.1f}% from ensemble baseline")

    if best_result['accuracy'] > 75:
        print("OUTSTANDING: State-of-the-art performance achieved")
    elif best_result['accuracy'] > 70:
        print("EXCELLENT: Strong competitive performance")
    elif best_result['accuracy'] > 65:
        print("SOLID: Good improvement from baseline")

if __name__ == "__main__":
    main()
