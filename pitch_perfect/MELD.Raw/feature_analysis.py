#!/usr/bin/env python3
"""
MELD Audio Feature Extraction and Correlation Analysis
Extracts 120 comprehensive audio features and analyzes correlations with emotions/sentiments
"""

import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MELDFeatureExtractor:
    """Extract comprehensive audio features from MELD dataset"""

    def __init__(self, target_sr=16000, max_length=8.0):
        self.target_sr = target_sr
        self.max_length = max_length
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """Define names for all 120 features"""
        names = []

        # Pitch features (12)
        names.extend([
            'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_median',
            'pitch_coeff_var', 'pitch_25th', 'pitch_75th', 'pitch_high_count',
            'pitch_low_count', 'pitch_variance', 'voiced_ratio', 'pitch_jitter'
        ])

        # Volume/Energy features (15)
        names.extend([
            'volume_mean', 'volume_std', 'volume_max', 'volume_min',
            'dynamic_range', 'volume_mean_db', 'volume_std_db', 'volume_median',
            'volume_25th', 'volume_75th', 'energy_variation', 'shimmer',
            'high_energy_frames', 'peak_energy_frames', 'power_mean'
        ])

        # Spectral features (20)
        names.extend([
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
            'spectral_rolloff_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'zcr_mean', 'zcr_std', 'spectral_flatness_mean', 'spectral_flatness_std',
            'spectral_centroid_max', 'spectral_centroid_min', 'spectral_centroid_median',
            'spectral_centroid_iqr', 'rolloff_centroid_ratio', 'bandwidth_centroid_ratio',
            'zcr_frequency', 'spectral_variability', 'spectral_centroid_variation',
            'spectral_rolloff_variation'
        ])

        # MFCC features (26)
        for i in range(13):
            names.append(f'mfcc_{i}_mean')
        for i in range(13):
            names.append(f'mfcc_{i}_delta')

        # Harmonic and Rhythm features (15)
        for i in range(12):
            names.append(f'chroma_{i}')
        names.extend(['tempo_bpm', 'rhythm_regularity', 'num_beats'])

        # Advanced features (32)
        for i in range(6):
            names.append(f'tonnetz_{i}')
        for i in range(13):
            names.append(f'mel_{i}')
        for i in range(7):
            names.append(f'spectral_contrast_{i}')
        names.extend([
            'harmonic_energy', 'percussive_energy', 'hpr_ratio',
            'harmonic_variability', 'percussive_variability', 'harmonic_ratio'
        ])

        return names

    def extract_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            if not os.path.exists(audio_path):
                return np.zeros(120, dtype=np.float32)

            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, duration=self.max_length)

            if len(y) == 0:
                return np.zeros(120, dtype=np.float32)

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
                        float(np.mean(pitch_values)),
                        float(np.std(pitch_values)),
                        float(max(pitch_values) - min(pitch_values)),
                        float(np.median(pitch_values)),
                        float(np.std(pitch_values) / np.mean(pitch_values)),
                        float(np.percentile(pitch_values, 25)),
                        float(np.percentile(pitch_values, 75)),
                        float(len([p for p in pitch_values if p > np.mean(pitch_values)])),
                        float(len([p for p in pitch_values if p < np.mean(pitch_values)])),
                        float(np.var(pitch_values)),
                        float(len(pitch_values) / len(y) * sr),
                        float(np.mean(np.abs(np.diff(pitch_values))))
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
                    float(np.mean(rms)),
                    float(np.std(rms)),
                    float(np.max(rms)),
                    float(np.min(rms)),
                    float(np.max(rms) - np.min(rms)),
                    float(np.mean(rms_db)),
                    float(np.std(rms_db)),
                    float(np.median(rms)),
                    float(np.percentile(rms, 25)),
                    float(np.percentile(rms, 75)),
                    float(np.mean(np.abs(np.diff(rms)))),
                    float(np.std(rms) / np.mean(rms)),
                    float(len([r for r in rms if r > np.mean(rms)])),
                    float(np.sum(rms > np.percentile(rms, 90))),
                    float(np.mean(rms**2))
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
                    float(np.mean(spectral_centroids)),
                    float(np.std(spectral_centroids)),
                    float(np.mean(spectral_rolloff)),
                    float(np.std(spectral_rolloff)),
                    float(np.mean(spectral_bandwidth)),
                    float(np.std(spectral_bandwidth)),
                    float(np.mean(zcr)),
                    float(np.std(zcr)),
                    float(np.mean(spectral_flatness)),
                    float(np.std(spectral_flatness)),
                    float(np.max(spectral_centroids)),
                    float(np.min(spectral_centroids)),
                    float(np.median(spectral_centroids)),
                    float(np.percentile(spectral_centroids, 75) - np.percentile(spectral_centroids, 25)),
                    float(np.mean(spectral_rolloff / (spectral_centroids + 1e-8))),
                    float(np.mean(spectral_bandwidth / (spectral_centroids + 1e-8))),
                    float(np.mean(zcr * sr / 2)),
                    float(np.std(spectral_rolloff / (spectral_centroids + 1e-8))),
                    float(np.mean(np.abs(np.diff(spectral_centroids)))),
                    float(np.mean(np.abs(np.diff(spectral_rolloff))))
                ])
            except:
                features.extend([0.0] * 20)

            # 4. MFCC Features (26 features)
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                delta_mfcc = librosa.feature.delta(mfcc)

                features.extend([float(np.mean(mfcc[i])) for i in range(13)])
                features.extend([float(np.mean(delta_mfcc[i])) for i in range(13)])
            except:
                features.extend([0.0] * 26)

            # 5. Harmonic and Rhythm Features (15 features)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_means = np.mean(chroma, axis=1)
                features.extend([float(x) for x in chroma_means])

                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.append(float(tempo))

                if len(beats) > 1:
                    beat_times = librosa.frames_to_time(beats, sr=sr)
                    inter_beat_intervals = np.diff(beat_times)
                    rhythm_regularity = np.std(inter_beat_intervals) / np.mean(inter_beat_intervals)
                    features.append(float(rhythm_regularity))
                else:
                    features.append(0.0)

                features.append(float(len(beats)))
            except:
                features.extend([0.0] * 15)

            # 6. Advanced Features (32 features)
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                tonnetz_means = np.mean(tonnetz, axis=1)
                features.extend([float(x) for x in tonnetz_means])

                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13)
                mel_means = np.mean(mel_spec, axis=1)
                features.extend([float(x) for x in mel_means])

                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_means = np.mean(spectral_contrast, axis=1)
                features.extend([float(x) for x in contrast_means])

                y_harmonic, y_percussive = librosa.effects.hpss(y)
                harmonic_rms = np.mean(librosa.feature.rms(y=y_harmonic)[0])
                percussive_rms = np.mean(librosa.feature.rms(y=y_percussive)[0])
                hpr_ratio = harmonic_rms / (percussive_rms + 1e-8)

                features.extend([
                    float(harmonic_rms),
                    float(percussive_rms),
                    float(hpr_ratio),
                    float(np.std(librosa.feature.rms(y=y_harmonic)[0])),
                    float(np.std(librosa.feature.rms(y=y_percussive)[0])),
                    float(np.mean(y_harmonic**2) / np.mean(y**2))
                ])
            except:
                features.extend([0.0] * 32)

            # Ensure exactly 120 features
            features = features[:120]
            while len(features) < 120:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros(120, dtype=np.float32)

class MELDCorrelationAnalyzer:
    """Analyze correlations between audio features and emotions/sentiments"""

    def __init__(self):
        self.extractor = MELDFeatureExtractor()
        self.feature_df = None

    def extract_all_features(self, sample_size=None):
        """Extract features from all MELD audio files"""
        print("Loading MELD dataset...")

        # Load CSV files
        datasets = {
            'train': ('train_sent_emo.csv', './train'),
            'dev': ('dev_sent_emo.csv', './dev_splits_complete'),
            'test': ('test_sent_emo.csv', './test_splits_complete')
        }

        all_data = []

        for split_name, (csv_file, audio_dir) in datasets.items():
            if not os.path.exists(csv_file):
                print(f"Warning: {csv_file} not found, skipping {split_name}")
                continue

            print(f"Processing {split_name} split...")
            df = pd.read_csv(csv_file)

            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                print(f"Sampling {len(df)} files from {split_name}")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name} features"):
                # Construct audio file path
                dialogue_id = row['Dialogue_ID']
                utterance_id = row['Utterance_ID']
                audio_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)

                # Extract features
                features = self.extractor.extract_features(audio_path)

                # Create feature dictionary
                feature_dict = {}
                for i, feature_name in enumerate(self.extractor.feature_names):
                    feature_dict[feature_name] = features[i]

                # Add metadata
                feature_dict.update({
                    'split': split_name,
                    'dialogue_id': dialogue_id,
                    'utterance_id': utterance_id,
                    'speaker': row['Speaker'],
                    'emotion': row['Emotion'],
                    'sentiment': row['Sentiment'],
                    'utterance': row['Utterance'],
                    'audio_exists': os.path.exists(audio_path)
                })

                all_data.append(feature_dict)

        # Create DataFrame
        self.feature_df = pd.DataFrame(all_data)
        print(f"Created feature dataframe with {len(self.feature_df)} samples and {len(self.feature_df.columns)} columns")

        return self.feature_df

    def analyze_correlations(self, save_plots=True):
        """Analyze correlations between features and emotions/sentiments"""
        if self.feature_df is None:
            raise ValueError("No feature data available. Run extract_all_features() first.")

        print("Analyzing correlations...")

        # Encode categorical variables
        le_emotion = LabelEncoder()
        le_sentiment = LabelEncoder()

        feature_df_analysis = self.feature_df.copy()
        feature_df_analysis['emotion_encoded'] = le_emotion.fit_transform(feature_df_analysis['emotion'])
        feature_df_analysis['sentiment_encoded'] = le_sentiment.fit_transform(feature_df_analysis['sentiment'])

        # Get feature columns only
        feature_cols = self.extractor.feature_names

        # Calculate correlations with emotions
        emotion_correlations = []
        for feature in feature_cols:
            if feature in feature_df_analysis.columns:
                corr = feature_df_analysis[feature].corr(feature_df_analysis['emotion_encoded'])
                emotion_correlations.append({'feature': feature, 'correlation': corr})

        emotion_corr_df = pd.DataFrame(emotion_correlations)
        emotion_corr_df['abs_correlation'] = emotion_corr_df['correlation'].abs()
        emotion_corr_df = emotion_corr_df.sort_values('abs_correlation', ascending=False)

        # Calculate correlations with sentiments
        sentiment_correlations = []
        for feature in feature_cols:
            if feature in feature_df_analysis.columns:
                corr = feature_df_analysis[feature].corr(feature_df_analysis['sentiment_encoded'])
                sentiment_correlations.append({'feature': feature, 'correlation': corr})

        sentiment_corr_df = pd.DataFrame(sentiment_correlations)
        sentiment_corr_df['abs_correlation'] = sentiment_corr_df['correlation'].abs()
        sentiment_corr_df = sentiment_corr_df.sort_values('abs_correlation', ascending=False)

        # Print top correlations
        print("\nTop 15 features correlated with EMOTIONS:")
        print("=" * 50)
        for _, row in emotion_corr_df.head(15).iterrows():
            print(f"{row['feature']:30} : {row['correlation']:6.3f}")

        print("\nTop 15 features correlated with SENTIMENTS:")
        print("=" * 50)
        for _, row in sentiment_corr_df.head(15).iterrows():
            print(f"{row['feature']:30} : {row['correlation']:6.3f}")

        if save_plots:
            self._create_correlation_plots(emotion_corr_df, sentiment_corr_df, feature_df_analysis,
                                         le_emotion, le_sentiment)

        return emotion_corr_df, sentiment_corr_df

    def _create_correlation_plots(self, emotion_corr_df, sentiment_corr_df, feature_df_analysis,
                                 le_emotion, le_sentiment):
        """Create correlation visualization plots"""
        plt.style.use('default')

        # 1. Top feature correlations with emotions
        plt.figure(figsize=(12, 8))
        top_emotion_features = emotion_corr_df.head(20)

        colors = ['red' if x < 0 else 'blue' for x in top_emotion_features['correlation']]
        plt.barh(range(len(top_emotion_features)), top_emotion_features['correlation'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_emotion_features)), top_emotion_features['feature'], fontsize=8)
        plt.xlabel('Correlation with Emotion')
        plt.title('Top 20 Audio Features Correlated with Emotions')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('emotion_feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Top feature correlations with sentiments
        plt.figure(figsize=(12, 8))
        top_sentiment_features = sentiment_corr_df.head(20)

        colors = ['red' if x < 0 else 'green' for x in top_sentiment_features['correlation']]
        plt.barh(range(len(top_sentiment_features)), top_sentiment_features['correlation'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_sentiment_features)), top_sentiment_features['feature'], fontsize=8)
        plt.xlabel('Correlation with Sentiment')
        plt.title('Top 20 Audio Features Correlated with Sentiments')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('sentiment_feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Heatmap of top features vs emotions
        plt.figure(figsize=(15, 10))
        top_features = emotion_corr_df.head(30)['feature'].tolist()

        # Create emotion-feature matrix
        emotion_feature_matrix = []
        emotions = le_emotion.classes_

        for emotion in emotions:
            emotion_data = feature_df_analysis[feature_df_analysis['emotion'] == emotion]
            emotion_means = [emotion_data[feature].mean() if feature in emotion_data.columns else 0
                           for feature in top_features]
            emotion_feature_matrix.append(emotion_means)

        emotion_feature_matrix = np.array(emotion_feature_matrix)

        # Normalize by feature (z-score)
        from scipy.stats import zscore
        emotion_feature_matrix_norm = zscore(emotion_feature_matrix, axis=0)

        plt.imshow(emotion_feature_matrix_norm, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='Z-score normalized feature values')
        plt.yticks(range(len(emotions)), emotions)
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right', fontsize=8)
        plt.xlabel('Audio Features')
        plt.ylabel('Emotions')
        plt.title('Heatmap: Top Audio Features vs Emotions (Z-score normalized)')
        plt.tight_layout()
        plt.savefig('emotion_feature_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Feature distributions by emotion (for top 3 features)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        top_3_features = emotion_corr_df.head(4)['feature'].tolist()

        for i, feature in enumerate(top_3_features):
            ax = axes[i]

            # Create violin plots for each emotion
            emotions = feature_df_analysis['emotion'].unique()
            feature_data = []
            emotion_labels = []

            for emotion in emotions:
                emotion_data = feature_df_analysis[feature_df_analysis['emotion'] == emotion][feature]
                feature_data.append(emotion_data)
                emotion_labels.append(emotion)

            ax.boxplot(feature_data, labels=emotion_labels)
            ax.set_title(f'Distribution of {feature} by Emotion')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('feature_distributions_by_emotion.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nPlots saved:")
        print("- emotion_feature_correlations.png")
        print("- sentiment_feature_correlations.png")
        print("- emotion_feature_heatmap.png")
        print("- feature_distributions_by_emotion.png")

    def save_feature_dataframe(self, filename='meld_features_complete.csv'):
        """Save the complete feature dataframe"""
        if self.feature_df is None:
            raise ValueError("No feature data available. Run extract_all_features() first.")

        self.feature_df.to_csv(filename, index=False)
        print(f"Feature dataframe saved to {filename}")
        print(f"Shape: {self.feature_df.shape}")
        print(f"Columns: {list(self.feature_df.columns)}")

    def load_feature_dataframe(self, filename='meld_features_complete.csv'):
        """Load a previously saved feature dataframe"""
        if os.path.exists(filename):
            self.feature_df = pd.read_csv(filename)
            print(f"Loaded feature dataframe from {filename}")
            print(f"Shape: {self.feature_df.shape}")
            return self.feature_df
        else:
            print(f"File {filename} not found")
            return None

def main():
    """Main function to run feature extraction and correlation analysis"""
    print("MELD Audio Feature Extraction and Correlation Analysis")
    print("=" * 60)

    analyzer = MELDCorrelationAnalyzer()

    # Option 1: Extract features from scratch (can be slow)
    print("\n1. Extracting features from audio files...")
    print("Note: This will take a while. Use sample_size parameter for testing.")

    # For testing, use a smaller sample
    sample_size = 500  # Set to None to process all files

    feature_df = analyzer.extract_all_features(sample_size=sample_size)

    # Save the dataframe
    analyzer.save_feature_dataframe('meld_features_sample.csv')

    # Option 2: Or load previously extracted features
    # analyzer.load_feature_dataframe('meld_features_complete.csv')

    # Analyze correlations
    print("\n2. Analyzing correlations...")
    emotion_corr, sentiment_corr = analyzer.analyze_correlations(save_plots=True)

    # Additional analysis
    print("\n3. Dataset statistics:")
    print(f"Total samples: {len(feature_df)}")
    print(f"Emotion distribution:")
    print(feature_df['emotion'].value_counts())
    print(f"\nSentiment distribution:")
    print(feature_df['sentiment'].value_counts())
    print(f"\nAudio files found: {feature_df['audio_exists'].sum()}/{len(feature_df)}")

    return analyzer, emotion_corr, sentiment_corr

if __name__ == "__main__":
    analyzer, emotion_corr, sentiment_corr = main()
