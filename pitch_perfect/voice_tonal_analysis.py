#!/usr/bin/env python3
"""
Voice Tonal Analysis System
Analyzes audio files to extract voice features like pitch, volume, speed, gaps, etc.
"""


# Whisper - Pre-trained by OpenAI on 680,000 hours of multilingual audio
# TextBlob - Pre-trained sentiment analysis
# Librosa - Mathematical audio feature extraction (no AI needed)

import librosa
import numpy as np
import wave
import whisper
from textblob import TextBlob
import matplotlib.pyplot as plt
import json
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class VoiceTonalAnalyzer:
    def __init__(self, whisper_model_size="base"):
        """
        Initialize the Voice Tonal Analyzer

        Args:
            whisper_model_size (str): Size of Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        print("Voice Tonal Analyzer initialized successfully!")

    def load_audio(self, audio_file_path, target_sr=16000):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=target_sr)
            print(f"Loaded audio: {len(y)/sr:.2f} seconds, {sr}Hz")
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return result
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return {"text": "", "segments": []}

    def extract_pitch_features(self, y, sr):
        """Extract pitch-related features"""
        # Extract fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)

        # Get pitch values over time
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) == 0:
            pitch_values = [0]

        # Calculate pitch statistics
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_range = max(pitch_values) - min(pitch_values) if len(pitch_values) > 1 else 0
        pitch_median = np.median(pitch_values)

        return {
            'pitch_mean_hz': float(pitch_mean),
            'pitch_std_hz': float(pitch_std),
            'pitch_range_hz': float(pitch_range),
            'pitch_median_hz': float(pitch_median),
            'pitch_coefficient_variation': float(pitch_std / pitch_mean) if pitch_mean > 0 else 0,
            'pitch_values': pitch_values[:100]  # First 100 values for plotting
        }

    def extract_volume_features(self, y, sr):
        """Extract volume/energy related features"""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Volume statistics
        volume_mean = np.mean(rms)
        volume_std = np.std(rms)
        volume_max = np.max(rms)
        volume_min = np.min(rms)

        # Dynamic range
        dynamic_range = volume_max - volume_min

        return {
            'volume_mean': float(volume_mean),
            'volume_std': float(volume_std),
            'volume_max': float(volume_max),
            'volume_min': float(volume_min),
            'dynamic_range': float(dynamic_range),
            'volume_mean_db': float(np.mean(rms_db)),
            'volume_std_db': float(np.std(rms_db)),
            'rms_values': rms.tolist()[:100]  # First 100 values for plotting
        }

    def extract_tempo_rhythm_features(self, y, sr):
        """Extract tempo and rhythm features"""
        try:
            # Tempo estimation
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

            # Beat timing
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Rhythm regularity (coefficient of variation of inter-beat intervals)
            if len(beat_times) > 1:
                inter_beat_intervals = np.diff(beat_times)
                rhythm_regularity = np.std(inter_beat_intervals) / np.mean(inter_beat_intervals)
            else:
                rhythm_regularity = 0

            return {
                'tempo_bpm': float(tempo),
                'num_beats': int(len(beat_frames)),
                'rhythm_regularity': float(rhythm_regularity),
                'beat_times': beat_times.tolist()[:50]  # First 50 beats
            }
        except Exception as e:
            print(f"Error extracting tempo features: {e}")
            return {
                'tempo_bpm': 0,
                'num_beats': 0,
                'rhythm_regularity': 0,
                'beat_times': []
            }

    def detect_pauses_gaps(self, y, sr, silence_threshold=-40, min_silence_duration=0.5):
        """Detect pauses and gaps in speech"""
        # Convert to dB
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Detect silence frames
        silence_frames = rms_db < silence_threshold

        # Convert frames to time
        frame_duration = len(y) / sr / len(rms_db)

        # Find silence segments
        silence_segments = []
        in_silence = False
        silence_start = 0

        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                # Start of silence
                silence_start = i * frame_duration
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                silence_duration = (i * frame_duration) - silence_start
                if silence_duration >= min_silence_duration:
                    silence_segments.append({
                        'start': silence_start,
                        'end': i * frame_duration,
                        'duration': silence_duration
                    })
                in_silence = False

        # Statistics
        total_silence_duration = sum(seg['duration'] for seg in silence_segments)
        audio_duration = len(y) / sr
        silence_ratio = total_silence_duration / audio_duration

        return {
            'num_pauses': len(silence_segments),
            'total_pause_duration': float(total_silence_duration),
            'average_pause_duration': float(total_silence_duration / len(silence_segments)) if silence_segments else 0,
            'silence_ratio': float(silence_ratio),
            'pause_segments': silence_segments[:20]  # First 20 pauses
        }

    def calculate_speaking_rate(self, transcript, audio_duration):
        """Calculate speaking rate metrics"""
        text = transcript.get('text', '')
        words = text.split()

        # Word-based metrics
        words_per_minute = len(words) / (audio_duration / 60) if audio_duration > 0 else 0
        words_per_second = len(words) / audio_duration if audio_duration > 0 else 0

        # Character-based metrics
        chars_per_minute = len(text) / (audio_duration / 60) if audio_duration > 0 else 0

        # Syllable estimation (rough)
        syllables = sum(max(1, len([c for c in word if c.lower() in 'aeiou'])) for word in words)
        syllables_per_minute = syllables / (audio_duration / 60) if audio_duration > 0 else 0

        return {
            'words_per_minute': float(words_per_minute),
            'words_per_second': float(words_per_second),
            'chars_per_minute': float(chars_per_minute),
            'syllables_per_minute': float(syllables_per_minute),
            'total_words': len(words),
            'total_characters': len(text),
            'estimated_syllables': syllables
        }

    def extract_spectral_features(self, y, sr):
        """Extract spectral features"""
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth))
        }

    def analyze_text_features(self, transcript):
        """Analyze text-based features"""
        text = transcript.get('text', '')

        # Basic sentiment analysis
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity  # -1 to 1
        sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1

        # Filler words detection
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'sort of', 'kind of']
        filler_count = sum(text.lower().count(word) for word in filler_words)

        # Text complexity
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words) if words else 0

        return {
            'sentiment_polarity': float(sentiment_polarity),
            'sentiment_subjectivity': float(sentiment_subjectivity),
            'filler_word_count': int(filler_count),
            'avg_word_length': float(avg_word_length),
            'vocabulary_richness': float(vocabulary_richness),
            'unique_words': int(unique_words)
        }

    def analyze_audio(self, audio_file_path):
        """
        Complete audio analysis

        Args:
            audio_file_path (str): Path to audio file

        Returns:
            dict: Comprehensive analysis results
        """
        print(f"Analyzing audio file: {audio_file_path}")

        # Load audio
        y, sr = self.load_audio(audio_file_path)
        if y is None:
            return None

        audio_duration = len(y) / sr

        # Transcribe audio
        print("Transcribing audio...")
        transcript = self.transcribe_audio(audio_file_path)

        # Extract features
        print("Extracting pitch features...")
        pitch_features = self.extract_pitch_features(y, sr)

        print("Extracting volume features...")
        volume_features = self.extract_volume_features(y, sr)

        print("Extracting tempo and rhythm features...")
        tempo_features = self.extract_tempo_rhythm_features(y, sr)

        print("Detecting pauses and gaps...")
        pause_features = self.detect_pauses_gaps(y, sr)

        print("Calculating speaking rate...")
        speaking_rate = self.calculate_speaking_rate(transcript, audio_duration)

        print("Extracting spectral features...")
        spectral_features = self.extract_spectral_features(y, sr)

        print("Analyzing text features...")
        text_features = self.analyze_text_features(transcript)

        # Compile results
        results = {
            'audio_info': {
                'duration_seconds': float(audio_duration),
                'sample_rate': int(sr),
                'file_path': audio_file_path
            },
            'transcript': transcript['text'],
            'pitch_features': pitch_features,
            'volume_features': volume_features,
            'tempo_rhythm_features': tempo_features,
            'pause_gap_features': pause_features,
            'speaking_rate': speaking_rate,
            'spectral_features': spectral_features,
            'text_features': text_features
        }

        print("Analysis complete!")
        return results

    def generate_summary(self, analysis_results):
        """Generate a human-readable summary of the analysis"""
        if not analysis_results:
            return "No analysis results available."

        summary = []
        summary.append("=== VOICE TONAL ANALYSIS SUMMARY ===\n")

        # Basic info
        duration = analysis_results['audio_info']['duration_seconds']
        summary.append(f"Audio Duration: {duration:.2f} seconds\n")

        # Speaking rate
        wpm = analysis_results['speaking_rate']['words_per_minute']
        summary.append(f"Speaking Rate: {wpm:.1f} words per minute")
        if wpm < 120:
            summary.append(" (Slow)")
        elif wpm > 180:
            summary.append(" (Fast)")
        else:
            summary.append(" (Normal)")
        summary.append("\n")

        # Pitch characteristics
        pitch_mean = analysis_results['pitch_features']['pitch_mean_hz']
        pitch_range = analysis_results['pitch_features']['pitch_range_hz']
        summary.append(f"Average Pitch: {pitch_mean:.1f} Hz")
        summary.append(f"Pitch Range: {pitch_range:.1f} Hz")
        if pitch_range < 50:
            summary.append(" (Monotone)")
        elif pitch_range > 200:
            summary.append(" (Very expressive)")
        summary.append("\n")

        # Volume characteristics
        volume_db = analysis_results['volume_features']['volume_mean_db']
        dynamic_range = analysis_results['volume_features']['dynamic_range']
        summary.append(f"Average Volume: {volume_db:.1f} dB")
        summary.append(f"Dynamic Range: {dynamic_range:.3f}\n")

        # Pauses
        num_pauses = analysis_results['pause_gap_features']['num_pauses']
        avg_pause = analysis_results['pause_gap_features']['average_pause_duration']
        silence_ratio = analysis_results['pause_gap_features']['silence_ratio']
        summary.append(f"Number of Pauses: {num_pauses}")
        summary.append(f"Average Pause Duration: {avg_pause:.2f} seconds")
        summary.append(f"Silence Ratio: {silence_ratio:.2%}\n")

        # Text analysis
        sentiment = analysis_results['text_features']['sentiment_polarity']
        fillers = analysis_results['text_features']['filler_word_count']
        summary.append(f"Sentiment: {sentiment:.2f} (-1=negative, +1=positive)")
        summary.append(f"Filler Words: {fillers}\n")

        return '\n'.join(summary)

    def save_results(self, results, output_file):
        """Save analysis results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Example usage of the Voice Tonal Analyzer"""
    import sys

    # Get audio file from command line argument or ask user
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Enter the path to your audio file: ").strip()
        if not audio_file:
            print("No audio file provided. Exiting.")
            return

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found.")
        print("Please check the file path and try again.")
        return

    print(f"Processing audio file: {audio_file}")

    # Initialize analyzer
    print("Loading Whisper model (this may take a moment)...")
    analyzer = VoiceTonalAnalyzer(whisper_model_size="base")

    try:
        # Analyze audio
        print("Starting analysis...")
        results = analyzer.analyze_audio(audio_file)

        if results:
            # Print summary
            print("\n" + "="*60)
            print(analyzer.generate_summary(results))
            print("="*60)

            # Save detailed results
            output_filename = f"analysis_{os.path.splitext(os.path.basename(audio_file))[0]}.json"
            analyzer.save_results(results, output_filename)

            # Print key metrics in a nice format
            print("\n🎯 KEY METRICS:")
            print("-" * 40)
            print(f"📝 Transcript: \"{results['transcript'][:100]}{'...' if len(results['transcript']) > 100 else ''}\"")
            print(f"⏱️  Speaking Rate: {results['speaking_rate']['words_per_minute']:.1f} WPM")
            print(f"🎵 Average Pitch: {results['pitch_features']['pitch_mean_hz']:.1f} Hz")
            print(f"🔊 Average Volume: {results['volume_features']['volume_mean_db']:.1f} dB")
            print(f"⏸️  Number of Pauses: {results['pause_gap_features']['num_pauses']}")
            print(f"😊 Sentiment Score: {results['text_features']['sentiment_polarity']:.2f} (-1 to +1)")
            print(f"📊 Filler Words: {results['text_features']['filler_word_count']}")

            # Interpretations
            print(f"\n💡 INTERPRETATIONS:")
            print("-" * 40)

            # Speaking rate interpretation
            wpm = results['speaking_rate']['words_per_minute']
            if wpm < 120:
                print("🐌 Speaking pace: Slow - consider speaking faster")
            elif wpm > 180:
                print("🏃 Speaking pace: Fast - consider slowing down")
            else:
                print("✅ Speaking pace: Normal")

            # Pitch interpretation
            pitch_range = results['pitch_features']['pitch_range_hz']
            if pitch_range < 50:
                print("😴 Voice expression: Monotone - try varying your pitch more")
            elif pitch_range > 200:
                print("🎭 Voice expression: Very expressive")
            else:
                print("✅ Voice expression: Good variation")

            # Sentiment interpretation
            sentiment = results['text_features']['sentiment_polarity']
            if sentiment > 0.1:
                print("😊 Overall tone: Positive")
            elif sentiment < -0.1:
                print("😞 Overall tone: Negative")
            else:
                print("😐 Overall tone: Neutral")

            print(f"\n📄 Detailed results saved to: {output_filename}")

        else:
            print("❌ Analysis failed. Please check your audio file and try again.")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Make sure your audio file is in a supported format (WAV, MP3, FLAC, M4A, etc.)")


if __name__ == "__main__":
    import os
    main()
