#!/usr/bin/env python3
"""
Quick start script for MELD emotion training
"""

import pandas as pd
import os
from meld_trainer import MELDTrainer

def check_dataset_structure():
    """Check if the MELD dataset is properly set up"""
    print("🔍 Checking MELD Dataset Structure...")

    # Check CSV files
    csv_files = ['train_sent_emo.csv', 'dev_sent_emo.csv', 'test_sent_emo.csv']
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"✅ {csv_file} found")
            # Quick peek at the structure
            df = pd.read_csv(csv_file)
            print(f"   - {len(df)} samples")
            print(f"   - Columns: {list(df.columns)}")
            if 'Emotion' in df.columns:
                print(f"   - Emotions: {df['Emotion'].unique()}")
        else:
            print(f"❌ {csv_file} not found")

    # Check audio folders
    audio_folders = ['train', 'dev_splits_complete', 'output_repeated_splits_test']
    for folder in audio_folders:
        if os.path.exists(folder):
            audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
            print(f"✅ {folder}/ found with {len(audio_files)} audio files")
        else:
            print(f"❌ {folder}/ not found")

    print("\n" + "="*50)

def start_training():
    """Start MELD emotion recognition training"""
    print("🎭 Starting MELD Emotion Recognition Training")
    print("=" * 60)

    # Check dataset first
    check_dataset_structure()

    # Initialize trainer with your folder structure
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
    print("\n📊 Loading dataset...")
    train_df, dev_df, test_df = trainer.load_and_prepare_data(
        balance_dataset=False  # Set to True if you want balanced classes
    )

    print(f"\n🎯 Dataset loaded successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Development samples: {len(dev_df)}")
    print(f"Test samples: {len(test_df)}")

    # Ask user if they want to proceed
    proceed = input("\n▶️ Start training? This may take 1-3 hours (y/n): ").lower().strip()

    if proceed == 'y':
        print("\n🚀 Training started...")

        # Train the model
        model, history = trainer.train_model(
            train_df, dev_df,
            epochs=50,  # Start with fewer epochs for testing
            batch_size=32,
            learning_rate=0.001
        )

        # Evaluate the model
        test_accuracy, report = trainer.evaluate_model(test_df)

        print(f"\n🎉 MELD Training Complete!")
        print(f"📊 Final Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"💾 Model saved as: best_meld_emotion_model.pth")

    else:
        print("👋 Training cancelled. You can start anytime by running this script again.")

def quick_test():
    """Quick test to see if everything works"""
    print("🧪 Quick Test Mode")
    print("=" * 30)

    # Load just a few samples to test
    try:
        train_df = pd.read_csv('train_sent_emo.csv').head(10)  # Just 10 samples
        dev_df = pd.read_csv('dev_sent_emo.csv').head(5)      # Just 5 samples

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

        print("✅ Trainer initialized successfully")
        print("✅ CSV files loaded successfully")
        print("🎉 Everything looks good! Ready for full training.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your file structure and try again.")

if __name__ == "__main__":
    print("🎤 MELD Emotion Recognition Setup")
    print("=" * 40)
    print("1. Quick test (check if everything works)")
    print("2. Full training (1-3 hours)")
    print("3. Check dataset structure only")

    choice = input("\nSelect option (1/2/3): ").strip()

    if choice == '1':
        quick_test()
    elif choice == '2':
        start_training()
    elif choice == '3':
        check_dataset_structure()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
