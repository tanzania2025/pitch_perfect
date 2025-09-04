#!/usr/bin/env python3
"""Test script to verify voice_id is working properly"""

import requests
import json

# Test the voices endpoint
print("Testing /voices endpoint...")
response = requests.get("http://localhost:8080/voices")
if response.status_code == 200:
    voices_data = response.json()
    voices = voices_data.get("voices", [])
    print(f"Found {len(voices)} voices:")
    for voice in voices[:3]:  # Show first 3
        print(f"  - {voice['name']} (ID: {voice['voice_id']})")
        
    # Get Rachel's voice_id
    rachel_voice = next((v for v in voices if 'Rachel' in v['name']), None)
    if rachel_voice:
        print(f"\nFound Rachel voice: {rachel_voice['name']} with ID: {rachel_voice['voice_id']}")
        
        # Test process-audio with voice_id
        print("\nTesting /process-audio with voice_id...")
        
        # Create a simple test audio file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Just use an empty file for testing
            test_file = f.name
            
        with open(test_file, 'rb') as audio:
            files = {"audio_file": ("test.wav", audio, "audio/wav")}
            data = {
                "target_style": "professional",
                "improvement_focus": "all",
                "save_audio": True,
                "voice_id": rachel_voice['voice_id']  # Send the actual voice_id
            }
            
            print(f"Sending data: {data}")
            response = requests.post("http://localhost:8080/process-audio", files=files, data=data)
            print(f"Response status: {response.status_code}")
else:
    print(f"Failed to get voices: {response.status_code}")
    print(response.text)