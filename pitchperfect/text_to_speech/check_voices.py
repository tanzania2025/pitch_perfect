#!/usr/bin/env python3
"""Script to check available ElevenLabs voices"""

import os

from elevenlabs import set_api_key, voices

# Set API key from environment
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    print("ELEVENLABS_API_KEY not set")
    exit(1)

set_api_key(api_key)

try:
    # Get available voices
    available_voices = voices()
    print(f"Found {len(available_voices)} voices:")
    print("-" * 50)

    for voice in available_voices:
        print(f"Name: {voice.name}")
        print(f"ID: {voice.voice_id}")
        print(f"Category: {voice.category}")
        print("-" * 50)

except Exception as e:
    print(f"Error fetching voices: {e}")
