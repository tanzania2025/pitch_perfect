def generate_ssml(self, text, prosody_guide):
    """Generate SSML markup for speech synthesis"""

    ssml = '<speak>'

    # Apply rate
    rate_percent = int(prosody_guide['rate']['multiplier'] * 100)
    ssml += f'<prosody rate="{rate_percent}%">'

    # Apply pitch
    pitch_adjustment = prosody_guide['pitch']['adjustment_factor']
    if pitch_adjustment != 1.0:
        pitch_percent = int(pitch_adjustment * 100 - 100)
        pitch_str = f"+{pitch_percent}%" if pitch_percent > 0 else f"{pitch_percent}%"
        ssml += f'<prosody pitch="{pitch_str}">'

    # Process text with emphasis and pauses
    words = text.split()
    emphasized_indices = [emp[0] for emp in prosody_guide['emphasis_words']]

    for i, word in enumerate(words):
        # Add pauses before certain words
        for pause in prosody_guide['pause_locations']:
            if pause.get('before') and pause['before'] in word.lower():
                ssml += '<break time="300ms"/>'

        # Add emphasis
        if i in emphasized_indices:
            ssml += f'<emphasis level="strong">{word}</emphasis> '
        else:
            ssml += f'{word} '

        # Add pauses after certain words
        for pause in prosody_guide['pause_locations']:
            if pause.get('after') and pause['after'] in word:
                if pause['type'] == 'short':
                    ssml += '<break time="200ms"/>'
                elif pause['type'] == 'medium':
                    ssml += '<break time="400ms"/>'

    # Close prosody tags
    if pitch_adjustment != 1.0:
        ssml += '</prosody>'
    ssml += '</prosody>'
    ssml += '</speak>'

    return ssml
