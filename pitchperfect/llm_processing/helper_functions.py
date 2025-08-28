def has_filler_words(self, text):
    """Check for filler words"""
    fillers = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'I mean', 'basically', 'actually']
    text_lower = text.lower()
    return any(filler in text_lower for filler in fillers)

def lacks_structure(self, text):
    """Check if text lacks proper structure"""
    sentences = text.split('.')
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

    # Too many short sentences or one very long sentence
    return avg_length < 5 or avg_length > 30

def post_process_text(self, text):
    """Clean up improved text"""
    # Remove any GPT artifacts
    text = text.strip()

    # Ensure proper capitalization
    sentences = text.split('. ')
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = '. '.join(sentences)

    # Remove double spaces
    text = ' '.join(text.split())

    return text

def calculate_volume_adjustment(self, acoustic_features):
    """Calculate volume adjustments needed"""
    energy = acoustic_features['energy']

    if energy < 0.02:
        return {'adjustment': 'increase', 'factor': 1.5}
    elif energy > 0.1:
        return {'adjustment': 'decrease', 'factor': 0.8}
    else:
        return {'adjustment': 'maintain', 'factor': 1.0}

def determine_target_emotion(self, sentiment, issues):
    """Determine target emotional tone"""
    if 'lacks_emotion' in issues['text_issues']:
        # Suggest more engaging emotion
        if sentiment['emotion'] == 'neutral':
            return 'confident_friendly'

    # Enhance existing emotion
    emotion_map = {
        'happy': 'enthusiastic',
        'sad': 'empathetic',
        'angry': 'assertive',
        'neutral': 'confident',
        'fear': 'calm_reassuring'
    }

    return emotion_map.get(sentiment['emotion'], 'confident')
