def identify_issues(self, text, sentiment, acoustic_features):
    """Identify speaking and text issues"""

    issues = {
        'text_issues': [],
        'delivery_issues': [],
        'severity': 'low'  # low, medium, high
    }

    # Text-based issues
    if self.has_filler_words(text):
        issues['text_issues'].append('filler_words')

    if self.lacks_structure(text):
        issues['text_issues'].append('poor_structure')

    if sentiment['emotion'] == 'neutral' and sentiment['score'] > 0.8:
        issues['text_issues'].append('lacks_emotion')

    # Acoustic issues
    if acoustic_features['monotone_score'] > 0.6:
        issues['delivery_issues'].append('monotone')

    if acoustic_features['speaking_rate'] > 4.5:
        issues['delivery_issues'].append('too_fast')
    elif acoustic_features['speaking_rate'] < 2.5:
        issues['delivery_issues'].append('too_slow')

    if acoustic_features['energy'] < 0.02:
        issues['delivery_issues'].append('low_energy')

    if acoustic_features['pause_ratio'] < 0.1:
        issues['delivery_issues'].append('needs_pauses')

    # Set severity
    total_issues = len(issues['text_issues']) + len(issues['delivery_issues'])
    if total_issues >= 4:
        issues['severity'] = 'high'
    elif total_issues >= 2:
        issues['severity'] = 'medium'

    return issues
