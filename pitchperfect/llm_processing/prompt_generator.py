def improve_text(self, text, issues, sentiment):
    """Generate improved text using OpenAI"""

    # Build context-aware prompt
    prompt = self.build_improvement_prompt(text, issues, sentiment)

    response = self.openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a speech coach helping improve spoken delivery."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    improved_text = response.choices[0].message.content

    # Post-process
    improved_text = self.post_process_text(improved_text)

    return improved_text

def build_improvement_prompt(self, text, issues, sentiment):
    """Create detailed prompt for text improvement"""

    prompt = f"""
    Improve this speech for better delivery:

    Original: "{text}"

    Current emotion: {sentiment['emotion']} (confidence: {sentiment['score']})

    Issues to address:
    """

    if 'filler_words' in issues['text_issues']:
        prompt += "\n- Remove filler words (um, uh, like, you know)"

    if 'poor_structure' in issues['text_issues']:
        prompt += "\n- Improve sentence structure and flow"

    if 'lacks_emotion' in issues['text_issues']:
        prompt += "\n- Add more engaging and emotive language"

    if issues['delivery_issues']:
        prompt += f"\n- Speaking issues: {', '.join(issues['delivery_issues'])}"

    prompt += """

    Requirements:
    1. Keep the core message identical
    2. Make it more impactful and confident
    3. Ensure natural spoken flow
    4. Add transition words where needed
    5. Keep length similar (±20%)

    Return ONLY the improved text, no explanations.
    """

    return prompt
