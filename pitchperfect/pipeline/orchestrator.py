# pitchperfect/pipeline/orchestrator.py
import logging
from typing import Dict, Optional
import time
from datetime import datetime

from pitchperfect.speech_to_text.transcriber import Transcriber
from pitchperfect.text_sentiment_analysis.analyzer import TextSentimentAnalyzer
from pitchperfect.tonal_analysis.analyzer import TonalAnalyzer
from pitchperfect.llm_processing.improvement_generator import ImprovementGenerator
from pitchperfect.text_to_speech.synthesis import Synthesizer

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Main pipeline orchestrator for the complete flow:
    Audio → Speech-to-Text → Sentiment Analysis
    Audio → Tonal Analysis
    Both → LLM Processing → Text-to-Speech
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize all modules
        logger.info("Initializing pipeline modules...")
        self.transcriber = Transcriber(config)
        self.sentiment_analyzer = TextSentimentAnalyzer(config)
        self.tonal_analyzer = TonalAnalyzer(config)
        self.improvement_generator = ImprovementGenerator(config)
        self.synthesizer = Synthesizer(config)

        logger.info("Pipeline orchestrator ready")

    def process(self,
                audio_path: str,
                voice_sample_path: Optional[str] = None,
                output_path: Optional[str] = None,
                user_preferences: Optional[Dict] = None) -> Dict:
        """
        Process audio through the complete pipeline

        Args:
            audio_path: Path to input audio
            voice_sample_path: Optional path for voice cloning
            output_path: Optional path to save output audio
            user_preferences: User preferences for improvement

        Returns:
            Complete processing results
        """
        start_time = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_audio': audio_path
        }

        try:
            # Step 1: Speech-to-Text
            logger.info("Step 1/5: Transcribing audio...")
            transcription = self.transcriber.transcribe(audio_path)
            results['transcription'] = transcription
            logger.info(f"Transcribed: {transcription['text'][:100]}...")

            # Step 2: Parallel analysis
            logger.info("Step 2/5: Analyzing sentiment...")
            sentiment = self.sentiment_analyzer.analyze(transcription['text'])
            results['sentiment'] = sentiment
            logger.info(f"Emotion: {sentiment['emotion']} ({sentiment['confidence']:.2f})")

            logger.info("Step 3/5: Analyzing tonal features...")
            tonal = self.tonal_analyzer.analyze(audio_path)
            results['tonal'] = tonal
            logger.info(f"Problems found: {tonal.get('acoustic_problems', [])}")

            # Step 3: LLM Processing
            logger.info("Step 4/5: Generating improvements...")
            improvements = self.improvement_generator.generate_improvements(
                text=transcription['text'],
                text_sentiment=sentiment,
                acoustic_features=tonal,
                user_preferences=user_preferences
            )
            results['improvements'] = {
                'improved_text': improvements.improved_text,
                'issues': improvements.issues,
                'feedback': improvements.feedback,
                'prosody_guide': improvements.prosody_guide,
                'ssml_markup': improvements.ssml_markup
            }
            logger.info(f"Improved text: {improvements.improved_text[:100]}...")

            # Step 4: Text-to-Speech
            logger.info("Step 5/5: Synthesizing improved speech...")

            try:
                # Clone voice if sample provided
                if voice_sample_path:
                    logger.info("Cloning voice from sample...")
                    synthesis_result = self.synthesizer.synthesize_with_clone(
                        text=improvements.improved_text,
                        clone_audio_path=voice_sample_path,
                        ssml=improvements.ssml_markup
                    )
                else:
                    synthesis_result = self.synthesizer.synthesize(
                        ssml=improvements.ssml_markup,
                        output_path=output_path
                    )

                results['synthesis'] = synthesis_result
                logger.info("Audio synthesis completed successfully")

            except Exception as e:
                logger.warning(f"Audio synthesis failed: {e}")
                results['synthesis'] = {
                    'error': str(e),
                    'status': 'failed'
                }

            # Calculate metrics
            processing_time = time.time() - start_time
            results['metrics'] = {
                'processing_time_seconds': processing_time,
                'original_word_count': len(transcription['text'].split()),
                'improved_word_count': len(improvements.improved_text.split()),
                'issues_found': improvements.feedback['issues_found'],
                'severity': improvements.feedback['severity']
            }

            logger.info(f"Pipeline completed in {processing_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
            raise

        return results
