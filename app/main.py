# app/main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from config import load_config
from pitchperfect.pipeline.orchestrator import PipelineOrchestrator
import logging
import tempfile
import os
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechImprovementApp:
    """Main application class"""

    def __init__(self):
        self.config = load_config()
        self.pipeline = PipelineOrchestrator(self.config)

        # Create output directory for saved audio
        self.output_dir = Path("outputs/generated_audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Temporary directory for session audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "pitch_perfect"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Speech Improvement App initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Temp directory: {self.temp_dir}")

    def process_speech(self, audio_file, voice_sample, target_style, focus, save_audio):
        """Process speech through pipeline"""
        if not audio_file:
            return (
                "Please upload an audio file",
                "",
                "",
                "",
                None,  # Audio output
                ""     # Download path
            )

        try:
            # Generate unique filename for this session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"improved_{timestamp}"

            # Temporary output path
            temp_output_path = self.temp_dir / f"{session_id}.mp3"

            # Set preferences
            preferences = {
                'target_style': target_style,
                'improvement_focus': focus
            }

            # Process through pipeline
            logger.info(f"Processing audio: {audio_file}")
            results = self.pipeline.process(
                audio_path=audio_file,
                voice_sample_path=voice_sample,
                output_path=str(temp_output_path) if save_audio else None,
                user_preferences=preferences
            )

            # Format outputs
            original = f"**Original Text:**\n\n{results['transcription']['text']}"
            improved = f"**Improved Text:**\n\n{results['improvements']['improved_text']}"

            # Format feedback
            feedback = results['improvements']['feedback']
            feedback_text = f"### {feedback['summary']}\n\n"

            if feedback['key_improvements']:
                feedback_text += "**Improvements Made:**\n"
                for imp in feedback['key_improvements']:
                    feedback_text += f"• {imp}\n"

            if feedback['speaking_tips']:
                feedback_text += "\n**Speaking Tips:**\n"
                for tip in feedback['speaking_tips']:
                    feedback_text += f"• {tip}\n"

            # Format metrics and issues
            metrics = results['metrics']
            issues = results['improvements']['issues']

            metrics_text = f"### Analysis Results\n\n"
            metrics_text += f"**Processing Time:** {metrics['processing_time_seconds']:.1f} seconds\n"
            metrics_text += f"**Word Count:** {metrics['original_word_count']} → {metrics['improved_word_count']}\n"
            metrics_text += f"**Issues Found:** {metrics['issues_found']}\n"
            metrics_text += f"**Severity Level:** {metrics['severity'].upper()}\n\n"

            if issues['text_issues']:
                metrics_text += f"**Text Issues:** {', '.join(issues['text_issues'])}\n"
            if issues['delivery_issues']:
                metrics_text += f"**Delivery Issues:** {', '.join(issues['delivery_issues'])}\n"

            # Handle audio output
            audio_output_path = None
            download_path = ""

            if save_audio and 'synthesis' in results and results['synthesis'].get('audio'):
                # Save to permanent location if requested
                permanent_output_path = self.output_dir / f"{session_id}.mp3"

                # Write audio bytes to file
                audio_bytes = results['synthesis']['audio']
                with open(permanent_output_path, 'wb') as f:
                    f.write(audio_bytes)

                # Also save to temp for immediate playback
                with open(temp_output_path, 'wb') as f:
                    f.write(audio_bytes)

                audio_output_path = str(temp_output_path)
                download_path = f"✅ Audio saved to: outputs/generated_audio/{session_id}.mp3"

                logger.info(f"Audio saved to {permanent_output_path}")

            elif 'synthesis' in results and results['synthesis'].get('audio'):
                # Just save to temp for playback without permanent storage
                audio_bytes = results['synthesis']['audio']
                with open(temp_output_path, 'wb') as f:
                    f.write(audio_bytes)
                audio_output_path = str(temp_output_path)
                download_path = "⚠️ Audio not saved permanently (enable 'Save Audio' to keep)"

            # Add emotion info
            sentiment = results['sentiment']
            emotion_text = f"\n\n### Emotion Analysis\n"
            emotion_text += f"**Primary Emotion:** {sentiment['emotion'].capitalize()} ({sentiment['confidence']:.1%})\n"
            emotion_text += f"**Sentiment:** {sentiment['sentiment'].capitalize()}\n"
            emotion_text += f"**Valence:** {sentiment['valence']:.2f} (-1 to 1)\n"
            emotion_text += f"**Arousal:** {sentiment['arousal']:.2f} (0 to 1)"

            metrics_text += emotion_text

            return (
                original,
                improved,
                feedback_text,
                metrics_text,
                audio_output_path,
                download_path
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            error_msg = f"❌ Error: {str(e)}"
            return error_msg, "", "", "", None, ""

    def cleanup_temp_files(self):
        """Clean up temporary files older than 1 hour"""
        try:
            import time
            current_time = time.time()

            for file_path in self.temp_dir.glob("*.mp3"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 3600:  # 1 hour
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up temp file: {file_path}")
                    except:
                        pass
            return "Cleanup completed"
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return f"Cleanup failed: {e}"

def create_interface():
    """Create enhanced Gradio interface"""
    app = SpeechImprovementApp()

    # Custom CSS for better styling
    custom_css = """
    .output-markdown {
        max-height: 400px;
        overflow-y: auto;
    }
    .audio-output {
        margin-top: 20px;
    }
    """

    with gr.Blocks(title="Pitch Perfect - Speech Improvement", css=custom_css) as interface:
        gr.Markdown("""
        # 🎙️ Pitch Perfect - Speech Improvement System

        Transform your speech with AI-powered analysis and improvements. Upload your recording to receive:
        - 📝 Text transcription and improvements
        - 🎭 Emotion and sentiment analysis
        - 🎵 Tonal quality assessment
        - 🔊 Re-synthesized speech with improvements
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Settings")

                audio_input = gr.Audio(
                    label="📎 Upload Speech Recording",
                    type="filepath",
                    elem_classes="input-audio"
                )

                voice_sample = gr.Audio(
                    label="🎤 Voice Sample for Cloning (optional)",
                    type="filepath",
                    elem_classes="voice-sample"
                )

                with gr.Row():
                    style = gr.Dropdown(
                        ["professional", "casual", "academic", "motivational"],
                        value="professional",
                        label="🎯 Target Style"
                    )
                    focus = gr.Dropdown(
                        ["all", "clarity", "confidence", "engagement"],
                        value="all",
                        label="🔍 Focus Area"
                    )

                save_audio = gr.Checkbox(
                    label="💾 Save Generated Audio",
                    value=True,
                    info="Save the improved audio to outputs folder"
                )

                process_btn = gr.Button(
                    "🚀 Analyze & Improve",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                ### Instructions:
                1. Upload a speech recording (WAV/MP3)
                2. Optionally add voice sample for cloning
                3. Select target style and focus area
                4. Click 'Analyze & Improve'
                5. Listen to the improved version below
                """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Text Analysis")
                original_output = gr.Markdown(
                    label="Original Text",
                    elem_classes="output-markdown"
                )
                improved_output = gr.Markdown(
                    label="Improved Text",
                    elem_classes="output-markdown"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results & Feedback")
                metrics_output = gr.Markdown(
                    label="Analysis Metrics",
                    elem_classes="output-markdown"
                )
                feedback_output = gr.Markdown(
                    label="Improvement Feedback",
                    elem_classes="output-markdown"
                )

        # Audio Output Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔊 Improved Speech Output")
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="filepath",
                    elem_classes="audio-output",
                    autoplay=False,
                    show_download_button=True
                )
                download_status = gr.Markdown(
                    label="Save Status",
                    elem_classes="download-status"
                )

        # Hidden output for cleanup function
        cleanup_output = gr.Textbox(visible=False)

        # Examples Section (commented out if examples don't exist)
        # with gr.Row():
        #     gr.Examples(
        #         examples=[
        #             ["examples/sample_speech.wav", None, "professional", "all", True],
        #             ["examples/presentation.wav", None, "academic", "clarity", True],
        #             ["examples/casual_talk.wav", None, "casual", "confidence", False],
        #         ],
        #         inputs=[audio_input, voice_sample, style, focus, save_audio],
        #         label="Try these examples:"
        #     )

        # Process button click
        process_btn.click(
            fn=app.process_speech,
            inputs=[
                audio_input,
                voice_sample,
                style,
                focus,
                save_audio
            ],
            outputs=[
                original_output,
                improved_output,
                feedback_output,
                metrics_output,
                audio_output,
                download_status
            ]
        )

        # Clean up old temp files when interface loads (inside the Blocks context)
        interface.load(
            fn=app.cleanup_temp_files,
            inputs=None,
            outputs=cleanup_output
        )

        # Footer
        gr.Markdown("""
        ---
        ### Features:
        - 🎯 **Speech-to-Text**: Accurate transcription using Whisper
        - 🎭 **Sentiment Analysis**: Emotion and mood detection
        - 🎵 **Tonal Analysis**: Pitch, pace, and energy assessment
        - ✨ **AI Improvements**: GPT-powered text enhancement
        - 🔊 **Voice Synthesis**: Text-to-speech with voice cloning
        - 💾 **Export Options**: Save improved audio for later use

        *Note: Processing may take 15-30 seconds depending on audio length.*
        """)

    return interface

if __name__ == "__main__":
    # Create necessary directories
    Path("outputs/generated_audio").mkdir(parents=True, exist_ok=True)
    Path("examples").mkdir(parents=True, exist_ok=True)

    # Launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        max_threads=10
    )
