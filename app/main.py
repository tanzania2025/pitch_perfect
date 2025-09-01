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

from dotenv import load_dotenv
load_dotenv(".env")

    
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

    def process_speech(self, uploaded_audio, recorded_audio, voice_sample, target_style, focus, save_audio):
        """Process speech through pipeline with support for both uploaded and recorded audio"""

        # Determine which audio source to use
        audio_file = None
        audio_source = ""

        if recorded_audio is not None:
            audio_file = recorded_audio
            audio_source = "recorded"
            logger.info("Using recorded audio")
        elif uploaded_audio is not None:
            audio_file = uploaded_audio
            audio_source = "uploaded"
            logger.info("Using uploaded audio")

        if not audio_file:
            return (
                "Please either upload an audio file or record your voice using the microphone",
                "",
                "",
                "",
                None,  # Audio output
                ""     # Download path
            )

        try:
            # Generate unique filename for this session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"improved_{audio_source}_{timestamp}"

            # Temporary output path
            temp_output_path = self.temp_dir / f"{session_id}.mp3"

            # Set preferences
            preferences = {
                'target_style': target_style,
                'improvement_focus': focus
            }

            # Process through pipeline
            logger.info(f"Processing {audio_source} audio: {audio_file}")
            results = self.pipeline.process(
                audio_path=audio_file,
                voice_sample_path=voice_sample,
                output_path=str(temp_output_path) if save_audio else None,
                user_preferences=preferences
            )

            # Format outputs
            original = f"**Original Text ({audio_source}):**\n\n{results['transcription']['text']}"
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
            metrics_text += f"**Audio Source:** {audio_source.capitalize()}\n"
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

    def clear_inputs(self):
        """Clear all input fields"""
        return None, None, None

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
    """Create enhanced Gradio interface with voice recording"""
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
    .input-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .record-button {
        margin: 10px 0;
    }
    .divider {
        margin: 15px 0;
        text-align: center;
        color: #666;
        font-weight: bold;
    }
    """

    with gr.Blocks(title="Pitch Perfect - Speech Improvement", css=custom_css) as interface:
        gr.Markdown("""
        # 🎙️ Pitch Perfect - Speech Improvement System

        Transform your speech with AI-powered analysis and improvements. Record or upload your speech to receive:
        - 📝 Text transcription and improvements
        - 🎭 Emotion and sentiment analysis
        - 🎵 Tonal quality assessment
        - 🔊 Re-synthesized speech with improvements
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Audio Input Options")

                # Recording Section
                with gr.Group(elem_classes="input-section"):
                    gr.Markdown("#### Option 1: Record Your Voice")
                    gr.Markdown("⚠️ **Browser Requirements:**")
                    gr.Markdown("- Allow microphone permissions when prompted")
                    gr.Markdown("- Use Chrome/Firefox/Safari for best compatibility")
                    gr.Markdown("- Ensure you're on HTTPS or localhost")

                    recorded_audio = gr.Audio(
                        label="🔴 Click to Record (may take a moment to initialize)",
                        sources=["microphone"],
                        type="filepath",
                        elem_classes="record-button",
                        show_download_button=False,
                        streaming=False
                    )

                gr.Markdown("**OR**", elem_classes="divider")

                # Upload Section
                with gr.Group(elem_classes="input-section"):
                    gr.Markdown("#### Option 2: Upload Audio File")
                    uploaded_audio = gr.Audio(
                        label="📎 Upload Speech Recording (WAV/MP3)",
                        sources=["upload"],
                        type="filepath",
                        elem_classes="input-audio"
                    )

                # Voice Sample Section
                gr.Markdown("### 🎭 Voice Cloning (Optional)")
                voice_sample = gr.Audio(
                    label="🎤 Voice Sample for Cloning",
                    sources=["upload"],
                    type="filepath",
                    elem_classes="voice-sample"
                )
                gr.Markdown("*Upload a sample of the target voice for speech synthesis*", elem_classes="help-text")

                # Settings
                gr.Markdown("### ⚙️ Processing Settings")
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

                # Action Buttons
                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Analyze & Improve Speech",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear All",
                        variant="secondary",
                        size="sm"
                    )

                gr.Markdown("""
                ### 📋 Quick Instructions:
                1. **Record** your voice or **upload** an audio file
                2. Optionally add a voice sample for cloning
                3. Choose your target style and focus area
                4. Click **'Analyze & Improve Speech'**
                5. Listen to the improved version and review feedback

                **Supported formats:** WAV, MP3, M4A, FLAC

                ### 🔧 Microphone Troubleshooting:
                - **Permission denied:** Click the 🔒 icon in browser address bar to allow microphone
                - **No microphone found:** Check system audio settings and browser permissions
                - **Still not working?** Use the upload option with a voice recording app
                - **Mobile users:** Recording works best in mobile browsers like Chrome/Safari
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

        # Event handlers
        process_btn.click(
            fn=app.process_speech,
            inputs=[
                uploaded_audio,
                recorded_audio,
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

        clear_btn.click(
            fn=app.clear_inputs,
            inputs=None,
            outputs=[uploaded_audio, recorded_audio, voice_sample]
        )

        # Clean up old temp files when interface loads
        interface.load(
            fn=app.cleanup_temp_files,
            inputs=None,
            outputs=cleanup_output
        )

        # Footer
        gr.Markdown("""
        ---
        ### 🚀 Features:
        - 🎯 **Speech-to-Text**: Accurate transcription using Whisper
        - 🎭 **Sentiment Analysis**: Emotion and mood detection
        - 🎵 **Tonal Analysis**: Pitch, pace, and energy assessment
        - ✨ **AI Improvements**: GPT-powered text enhancement
        - 🔊 **Voice Synthesis**: Text-to-speech with voice cloning
        - 🎤 **Live Recording**: Record directly in the browser
        - 💾 **Export Options**: Save improved audio for later use

        *Note: Processing may take 15-30 seconds depending on audio length. For best recording quality, use a quiet environment.*
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
