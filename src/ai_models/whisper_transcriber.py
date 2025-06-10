import logging
from pathlib import Path
import tempfile
import shutil
import os

import whisper, whisper.audio

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_path: str = "base"): # model_path can be size like "base", "small", etc.
        self.model_name = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            # You might want to raise a custom exception or handle this gracefully
            raise

    def transcribe(self, audio_path: str, language: str = None) -> str:
        if not self.model:
            logger.error("Whisper model is not loaded. Cannot transcribe.")
            return "Error: Transcription model not loaded."

        temp_path = None
        try:
            # Convert to Path object and resolve it
            audio_file = Path(audio_path).resolve()
            
            # Add detailed debug logging
            logger.debug("=== Whisper Audio File Debug Info ===")
            logger.debug(f"Input path: {audio_path}")
            logger.debug(f"Resolved path: {audio_file}")
            logger.debug(f"File exists: {audio_file.exists()}")
            
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            # Convert to string with forward slashes only when passing to whisper
            audio_path_str = str(audio_file).replace("\\", "/")
            logger.debug(f"Normalized path: {audio_path_str}")
            
            options = {}
            if language:
                options["language"] = language

            # Use the string version for whisper functions
            audio = whisper.audio.load_audio(audio_path_str)
            logger.debug(f"Audio loaded successfully, shape: {audio.shape}")


            # Use the string version for transcribe
            logger.debug("Starting transcription...")
            result = self.model.transcribe(audio_path_str, **options)
            if result is None:
                raise RuntimeError("Transcription failed - no result returned")
            logger.debug("Transcription completed successfully.")
            return result["text"]

        except Exception as e:
            logger.error(f"Error during transcription of {audio_path}: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error("Exception details:", exc_info=True)
            return f"Error during transcription: {e}"

# Similar handler classes would be created for YOLO, BLIP, PANNs, and the LLM.
# Each would load its specific model and provide a clean interface.
# For example, llm_summarizer.py would interface with Ollama.