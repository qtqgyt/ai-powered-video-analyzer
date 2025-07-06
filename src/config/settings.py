import os
import yaml
from pathlib import Path
from loguru import logger


class SummarizerPrompt:
    def __init__(self, prompt_path: str):
        self.prompt_path = Path(prompt_path)
        self.name = None
        self.min_words = None
        self.max_words = None
        self.prompt = None
        self._load_prompt()

    def _load_prompt(self):
        if not self.prompt_path.exists():
            logger.error(f"Prompt file not found: {self.prompt_path.resolve()}")
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read or parse prompt file '{self.prompt_path}': {e}")
            raise

        try:
            self.name = data.get("name", "default")
            self.min_words = data.get("min_words", 50)
            self.max_words = data.get("max_words", 100)
            self.prompt = data["prompt"]
        except Exception as e:
            logger.error(f"Prompt file '{self.prompt_path}' is missing required fields: {e}")
            raise

        if not self.prompt or not isinstance(self.prompt, str):
            logger.error(f"Prompt text is missing or invalid in '{self.prompt_path}'")
            raise ValueError(f"Prompt text is missing or invalid in '{self.prompt_path}'")

        logger.info(f"Loaded summarizer prompt '{self.name}' from {self.prompt_path}")

    def format_prompt(self):
        return self.prompt.format(min_words=self.min_words, max_words=self.max_words)


# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Model Paths ---
# These should point to where your downloaded models are stored,
# relative to the project or absolute paths.
# Example: os.path.join(BASE_DIR, "models", "yolov8n.pt")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")  # Default, or get from env
BLIP_MODEL_PATH = os.getenv("BLIP_MODEL_PATH", "path/to/your/blip/model")
WHISPER_MODEL_PATH = os.getenv(
    "WHISPER_MODEL_TYPE", "medium"
)  # e.g., "base", "small", "medium"
PANNS_MODEL_PATH = os.getenv(
    "PANNS_MODEL_PATH", os.path.join(BASE_DIR, "models", "cnn14.pth")
)

# --- Ollama Settings ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3")

# --- Other Settings ---
DEFAULT_TRANSCRIPTION_LANGUAGE = "en"
VIDEO_FRAME_EXTRACTION_INTERVAL = 5  # seconds

# YOLO settings
YOLO_MODEL_PATH = "yolo11x.pt"  # Use small model by default
FRAME_INTERVAL_SECONDS = 5  # Extract frame every 5 seconds

# Video summary settings for LLM

prompt_file = os.getenv("SUMMARIZER_PROMPT_FILE", "src/config/summarizer_prompts/default.yaml")
summarizer_prompt = SummarizerPrompt(prompt_file)
video_summary_prompt = summarizer_prompt.format_prompt()
