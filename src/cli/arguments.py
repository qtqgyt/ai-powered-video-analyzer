import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Powered Video Analyzer CLI")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file to analyze."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results (report, annotated video). Default: analysis_results"
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save a text report of the analysis."
    )
    parser.add_argument(
        "--save_annotated_video",
        action="store_true",
        help="Save an annotated version of the video (if applicable)."
    )
    parser.add_argument(
        "--transcription_language",
        type=str,
        default="en", # Or autodetect if Whisper supports it well
        help="Language for speech transcription (e.g., 'en', 'es', 'fa'). Default: en"
    )
    parser.add_argument(
        "--summarization_model",
        type=str,
        # default="ollama_default_model", # Example, fetch from config
        help="Specify the LLM model to use for summarization (via Ollama)."
    )
    # Add arguments to enable/disable specific analyses
    parser.add_argument(
        "--skip_object_detection",
        action="store_true",
        help="Skip object detection (YOLO)."
    )
    parser.add_argument(
        "--skip_scene_description",
        action="store_true",
        help="Skip scene description (BLIP)."
    )
    # ... other skip flags for Whisper, PANNs, Summarization

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging."
    )
    return parser.parse_args()