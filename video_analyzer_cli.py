import logging
from src.cli.arguments import parse_arguments
from src.cli.handlers import handle_video_analysis
from src.utils.logging_setup import setup_logging
from src.config import settings # To potentially load and display some configs

def main():
    args = parse_arguments()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("AI Powered Video Analyzer - CLI Mode")
    logger.debug(f"Arguments: {args}")
    # Potentially print or verify loaded model paths from settings
    # logger.info(f"Using YOLO model from: {settings.YOLO_MODEL_PATH}")

    try:
        handle_video_analysis(args)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=args.verbose)
        # exc_info=True will add traceback if verbose

if __name__ == "__main__":
    main()