import logging
import sys

def setup_logging(level=logging.INFO):
    # Basic configuration example
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
            # You could add a FileHandler here to log to a file
            # logging.FileHandler("analyzer.log")
        ]
    )
    # Quieten overly verbose libraries if necessary
    # logging.getLogger("some_noisy_library").setLevel(logging.WARNING)