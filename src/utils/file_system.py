import os
from loguru import logger

def ensure_dir_exists(dir_path: str):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
            raise