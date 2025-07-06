import sys
from loguru import logger

def setup_logging(level="INFO"):
    """Setup loguru logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler for persistent logs
    logger.add(
        "video_analyzer.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    # Intercept standard library logging
    logger.add(
        lambda msg: logger.opt(depth=6, exception=msg.record["exception"]).log(
            msg.record["level"].name, msg.record["message"]
        ),
        level=0,
        format="{message}"
    )
    
    return logger