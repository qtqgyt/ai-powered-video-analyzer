import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from src.config import settings

logger = logging.getLogger(__name__)

def extract_audio(video_path):
    """
    Extracts audio from a video file and saves it as WAV.
    
    Args:
        video_path (str): Path to the input video file
        
    Returns:
        str: Path to the extracted audio file, or None if extraction fails
    """
    try:
        # Convert to Path object and resolve to absolute path
        video_path = Path(video_path).resolve()
        audio_path = video_path.with_suffix('.wav')
        
        # Load the video using properly quoted path
        video = VideoFileClip(str(video_path))
        
        # Extract audio and save
        video.audio.write_audiofile(str(audio_path))
        
        # Close the video to free resources
        video.close()
        
        logger.info(f"Successfully extracted audio to {audio_path}")
        return str(audio_path)
        
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {str(e)}")
        return None

def extract_frames_for_analysis(video_path, interval_seconds=5):
    """
    Placeholder for frame extraction function.
    To be implemented.
    """
    raise NotImplementedError("Frame extraction not yet implemented")

@dataclass
class Frame:
    """Represents a video frame with its timestamp"""
    timestamp: float
    image: np.ndarray

def extract_frames_for_analysis(video_path: str, interval_seconds: float = None) -> List[Frame]:
    """
    Extracts frames from a video at specified intervals.
    
    Args:
        video_path (str): Path to the video file
        interval_seconds (float, optional): Override for interval between frames.
            Defaults to VIDEO_FRAME_EXTRACTION_INTERVAL from settings.
        
    Returns:
        List[Frame]: List of Frame objects containing timestamps and images
    """
    try:
        # Use settings value if no override provided
        interval = interval_seconds or settings.VIDEO_FRAME_EXTRACTION_INTERVAL
        
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frames = []
        frame_count = 0

        logger.info(f"Extracting frames every {interval} seconds (every {frame_interval} frames)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append(Frame(timestamp=timestamp, image=frame))
                logger.debug(f"Extracted frame at timestamp {timestamp:.2f}s")

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames for analysis")
        return frames

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []