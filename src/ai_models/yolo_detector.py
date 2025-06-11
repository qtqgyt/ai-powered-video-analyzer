import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize YOLO detector with specified model.
        
        Args:
            model_path (str): Path to YOLO model or model name
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model"""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            List[Dict[str, Any]]: List of detections with coordinates and labels
        """
        try:
            results = self.model(frame, verbose=False)[0]
            detections = []

            for box in results.boxes:
                detection = {
                    "bbox": box.xyxy[0].cpu().numpy(),  # Convert to numpy array
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                    "class_name": results.names[int(box.cls)]
                }
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()