"""
Application settings
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AppSettings:
    """Application settings"""

    # Model paths
    equation_weight_path: str
    character_weight_path: str
    config_path: str
    ocr_model_path: str

    # Flask settings
    upload_folder: str
    secret_key: str
    max_content_length: int
    allowed_extensions: set

    # Detection settings
    equation_confidence_threshold: float = 0.5
    equation_nms_threshold: float = 0.4
    character_confidence_threshold: float = 0.5
    character_nms_threshold: float = 0.3
    classifier_confidence_threshold: float = 0.5

    @classmethod
    def from_environment(cls) -> "AppSettings":
        """Create settings from environment variables"""
        return cls(
            equation_weight_path=os.environ.get(
                "EQUATION_WEIGHT_PATH", "weights/yolov4_training_2000_eq.weights"
            ),
            character_weight_path=os.environ.get(
                "CHARACTER_WEIGHT_PATH", "weights/yolov4_training_2000_char.weights"
            ),
            config_path=os.environ.get("CONFIG_PATH", "backup/cfg/yolov4_training.cfg"),
            ocr_model_path=os.environ.get("OCR_MODEL_PATH", "weights/model_ocr.h5"),
            upload_folder=os.environ.get("UPLOAD_FOLDER", "static/uploads"),
            secret_key=os.environ.get("SECRET_KEY", "secret key"),
            max_content_length=int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)),
            allowed_extensions=set(["png", "jpg", "jpeg", "gif"]),
            equation_confidence_threshold=float(
                os.environ.get("EQUATION_CONFIDENCE_THRESHOLD", 0.5)
            ),
            equation_nms_threshold=float(os.environ.get("EQUATION_NMS_THRESHOLD", 0.4)),
            character_confidence_threshold=float(
                os.environ.get("CHARACTER_CONFIDENCE_THRESHOLD", 0.5)
            ),
            character_nms_threshold=float(os.environ.get("CHARACTER_NMS_THRESHOLD", 0.3)),
            classifier_confidence_threshold=float(
                os.environ.get("CLASSIFIER_CONFIDENCE_THRESHOLD", 0.5)
            ),
        )


# Default settings instance
settings = AppSettings.from_environment()
