"""
Model repository implementation
"""
import time
from typing import Any

import cv2
from tensorflow.keras.models import load_model

from ocr_equation_solver.application.interfaces.repositories import (
    ModelRepositoryInterface,
)


class ModelRepository(ModelRepositoryInterface):
    """Repository for loading and managing ML models"""

    def load_model(self, model_path: str) -> Any:
        """
        Load a model from a file path

        Args:
            model_path: Path to the model file

        Returns:
            The loaded model
        """
        print(f"[INFO] Loading model from {model_path}...")
        start_time = time.time()

        if model_path.endswith(".h5"):
            # Load Keras model
            model = load_model(model_path)
        else:
            # Assume it's a darknet model that needs to be loaded with OpenCV
            raise ValueError(f"Unsupported model format: {model_path}")

        print(f"[INFO] Model loaded in {time.time() - start_time:.2f}s")
        return model


class YoloModelRepository(ModelRepositoryInterface):
    """Repository for loading and managing YOLO models"""

    def __init__(self):
        self.models = {}

    def load_model(self, model_path: str, config_path: str = None) -> Any:
        """
        Load a YOLO model using OpenCV

        Args:
            model_path: Path to the weights file
            config_path: Path to the config file

        Returns:
            The loaded model
        """
        if config_path is None:
            raise ValueError("Config path is required for YOLO models")

        key = f"{model_path}_{config_path}"
        if key in self.models:
            return self.models[key]

        print(f"[INFO] Loading YOLO model from {model_path}...")
        start_time = time.time()

        model = cv2.dnn.readNetFromDarknet(config_path, model_path)
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.models[key] = model

        print(f"[INFO] YOLO model loaded in {time.time() - start_time:.2f}s")
        return model
