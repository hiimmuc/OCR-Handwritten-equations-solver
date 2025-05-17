"""
Repository interfaces
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelRepositoryInterface(ABC):
    """Interface for accessing machine learning models"""

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load a model from a file path"""
        pass
