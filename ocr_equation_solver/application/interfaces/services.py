"""
Service interfaces
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ocr_equation_solver.domain.models.equation import Character, Equation


class ImageProcessorInterface(ABC):
    """Interface for processing images"""

    @abstractmethod
    def preprocess_image(self, image: Any) -> Any:
        """Preprocess an image for detection"""
        pass

    @abstractmethod
    def preprocess_character(self, image: Any) -> Any:
        """Preprocess a character image for classification"""
        pass

    @abstractmethod
    def rotate_image(self, image: Any, coordinates: List[List[int]]) -> Any:
        """Rotate an image based on detected text orientation"""
        pass


class EquationDetectorInterface(ABC):
    """Interface for detecting equations in an image"""

    @abstractmethod
    def detect_equations(self, image: Any) -> List[List[int]]:
        """Detect equations in an image and return their coordinates"""
        pass


class CharacterDetectorInterface(ABC):
    """Interface for detecting characters in an equation image"""

    @abstractmethod
    def detect_characters(self, equation_image: Any) -> List[List[int]]:
        """Detect characters in an equation image and return their coordinates"""
        pass


class CharacterClassifierInterface(ABC):
    """Interface for classifying detected characters"""

    @abstractmethod
    def classify_character(self, character_image: Any) -> Tuple[str, float]:
        """Classify a character image and return the predicted character and confidence"""
        pass


class EquationSolverInterface(ABC):
    """Interface for solving equations"""

    @abstractmethod
    def solve(self, equations: List[str]) -> List[str]:
        """Solve a list of equations and return the results"""
        pass
