"""
Equation detection service
"""
from typing import Any, List

from ocr_equation_solver.application.interfaces.services import (
    EquationDetectorInterface,
    ImageProcessorInterface,
)


class EquationDetectionService:
    """Service for detecting equations in images"""

    def __init__(
        self,
        equation_detector: EquationDetectorInterface,
        image_processor: ImageProcessorInterface,
    ):
        self.equation_detector = equation_detector
        self.image_processor = image_processor

    def detect_equations(self, image: Any) -> Tuple[Any, List[List[int]]]:
        """
        Detect equations in an image

        Returns:
            Tuple of (processed image, list of equation coordinates)
        """
        # Preprocess the image
        processed_image = self.image_processor.preprocess_image(image)

        # Detect equations in the image
        equation_coordinates = self.equation_detector.detect_equations(processed_image)

        # Rotate the image based on the detected equations
        rotated_image = self.image_processor.rotate_image(processed_image, equation_coordinates)

        # Detect equations again in the rotated image
        equation_coordinates = self.equation_detector.detect_equations(rotated_image)

        # Sort equations by y-coordinate (top to bottom)
        equation_coordinates.sort(key=lambda coords: coords[1])

        return rotated_image, equation_coordinates
