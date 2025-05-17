"""
Character recognition service
"""
from typing import Any, List, Tuple

from ocr_equation_solver.application.interfaces.services import (
    CharacterClassifierInterface,
    CharacterDetectorInterface,
    ImageProcessorInterface,
)
from ocr_equation_solver.domain.models.equation import Character


class CharacterRecognitionService:
    """Service for recognizing characters in equations"""

    def __init__(
        self,
        character_detector: CharacterDetectorInterface,
        character_classifier: CharacterClassifierInterface,
        image_processor: ImageProcessorInterface,
    ):
        self.character_detector = character_detector
        self.character_classifier = character_classifier
        self.image_processor = image_processor

    def recognize_characters_in_equation(self, equation_image: Any) -> Tuple[str, List[Character]]:
        """
        Recognize characters in an equation image

        Returns:
            Tuple of (equation text, list of characters)
        """
        # Detect characters in the equation image
        character_coordinates = self.character_detector.detect_characters(equation_image)

        # Extract character images
        character_images = [
            equation_image[y : y + h, x : x + w] for x, y, w, h in character_coordinates
        ]

        characters = []
        equation_text = ""

        # Process each character image
        for i, character_image in enumerate(character_images):
            # Preprocess the character image
            preprocessed_image = self.image_processor.preprocess_character(character_image)

            # Classify the character
            predicted_char, confidence = self.character_classifier.classify_character(
                preprocessed_image
            )

            # If confidence is high enough, add to the equation text
            if confidence >= 0.5:
                x, y, w, h = character_coordinates[i]
                character = Character(
                    value=predicted_char, x=x, y=y, width=w, height=h, confidence=confidence
                )
                characters.append(character)
                equation_text += predicted_char

        return equation_text, characters
