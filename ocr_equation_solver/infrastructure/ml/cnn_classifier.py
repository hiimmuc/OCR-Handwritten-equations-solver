"""
Character classifier implementation
"""
from typing import Any, List, Tuple

import numpy as np

from ocr_equation_solver.application.interfaces.services import (
    CharacterClassifierInterface,
)

# Character classes
CHARACTER_NAMES = [
    "+",
    "-",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "=",
    "a",
    "b",
    "c",
    "d",
    "x",
    "y",
    "z",
]


class CNNCharacterClassifier(CharacterClassifierInterface):
    """Character classifier using a CNN model"""

    def __init__(self, model, confidence_threshold: float = 0.5):
        """
        Initialize the character classifier

        Args:
            model: The trained CNN model
            confidence_threshold: Threshold for classification confidence
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.character_names = CHARACTER_NAMES

    def classify_character(self, character_image: Any) -> Tuple[str, float]:
        """
        Classify a character image

        Args:
            character_image: The preprocessed character image

        Returns:
            Tuple of (predicted character, confidence)
        """
        # Ensure image is in the right shape for the model
        if character_image.ndim == 3 and character_image.shape[0] == 1:
            # If already batch-shaped with one image
            input_image = character_image
        else:
            # Add batch dimension if not already present
            input_image = np.expand_dims(character_image, 0)

        # Make sure we have the right number of channels for the model
        if input_image.shape[-1] != 1:
            input_image = np.expand_dims(input_image, -1)

        # Predict
        prediction = self.model.predict(input_image)

        # Get the class with the highest probability
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Get the character name
        character = (
            self.character_names[class_idx] if class_idx < len(self.character_names) else "?"
        )

        return character, confidence
