"""
OCR equation solver orchestrator
"""
from typing import Any, List, Tuple

import cv2
import latex2mathml.converter

from ocr_equation_solver.application.services.character_recognition import (
    CharacterRecognitionService,
)
from ocr_equation_solver.application.services.equation_detection import (
    EquationDetectionService,
)
from ocr_equation_solver.application.services.equation_solver import (
    EquationSolverService,
)
from ocr_equation_solver.domain.models.equation import Equation


class OCREquationSolverService:
    """Orchestrator for the OCR equation solving process"""

    def __init__(
        self,
        equation_detection_service: EquationDetectionService,
        character_recognition_service: CharacterRecognitionService,
        equation_solver_service: EquationSolverService,
        image_processor,
    ):
        """
        Initialize the service

        Args:
            equation_detection_service: Service for detecting equations
            character_recognition_service: Service for recognizing characters
            equation_solver_service: Service for solving equations
            image_processor: Image processor for preprocessing
        """
        self.equation_detection_service = equation_detection_service
        self.character_recognition_service = character_recognition_service
        self.equation_solver_service = equation_solver_service
        self.image_processor = image_processor

    def solve_from_image(
        self, image_input: Any, input_type: str = "image"
    ) -> Tuple[Any, List[str], List[str]]:
        """
        Solve equations from an image

        Args:
            image_input: The input image or path to the image
            input_type: Type of input, either "image" or "path"

        Returns:
            Tuple of (processed image, list of solution strings, list of equation strings)
        """
        # Load image if a path is provided
        if input_type == "path":
            image = cv2.imread(image_input)
        elif input_type == "image":
            image = image_input
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        # Detect equations in the image
        processed_image, equation_coordinates = self.equation_detection_service.detect_equations(
            image
        )

        # Extract equation images
        equation_images = [
            processed_image[y : y + h, x : x + w] for x, y, w, h in equation_coordinates
        ]

        # Recognize characters in each equation
        equation_texts = []
        equations = []

        for i, eq_image in enumerate(equation_images):
            # Recognize characters
            (
                equation_text,
                characters,
            ) = self.character_recognition_service.recognize_characters_in_equation(eq_image)
            equation_texts.append(equation_text)

            # Create equation object
            x, y, w, h = equation_coordinates[i]
            equations.append(
                Equation(text=equation_text, characters=characters, x=x, y=y, width=w, height=h)
            )

            print(f"Equation {i+1}: {equation_text}")

        # Solve the equations
        solutions = self.equation_solver_service.solve_equations(equation_texts)

        # Crop the image for display
        if equation_coordinates:
            # Create a crop that includes all detected equations
            cropped_image = self.image_processor.text_skew(
                processed_image, equation_coordinates, True
            )
        else:
            cropped_image = processed_image

        # Convert the equations to MathML for display
        display_equations = self.format_equations_for_display(equation_texts)

        # Convert the solutions to MathML for display
        display_solutions = [latex2mathml.converter.convert(sol) for sol in solutions]

        return cropped_image, display_solutions, display_equations

    def format_equations_for_display(self, equations: List[str]) -> List[str]:
        """
        Format equations for display using MathML

        Args:
            equations: List of equation strings

        Returns:
            List of MathML equation strings
        """
        display_equations = []

        for eq in equations:
            formatted = ""
            for i, char in enumerate(eq):
                if char in ["+", "="]:
                    formatted += " " + char + " "
                elif char == "-":
                    if i == 0 or formatted[len(formatted) - 2 : len(formatted)] == "= ":
                        formatted += char
                    else:
                        formatted += " " + char + " "
                elif char.isnumeric():
                    if i == 0:
                        formatted += char
                    elif formatted[-1].isalpha():
                        formatted += "^" + char
                    else:
                        formatted += char
                else:
                    formatted += char

            # Convert to MathML
            mathml = latex2mathml.converter.convert(formatted)
            display_equations.append(mathml)

        return display_equations
