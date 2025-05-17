"""
Flask application factory
"""
import os

from flask import Flask

from ocr_equation_solver.application.services.character_recognition import (
    CharacterRecognitionService,
)
from ocr_equation_solver.application.services.equation_detection import (
    EquationDetectionService,
)
from ocr_equation_solver.application.services.equation_solver import (
    EquationSolverService,
)
from ocr_equation_solver.application.services.ocr_solver import OCREquationSolverService
from ocr_equation_solver.infrastructure.config.settings import AppSettings
from ocr_equation_solver.infrastructure.helpers.image_processing import (
    OpenCVImageProcessor,
)
from ocr_equation_solver.infrastructure.ml.cnn_classifier import CNNCharacterClassifier
from ocr_equation_solver.infrastructure.ml.equation_solver import SymPyEquationSolver
from ocr_equation_solver.infrastructure.ml.yolo_detector import (
    YoloCharacterDetector,
    YoloEquationDetector,
)
from ocr_equation_solver.infrastructure.repositories.model_repository import (
    ModelRepository,
    YoloModelRepository,
)
from ocr_equation_solver.web.controllers.equation_controller import EquationController


def create_app(test_config=None):
    """
    Create and configure the Flask application

    Args:
        test_config: Test configuration

    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__, template_folder="templates")

    # Load settings
    settings = AppSettings.from_environment()

    # Configure the app
    app.secret_key = settings.secret_key
    app.config["UPLOAD_FOLDER"] = settings.upload_folder
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Create services
    # Load repositories
    model_repo = ModelRepository()
    yolo_repo = YoloModelRepository()

    # Load models
    ocr_model = model_repo.load_model(settings.ocr_model_path)
    yolo_equation_model = yolo_repo.load_model(settings.equation_weight_path, settings.config_path)
    yolo_character_model = yolo_repo.load_model(
        settings.character_weight_path, settings.config_path
    )

    # Create infrastructure components
    image_processor = OpenCVImageProcessor()
    equation_detector = YoloEquationDetector(
        yolo_equation_model,
        confidence_threshold=settings.equation_confidence_threshold,
        nms_threshold=settings.equation_nms_threshold,
    )
    character_detector = YoloCharacterDetector(
        yolo_character_model,
        confidence_threshold=settings.character_confidence_threshold,
        nms_threshold=settings.character_nms_threshold,
    )
    character_classifier = CNNCharacterClassifier(
        ocr_model, confidence_threshold=settings.classifier_confidence_threshold
    )
    equation_solver = SymPyEquationSolver()

    # Create application services
    equation_detection_service = EquationDetectionService(
        equation_detector=equation_detector, image_processor=image_processor
    )
    character_recognition_service = CharacterRecognitionService(
        character_detector=character_detector,
        character_classifier=character_classifier,
        image_processor=image_processor,
    )
    equation_solver_service = EquationSolverService()

    # Create orchestrator
    ocr_service = OCREquationSolverService(
        equation_detection_service=equation_detection_service,
        character_recognition_service=character_recognition_service,
        equation_solver_service=equation_solver_service,
        image_processor=image_processor,
    )

    # Create controller
    equation_controller = EquationController(ocr_service, settings)

    # Register blueprints
    app.register_blueprint(equation_controller.blueprint)

    return app
