"""
Web controller for the equation solver
"""
import os
from typing import Any, Dict

import cv2
from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from ocr_equation_solver.application.services.ocr_solver import OCREquationSolverService
from ocr_equation_solver.infrastructure.config.settings import AppSettings


class EquationController:
    """Controller for equation solving endpoints"""

    def __init__(self, ocr_service: OCREquationSolverService, settings: AppSettings):
        """
        Initialize the controller

        Args:
            ocr_service: Service for OCR equation solving
            settings: Application settings
        """
        self.ocr_service = ocr_service
        self.settings = settings

        # Create blueprint
        self.blueprint = Blueprint("equation", __name__)

        # Register routes
        self.blueprint.route("/", methods=["GET"])(self.home)
        self.blueprint.route("/", methods=["POST"])(self.upload_image)
        self.blueprint.route("/images/<filename>")(self.get_image)

    def allowed_file(self, filename: str) -> bool:
        """
        Check if a file has an allowed extension

        Args:
            filename: Name of the file

        Returns:
            True if the file is allowed, False otherwise
        """
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.settings.allowed_extensions
        )

    def home(self):
        """
        Render the home page

        Returns:
            Rendered template
        """
        return render_template("upload.html")

    def get_image(self, filename: str):
        """
        Get an image from the upload folder

        Args:
            filename: Name of the image file

        Returns:
            Image file
        """
        image_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        return send_file(image_path)

    def upload_image(self):
        """
        Handle image upload and equation solving

        Returns:
            Rendered template with results
        """
        if request.method != "POST":
            return redirect("/")

        # Check if file exists in request
        if "file" not in request.files:
            return redirect("/")

        file = request.files["file"]

        # Check if file is empty
        if not file or file.filename == "":
            return redirect("/")

        # Check if file is allowed
        if not self.allowed_file(file.filename):
            return redirect("/")

        # Save the file
        filename = secure_filename(file.filename)
        full_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(full_path)

        # Solve equations
        try:
            soe_image, result, list_text_equation = self.ocr_service.solve_from_image(
                full_path, input_type="path"
            )

            # Save the cropped image
            cv2.imwrite(
                os.path.join(current_app.config["UPLOAD_FOLDER"], f"cropped_{filename}"), soe_image
            )

            # Add a border to the cropped image
            soe_image = cv2.copyMakeBorder(
                soe_image, 0, 0, 10, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Get URL for the cropped image
            image_cropped_url = url_for("equation.get_image", filename=f"cropped_{filename}")

            # Process the original image
            original_image = cv2.imread(full_path)
            original_image = cv2.copyMakeBorder(
                original_image, 0, 0, 0, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            cv2.imwrite(
                os.path.join(current_app.config["UPLOAD_FOLDER"], f"original_{filename}"),
                original_image,
            )
            original_image_url = url_for("equation.get_image", filename=f"original_{filename}")

            # Render the template with results
            return render_template(
                "upload.html",
                original_image=original_image_url,
                cropped_image=image_cropped_url,
                result=result,
                text=list_text_equation,
            )

        except Exception as e:
            # Handle errors
            return render_template(
                "upload.html",
                error=f"Error processing image: {str(e)}",
            )
