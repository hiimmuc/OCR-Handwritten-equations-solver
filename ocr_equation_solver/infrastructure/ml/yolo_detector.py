"""
YOLO detector implementation
"""
import time
from typing import Any, List

import cv2
import numpy as np

from ocr_equation_solver.application.interfaces.services import (
    CharacterDetectorInterface,
    EquationDetectorInterface,
)


class YoloDetector:
    """Base class for YOLO detection"""

    def __init__(self, net, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize the YOLO detector

        Args:
            net: The YOLO neural network
            confidence_threshold: Threshold for object detection confidence
            nms_threshold: Threshold for non-maximum suppression
        """
        self.net = net
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.num_obj = 0

    def detect(self, image: Any) -> List[List[int]]:
        """
        Detect objects in an image

        Args:
            image: The input image

        Returns:
            List of coordinates [x, y, width, height]
        """
        h, w, _ = image.shape

        # Get layers
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set input to network
        self.net.setInput(blob)

        # Forward pass
        layer_outputs = self.net.forward(output_layers)

        # Process outputs
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Scale the bounding box coordinates to the original image
                    center_x, center_y, width, height = list(
                        map(int, detection[0:4] * [w, h, w, h])
                    )

                    # Get top-left corner coordinates
                    top_left_x = int(center_x - (width / 2))
                    top_left_y = int(center_y - (height / 2))

                    # Add to lists
                    boxes.append([top_left_x, top_left_y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        self.num_obj = len(indices)

        # Create output list
        output_coordinates = []
        crop_scale = 0.05  # Scale for enlarging detection boxes

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Slightly enlarge the box
                x = abs(int(x - crop_scale * w))
                y = abs(int(y - crop_scale * h))
                w = abs(int((1 + 2 * crop_scale) * w))
                h = abs(int((1 + 2 * crop_scale) * h))

                output_coordinates.append([x, y, w, h])

        # Sort by x-coordinate (left to right)
        output_coordinates = sorted(output_coordinates, key=lambda x: x[0])

        return output_coordinates


class YoloEquationDetector(YoloDetector, EquationDetectorInterface):
    """YOLO detector for equations"""

    def detect_equations(self, image: Any) -> List[List[int]]:
        """
        Detect equations in an image

        Args:
            image: The input image

        Returns:
            List of equation coordinates [x, y, width, height]
        """
        return self.detect(image)


class YoloCharacterDetector(YoloDetector, CharacterDetectorInterface):
    """YOLO detector for characters"""

    def detect_characters(self, equation_image: Any) -> List[List[int]]:
        """
        Detect characters in an equation image

        Args:
            equation_image: The input equation image

        Returns:
            List of character coordinates [x, y, width, height]
        """
        return self.detect(equation_image)
