"""
Image processing helper functions
"""
import cv2
import numpy as np

from ocr_equation_solver.application.interfaces.services import ImageProcessorInterface


class OpenCVImageProcessor(ImageProcessorInterface):
    """Image processor implementation using OpenCV"""

    def preprocess_image(self, image):
        """Preprocess an entire image for equation detection"""
        # Basic preprocessing for the entire image
        # Resizing, normalization, etc. could be added here if needed
        return image

    def preprocess_character(self, image):
        """Preprocess a character image for OCR"""
        # Convert to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply median blur to reduce noise
        image_blur = cv2.medianBlur(image_gray, 3)
        # Apply adaptive thresholding
        _, thresh = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert image (black background, white text becomes white background, black text)
        image_bw = cv2.bitwise_not(thresh)

        # Padding
        h, w = image_gray.shape
        pad_size = int(abs(h - w) / 2)
        pad_extra = abs(h - w) % 2

        if h > w:
            # Width is smaller than height, add padding to the sides
            image_padding = cv2.copyMakeBorder(
                image_bw,
                0,
                0,
                pad_size + pad_extra,
                pad_size,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            # Add a small border around the entire image
            image_padding = cv2.copyMakeBorder(
                image_padding, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        elif w > h:
            # Height is smaller than width, add padding to top and bottom
            image_padding = cv2.copyMakeBorder(
                image_bw,
                pad_size + pad_extra,
                pad_size,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            # Add a small border around the entire image
            image_padding = cv2.copyMakeBorder(
                image_padding, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            # Already square, just use as is
            image_padding = image_bw

        # Resize to 28x28 for classification
        image_resized = cv2.resize(image_padding, (28, 28))

        # Dilate to enhance features
        kernel = np.ones((3, 3), np.uint8)
        image_dilate = cv2.dilate(image_resized, kernel)

        # Expand dimensions to match model input shape (1, 28, 28, 1)
        image_result = np.expand_dims(image_dilate, 0)

        return image_result

    def rotate_image(self, image, coordinates):
        """Rotate an image based on detected text orientation"""
        if not coordinates:
            return image  # No equations detected, return original image

        # Extract coordinates for cropping
        np_coor = np.array(coordinates)
        x_new = np.min(np_coor[:, 0])
        y_new = np.min(np_coor[:, 1])
        w_new = np.max(np.array([x + w for x, y, w, h in coordinates])) - x_new
        h_new = (
            np.max(np_coor[:, 1]) + np_coor[np.argmax(np_coor[:, 1])][3] - np.min(np_coor[:, 1])
        )

        # Crop the image to the equations region
        soe_cropped = image[y_new : y_new + h_new, x_new : x_new + w_new]

        # Convert to grayscale and invert for text detection
        gray = cv2.cvtColor(soe_cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Create a black image of the same size as the original
        image_black = np.zeros((image.shape[0], image.shape[1]))
        # Place the thresholded cropped area onto the black image
        image_black[y_new : y_new + h_new, x_new : x_new + w_new] = thresh

        # Find coordinates of non-zero points in the image
        coords = np.column_stack(np.where(image_black > 0))
        if len(coords) == 0:
            return image  # No text found, return original image

        # Calculate the minimum area rectangle that contains all non-zero points
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If the angle is small, don't rotate
        if abs(angle) < 1:
            return image

        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated
