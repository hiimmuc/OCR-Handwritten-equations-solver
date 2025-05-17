# OCR Handwritten Equation Solver

This project is a web application that can detect and solve handwritten mathematical equations from images using computer vision and OCR techniques.

## Features

-   Detect multiple equations in an image
-   Recognize handwritten mathematical symbols and characters
-   Solve systems of equations
-   Display the results with proper mathematical formatting

## Architecture

This project follows CLEAN architecture principles to ensure separation of concerns and maintainability:

-   **Domain Layer**: Contains the core business entities and logic
-   **Application Layer**: Implements use cases and business rules
-   **Infrastructure Layer**: Provides concrete implementations of interfaces and adapters
-   **Web Layer**: Handles HTTP requests and responses using Flask

## Project Structure

```
ocr_equation_solver/
├── application/                # Use cases
│   ├── interfaces/             # Ports (interfaces)
│   │   ├── repositories.py
│   │   └── services.py
│   └── services/               # Use cases implementation
│       ├── character_recognition.py
│       ├── equation_detection.py
│       ├── equation_solver.py
│       └── ocr_solver.py
├── domain/                     # Entities
│   ├── models/
│   │   └── equation.py
│   └── exceptions.py
├── infrastructure/             # Adapters and frameworks
│   ├── config/
│   │   └── settings.py
│   ├── repositories/
│   │   └── model_repository.py
│   ├── ml/
│   │   ├── yolo_detector.py
│   │   ├── cnn_classifier.py
│   │   └── equation_solver.py
│   └── helpers/
│       └── image_processing.py
└── web/                       # UI Layer
    ├── app.py                 # Flask app factory
    ├── controllers/
    │   └── equation_controller.py
    ├── static/
    │   └── uploads/           # Upload directory
    └── templates/
        └── upload.html
```

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/ocr-equation-solver.git
    cd ocr-equation-solver
    ```

2. Download the required model weights:

    - YOLOv4 equation detection model: `weights/yolov4_training_2000_eq.weights`
    - YOLOv4 character detection model: `weights/yolov4_training_2000_char.weights`
    - OCR model for character classification: `weights/model_ocr.h5`

3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:

    ```
    python main.py
    ```

2. Open your browser and go to `http://127.0.0.1:5000/`

3. Upload an image containing handwritten equations

4. View the detection results and equation solutions

## Technical Details

-   **Equation Detection**: Uses YOLOv4 for detecting equations in images
-   **Character Detection**: Uses YOLOv4 for detecting characters within equations
-   **Character Classification**: Uses a CNN model to classify the detected characters
-   **Equation Solving**: Uses SymPy to solve the detected equations
-   **Web Interface**: Flask web application for user interaction

## License

This project is licensed under the MIT License - see the LICENSE file for details.
