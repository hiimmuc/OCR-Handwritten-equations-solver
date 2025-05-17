# OCR Handwritten Equation Solver

This project is a web application that can detect and solve handwritten mathematical equations from images using computer vision and OCR techniques.

## Features

-   Detect multiple equations in an image
-   Recognize handwritten mathematical symbols and characters
-   Solve systems of equations
-   Display the results with proper mathematical formatting

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
    - Download the weight file at https://drive.google.com/file/d/15NGlxNoybiPfOjey9ObV6O67HcaOgMlN/view?usp=sharing
    - Download darknet at https://github.com/AlexeyAB/darknet.git
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
  ![](https://github.com/hiimmuc/Handwritten-equation-solver/blob/master/img2.jpg)
  ![](https://github.com/hiimmuc/Handwritten-equation-solver/blob/master/img3.jpg)
  ![](https://youtu.be/qUWR9YQMzyE)

6. Customize or retrain model:
    - Follow this tutorial https://blog.roboflow.com/training-yolov4-on-a-custom-dataset/
    - In file yolov4_training.cfg :
    - Change classes = 80 to classes = 1 at lines 970 1058 1146
    - change filters = 255 to filters = 18 at line 1139 1051 963

## Technical Details

-   **Equation Detection**: Uses YOLOv4 for detecting equations in images
-   **Character Detection**: Uses YOLOv4 for detecting characters within equations
-   **Character Classification**: Uses a CNN model to classify the detected characters
-   **Equation Solving**: Uses SymPy to solve the detected equations
-   **Web Interface**: Flask web application for user interaction

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: hiimmuc1811@gmail.com
or my facebook:
https://www.facebook.com/phgnam1811/

## Contributors:
- Đặng Phương Nam - https://github.com/hiimmuc?tab=repositories
- Doãn Xuân Khang - https://github.com/khangdx1998
- Nguyễn Đức Thắng - https://www.facebook.com/ducthangbka8

## License

This project is licensed under the MIT License - see the LICENSE file for details.
