"""
Helper script to run the application with a single command
"""

# Run the OCR Equation Solver application
from ocr_equation_solver.web.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
