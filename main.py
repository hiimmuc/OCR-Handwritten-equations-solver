"""
Main entry point for the OCR Equation Solver application
"""

from ocr_equation_solver.web.app import create_app

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    # Run the application
    app.run(debug=True)
