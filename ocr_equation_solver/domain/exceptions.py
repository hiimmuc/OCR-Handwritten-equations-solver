"""
Domain exceptions
"""


class DomainException(Exception):
    """Base exception for all domain exceptions"""

    pass


class InvalidEquationError(DomainException):
    """Exception raised when an equation is invalid"""

    pass


class OCRError(DomainException):
    """Exception raised when OCR process fails"""

    pass


class ModelError(DomainException):
    """Exception raised when a machine learning model fails"""

    pass


class SolverError(DomainException):
    """Exception raised when an equation cannot be solved"""

    pass
