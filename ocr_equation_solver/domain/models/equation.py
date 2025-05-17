"""
Entity representing an equation in the system
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Character:
    """Represents a character in an equation"""

    value: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def coordinates(self):
        """Get the character bounding box coordinates"""
        return [self.x, self.y, self.width, self.height]


@dataclass
class Equation:
    """Represents a mathematical equation"""

    text: str
    characters: List[Character]
    x: int
    y: int
    width: int
    height: int

    @property
    def coordinates(self):
        """Get the equation bounding box coordinates"""
        return [self.x, self.y, self.width, self.height]
