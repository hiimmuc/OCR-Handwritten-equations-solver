"""
Equation solver implementation
"""
from typing import List

import sympy
from sympy.parsing.sympy_parser import parse_expr

from ocr_equation_solver.application.interfaces.services import EquationSolverInterface


class SymPyEquationSolver(EquationSolverInterface):
    """Equation solver using SymPy"""

    def solve(self, equations: List[str]) -> List[str]:
        """
        Solve a list of equations

        Args:
            equations: List of equation strings

        Returns:
            List of solution strings
        """
        if not equations:
            return ["No equations provided"]

        processed_equations = []

        for equation in equations:
            try:
                left_side, right_side = equation.split("=")
                # Add a "+" sign if the first term doesn't have a sign
                if left_side and left_side[0] not in ["+", "-"]:
                    left_side = "+" + left_side
                if right_side and right_side[0] not in ["+", "-"]:
                    right_side = "+" + right_side

                # Find positions of all "+" and "-" signs
                left_positions = [i for i, char in enumerate(left_side) if char in ["+", "-"]]
                left_positions.append(len(left_side))
                right_positions = [i for i, char in enumerate(right_side) if char in ["+", "-"]]
                right_positions.append(len(right_side))

                # Process each term
                processed_equation = ""

                # Process left side
                for j in range(len(left_positions) - 1):
                    term = left_side[left_positions[j] : left_positions[j + 1]]
                    processed_equation += self._process_term(term)

                # Process right side (negate all terms)
                for j in range(len(right_positions) - 1):
                    term = right_side[right_positions[j] : right_positions[j + 1]]
                    # Negate the term
                    if term[0] == "+":
                        processed_equation += "-" + self._process_term(term[1:])
                    else:
                        processed_equation += "+" + self._process_term(term[1:])

                processed_equations.append(processed_equation)

            except Exception as e:
                print(f"Error processing equation {equation}: {e}")
                # If we can't split by "=", skip this equation
                continue

        if not processed_equations:
            return ["No valid equations found"]

        try:
            result = sympy.solve([parse_expr(eq) for eq in processed_equations])
            return self._format_result(result)
        except Exception as e:
            print(f"Error solving equations: {e}")
            return [f"Error solving equations: {str(e)}"]

    def _process_term(self, term: str) -> str:
        """
        Process a term in an equation

        Args:
            term: The term to process

        Returns:
            The processed term
        """
        if not term:
            return term

        # Types: 0 for operators, 1 for numbers, 2 for variables
        types = []
        for char in term:
            if char.isalpha():
                types.append(2)
            elif char.isnumeric():
                types.append(1)
            elif char in ["+", "-"]:
                types.append(0)
            else:
                types.append(-1)  # Unknown

        # Process the term
        result = term[0]  # Keep the first character (sign)

        for i in range(1, len(term) - 1):
            if types[i] == 1 and types[i + 1] == 2:
                # Number followed by variable -> add multiplication
                result += term[i] + "*"
            elif types[i] == 2 and types[i + 1] == 2:
                # Variable followed by variable -> add multiplication
                result += term[i] + "*"
            elif types[i] == 2 and types[i + 1] == 1:
                # Variable followed by number -> add exponentiation
                result += term[i] + "**"
            else:
                result += term[i]

        # Add the last character
        if len(term) > 1:
            result += term[-1]

        return result

    def _format_result(self, result) -> List[str]:
        """
        Format the result for display

        Args:
            result: The solution from SymPy

        Returns:
            List of formatted solution strings
        """
        if not result:
            return ["No solution found"]

        if isinstance(result, dict):
            # Single solution
            formatted = self._format_solution_dict(result)
            return [formatted]
        else:
            # Multiple solutions
            return [self._format_solution_dict(sol) for sol in result]

    def _format_solution_dict(self, solution_dict: dict) -> str:
        """
        Format a solution dictionary

        Args:
            solution_dict: Dictionary with variable: value pairs

        Returns:
            Formatted solution string
        """
        result = ""
        for i, (var, val) in enumerate(solution_dict.items()):
            if i == 0:
                if len(solution_dict) == 1:
                    result += f"( {sympy.latex(var)} = {sympy.latex(val)} )"
                else:
                    result += f"( {sympy.latex(var)} = {sympy.latex(val)} ,"
            elif i == len(solution_dict) - 1:
                result += f" {sympy.latex(var)} = {sympy.latex(val)} )"
            else:
                result += f" {sympy.latex(var)} = {sympy.latex(val)} ,"
        return result
