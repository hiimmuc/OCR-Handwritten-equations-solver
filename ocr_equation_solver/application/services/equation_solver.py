"""
Equation solver service
"""
from typing import Any, List, Tuple

import sympy
from sympy.parsing.sympy_parser import parse_expr

from ocr_equation_solver.application.interfaces.services import EquationSolverInterface


class EquationSolverService:
    """Service for solving mathematical equations"""

    def process_equation_string(self, equation: str) -> str:
        """Process an equation string to make it suitable for the solver"""
        list_type = []
        for char in equation:
            if char.isalpha():
                list_type.append(2)
            if char.isnumeric():
                list_type.append(1)
            if char in ["+", "-"]:
                list_type.append(0)

        processed_string = ""
        for i in range(len(list_type) - 1):
            if list_type[i] == 1 and list_type[i + 1] == 2:
                processed_string += equation[i] + "*"
            elif list_type[i] == 2 and list_type[i + 1] == 2:
                processed_string += equation[i] + "*"
            elif list_type[i] == 2 and list_type[i + 1] == 1:
                processed_string += equation[i] + "**"
            else:
                processed_string += equation[i]
        processed_string += equation[-1]
        return processed_string

    def prepare_equation_for_solver(self, equation: str, is_right_side: bool = False) -> str:
        """Prepare a side of an equation for the solver"""
        if equation[0] not in ["+", "-"]:
            equation = "+" + equation

        positions = [i for i, char in enumerate(equation) if char in ["+", "-"]]
        positions.append(len(equation))

        result = ""
        for i in range(len(positions) - 1):
            segment = equation[positions[i] : positions[i + 1]]
            if is_right_side:
                # Invert signs for right side of equation
                if segment[0] == "+":
                    result += "-" + segment[1:]
                else:
                    result += "+" + segment[1:]
            else:
                result += segment

        return result

    def solve_equations(self, equations: List[str]) -> List[str]:
        """Solve a system of equations"""
        solver_equations = []

        for equation in equations:
            if "=" not in equation:
                continue

            left_side, right_side = equation.split("=")
            solver_string = self.prepare_equation_for_solver(left_side)
            solver_string += self.prepare_equation_for_solver(right_side, is_right_side=True)
            solver_equations.append(solver_string)

        if not solver_equations:
            return ["No valid equations found"]

        try:
            result = sympy.solve([parse_expr(eq) for eq in solver_equations])
            return self.format_solution(result)
        except Exception as e:
            return [f"Error solving equations: {str(e)}"]

    def format_solution(self, solution) -> List[str]:
        """Format the solution for display"""
        if not solution:
            return ["No solution found"]

        # Handle dictionary result (single solution)
        if isinstance(solution, dict):
            result_str = ""
            for i, (key, value) in enumerate(solution.items()):
                if i == 0:
                    if len(solution) == 1:
                        result_str += f"( {sympy.latex(key)} = {sympy.latex(value)} )"
                    else:
                        result_str += f"( {sympy.latex(key)} = {sympy.latex(value)} ,"
                elif i == len(solution) - 1:
                    result_str += f" {sympy.latex(key)} = {sympy.latex(value)} )"
                else:
                    result_str += f" {sympy.latex(key)} = {sympy.latex(value)} ,"
            return [result_str]

        # Handle list result (multiple solutions)
        result_list = []
        for solution_set in solution:
            result_str = ""
            for i, (key, value) in enumerate(solution_set.items()):
                if i == 0:
                    if len(solution_set) == 1:
                        result_str += f"( {sympy.latex(key)} = {sympy.latex(value)} )"
                    else:
                        result_str += f"( {sympy.latex(key)} = {sympy.latex(value)} ,"
                elif i == len(solution_set) - 1:
                    result_str += f" {sympy.latex(key)} = {sympy.latex(value)} )"
                else:
                    result_str += f" {sympy.latex(key)} = {sympy.latex(value)} ,"
            result_list.append(result_str)

        return result_list if result_list else ["No solution found"]
