from __future__ import annotations

import ast
import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Dict

from ..schemas import ToolResult
from ..tools import ToolActionSpec, ToolFamily


def _calc_cagr(start_value: float, end_value: float, periods: float) -> float:
    if start_value <= 0:
        raise ValueError("cagr requires start_value > 0")
    if end_value < 0:
        raise ValueError("cagr requires end_value >= 0")
    if periods <= 0:
        raise ValueError("cagr requires periods > 0")
    return (end_value / start_value) ** (1 / periods) - 1


def _calc_cagr_pct(start_value: float, end_value: float, periods: float) -> float:
    return _calc_cagr(start_value, end_value, periods) * 100


class LocalCalculatorTool(ToolFamily):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the raw result plus neutral numeric variants "
        "such as comma formatting, integer floor/ceil/nearest, and fixed decimal forms. Use for "
        "percentages, basis points, CAGR, ratios, range midpoints, differences, totals, and any "
        "multi-step arithmetic instead of computing by hand. Supports round(...), pow(...), "
        "cagr(start_value, end_value, periods), cagr_pct(start_value, end_value, periods), "
        "and normalizes ^ to ** for exponentiation. Optional precision returns an additional "
        "rounded_requested value. The model must choose the final display from the task's requested "
        "unit, scale, and precision."
    )
    paid = False
    default_action = "default"
    actions = {
        "default": ToolActionSpec(
            description=(
                "Evaluate a mathematical expression. Optional precision returns a rounded_requested "
                "variant; the raw result is always returned."
            ),
            required=["expression"],
            optional=["precision"],
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '(432876657 + 188462942)'.",
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Optional number of decimal places for an additional rounded variant.",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        )
    }
    _ALLOWED_BINOPS = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
        ast.Mod: lambda left, right: left % right,
    }
    _ALLOWED_UNARYOPS = {
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }
    _ALLOWED_FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "cagr": _calc_cagr,
        "cagr_pct": _calc_cagr_pct,
    }

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        input_expression = str(arguments.get("expression", "")).strip()
        if not input_expression:
            return ToolResult(
                self.name,
                "local_calculator",
                "error",
                action="default",
                error="expression must not be empty",
                paid=False,
                confidence=0.1,
            )
        expression = self._normalize_expression(input_expression)
        try:
            result = self._eval_expression(expression)
        except ZeroDivisionError:
            return ToolResult(
                self.name,
                "local_calculator",
                "error",
                action="default",
                error=f"division by zero in '{input_expression}'",
                paid=False,
                confidence=0.1,
            )
        except Exception as exc:
            return ToolResult(
                self.name,
                "local_calculator",
                "error",
                action="default",
                error=f"invalid expression '{input_expression}': {exc}",
                paid=False,
                confidence=0.1,
            )
        structured = self._structured_numeric_result(result, arguments)
        result_text = str(result)
        structured_text = json.dumps(structured, ensure_ascii=False, default=str, separators=(",", ":"))
        observation = f"raw: {result_text}\nstructured_result: {structured_text}"
        range_hint = self._range_completeness_hint(expression)
        if range_hint:
            observation = f"{observation}\n{range_hint}"
        metadata: Dict[str, Any] = {
            "result": result,
            "expression": expression,
            "structured_result": structured,
        }
        if expression != input_expression:
            metadata["input_expression"] = input_expression
        if range_hint:
            metadata["range_hint"] = True
        return ToolResult(
            self.name,
            "local_calculator",
            "success",
            action="default",
            observation=observation,
            confidence=1.0,
            paid=False,
            metadata=metadata,
        )

    @staticmethod
    def _structured_numeric_result(result: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        precision = arguments.get("precision")
        try:
            decimal_value = Decimal(str(result))
        except (InvalidOperation, ValueError):
            return {
                "raw": str(result),
                "numeric": False,
            }

        structured: Dict[str, Any] = {
            "raw": str(result),
            "numeric": True,
        }
        if decimal_value.is_finite():
            nearest_integer = decimal_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            floor_value = decimal_value.to_integral_value(rounding=ROUND_FLOOR)
            ceil_value = decimal_value.to_integral_value(rounding=ROUND_CEILING)
            structured.update(
                {
                    "decimal": format(decimal_value, "f"),
                    "display": LocalCalculatorTool._format_number_string(decimal_value),
                    "is_integer": decimal_value == decimal_value.to_integral_value(),
                    "integer_nearest": str(nearest_integer),
                    "integer_nearest_display": LocalCalculatorTool._format_number_string(nearest_integer),
                    "integer_floor": str(floor_value),
                    "integer_floor_display": LocalCalculatorTool._format_number_string(floor_value),
                    "integer_ceil": str(ceil_value),
                    "integer_ceil_display": LocalCalculatorTool._format_number_string(ceil_value),
                    "fixed_0dp": LocalCalculatorTool._format_decimal(decimal_value, 0),
                    "fixed_1dp": LocalCalculatorTool._format_decimal(decimal_value, 1),
                    "fixed_2dp": LocalCalculatorTool._format_decimal(decimal_value, 2),
                    "fixed_4dp": LocalCalculatorTool._format_decimal(decimal_value, 4),
                }
            )
            try:
                precision_int = int(precision)
            except Exception:
                precision_int = None
            if precision_int is not None and precision_int >= 0:
                structured["requested_precision"] = precision_int
                if precision_int <= 12:
                    structured["rounded_requested"] = LocalCalculatorTool._format_decimal(decimal_value, precision_int)
                else:
                    structured["rounded_requested"] = "omitted: requested precision is above the tool display cap; use raw"
        return structured

    @staticmethod
    def _format_decimal(value: Decimal, places: int) -> str:
        quant = Decimal("1") if places <= 0 else Decimal("1").scaleb(-places)
        rounded = value.quantize(quant, rounding=ROUND_HALF_UP)
        return LocalCalculatorTool._format_number_string(rounded)

    @staticmethod
    def _format_number_string(value: Decimal) -> str:
        if value == value.to_integral_value():
            return f"{int(value):,}"
        normalized = format(value.normalize(), "f")
        whole, _, frac = normalized.partition(".")
        sign = ""
        if whole.startswith("-"):
            sign = "-"
            whole = whole[1:]
        rendered = f"{int(whole or '0'):,}"
        return f"{sign}{rendered}.{frac.rstrip('0')}" if frac.rstrip("0") else f"{sign}{rendered}"

    @staticmethod
    def _normalize_expression(expression: str) -> str:
        return str(expression or "").replace("^", "**")

    @staticmethod
    def _range_completeness_hint(expression: str) -> str:
        expr = str(expression or "")
        compact = re.sub(r"\s+", "", expr)
        lowered = compact.lower()
        midpoint_pattern = re.search(r"\([^()]*[+\-][^()]*\)/2", compact)
        has_range_function = "min(" in lowered or "max(" in lowered
        has_avg_of_two = bool(midpoint_pattern)
        if not (has_avg_of_two or has_range_function):
            return ""
        return (
            "note: range/endpoint inputs detected. If this calculation supports a beat/miss, guidance, "
            "comparison, or difference against a range and the task does not explicitly request midpoint, "
            "also compute/check the low-end and high-end deltas before final_answer."
        )

    def _eval_expression(self, expression: str) -> Any:
        parsed = ast.parse(expression, mode="eval")
        return self._eval_node(parsed.body)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            operator = self._ALLOWED_BINOPS.get(type(node.op))
            if operator is None:
                raise ValueError(f"unsupported operator {type(node.op).__name__}")
            return operator(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            operator = self._ALLOWED_UNARYOPS.get(type(node.op))
            if operator is None:
                raise ValueError(f"unsupported unary operator {type(node.op).__name__}")
            return operator(self._eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = self._ALLOWED_FUNCTIONS.get(node.func.id)
            if function is None:
                raise ValueError(f"unsupported function {node.func.id}")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return function(*(self._eval_node(argument) for argument in node.args))
        raise ValueError(f"unsupported expression node {type(node).__name__}")


class CalculatorTool(LocalCalculatorTool):
    name = "calculator"
    default_action = "calculate"
    actions = {
        "calculate": ToolActionSpec(
            "Evaluate an exact arithmetic expression for percentages, basis points, CAGR, ratios, "
            "midpoints, differences, and totals. Supports round(...), pow(...), "
            "cagr(start_value, end_value, periods), cagr_pct(start_value, end_value, periods), "
            "and ^ as exponentiation.",
            required=["expression"],
            optional=["precision"],
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Optional number of decimal places for an additional rounded variant.",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        ),
    }

    def normalize_action(self, action: str) -> str:
        candidate = (action or "").strip() or self.default_action
        return "calculate" if candidate in {"default", "calculate"} else candidate

    def forward(self, action: str, arguments: Dict[str, Any], timeout: int = 30) -> ToolResult:
        result = super().forward("default", arguments, timeout=timeout)
        result.tool_family = self.name
        result.action = "calculate"
        return result
