from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result as a string.
    Supports basic arithmetic: +, -, *, /, **, (, ).
    Use for financial calculations like growth rates, margins, ratios.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are supported"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(round(float(result), 6))
    except Exception as e:
        return f"Error: {e}"
