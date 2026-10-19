"""Deterministic scores for the three declared small text-quality tasks.

These scores do not run inference, execute tools, or evaluate arbitrary Python.
An unfinished generation cannot receive a passing score. Coding uses a small
AST interpreter; an unsupported expression is unscorable rather than guessed.
"""

import ast
import json

from .messages import EOS, parse_completion


class UnsupportedExpression(ValueError): pass


def _expression(node, xs):
    if isinstance(node, ast.Name) and node.id == "xs": return xs
    if isinstance(node, ast.Constant) and type(node.value) is int: return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _expression(node.operand, xs)
        if type(value) is not int: raise UnsupportedExpression("Expected an integer index")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        a, b = _expression(node.left, xs), _expression(node.right, xs)
        if type(a) is not int or type(b) is not int: raise UnsupportedExpression("Only integer index arithmetic is supported")
        return a + b if isinstance(node.op, ast.Add) else a - b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1 and not node.keywords:
        value = _expression(node.args[0], xs)
        if type(value) is not list: raise UnsupportedExpression("Only list lengths are supported")
        return len(value)
    if isinstance(node, ast.Subscript):
        value, index = _expression(node.value, xs), _expression(node.slice, xs)
        if type(value) is not list or type(index) is not int: raise UnsupportedExpression("Only integer list indexing is supported")
        return value[index]
    raise UnsupportedExpression("Expression uses an operation outside the declared evaluator subset")


def score_completion(case, text, *, stop_reason):
    """Score an actual decoded response, including its actual EOS marker.

    The caller must bind this text and stop reason to a model capture. This
    function supplies no such provenance and never appends a missing EOS.
    """
    if stop_reason != "eos": return {"status": "incomplete", "passed": None, "reason": "generation_did_not_reach_eos"}
    if type(text) is not str or not text.endswith(EOS):
        return {"status": "fail", "passed": False, "reason": "missing_actual_eos"}
    try: parsed = parse_completion(text, thinking_mode=case["thinking_mode"])
    except ValueError: return {"status": "fail", "passed": False, "reason": "malformed_completion"}
    scoring = case["scoring"]; kind = scoring["kind"]
    if kind == "final_text_exact":
        passed = not parsed["tool_calls"] and parsed["content"].strip() == scoring["expected"]
        return {"status": "pass" if passed else "fail", "passed": passed, "reason": "exact_final_text"}
    if kind == "one_tool_call_exact":
        calls = parsed["tool_calls"]; passed = False
        if len(calls) == 1 and not parsed["content"].strip():
            c = calls[0]
            def unique(pairs):
                result = {}
                for k, v in pairs:
                    if k in result: raise ValueError("Duplicate JSON key")
                    result[k] = v
                return result
            try: args = json.loads(c["function"]["arguments"], object_pairs_hook=unique)
            except (ValueError, TypeError, RecursionError): args = None
            expected = scoring["arguments"]
            passed = (c["function"]["name"] == scoring["name"] and "namespace" not in c and type(args) is dict
                and set(args) == set(expected) and all(type(args[k]) is type(v) and args[k] == v for k, v in expected.items()))
        return {"status": "pass" if passed else "fail", "passed": passed, "reason": "exact_tool_name_and_typed_arguments"}
    if kind == "python_expression_cases":
        if parsed["tool_calls"]: return {"status": "fail", "passed": False, "reason": "unexpected_tool_call"}
        expression = parsed["content"].strip()
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64: raise UnsupportedExpression("Excessive expression")
            actual = [_expression(tree.body, c["input"]) for c in scoring["cases"]]
            passed = all(type(a) is type(c["expected"]) and a == c["expected"] for a, c in zip(actual, scoring["cases"]))
            return {"status": "pass" if passed else "fail", "passed": passed, "reason": "declared_python_expression_cases", "actual": actual}
        except UnsupportedExpression:
            return {"status": "unscorable", "passed": None, "reason": "expression_outside_declared_AST_subset"}
        except (SyntaxError, IndexError, TypeError, ValueError, RecursionError):
            return {"status": "fail", "passed": False, "reason": "invalid_expression_or_index"}
    raise ValueError("Unknown declared quality score")
