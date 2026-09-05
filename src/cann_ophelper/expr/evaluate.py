"""cann_ophelper.expr.evaluate -- Pure-Python reference evaluator.

Evaluates an :class:`ExprNode` the same way the generated kernel computes it.
Used to produce deterministic expected outputs on the local machine (no CANN,
no NumPy). The dtype contract is float32; Python floats are far more precise,
and the device comparison tolerates this with the official rtol/atol=1e-3.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..i18n import t
from ..model import OpSpecError
from .ast import ExprNode

__all__ = ["evaluate_tree", "evaluate_tensors", "collect_tensor_names"]


def _sigmoid(x: float) -> float:
    # 1 / (1 + exp(-x)); matches AscendC::Sigmoid semantics.
    return 1.0 / (1.0 + math.exp(-x))


def evaluate_tree(node: ExprNode, env: Mapping[str, float]) -> float:
    """Evaluate a scalar expression given tensor-name -> scalar bindings."""
    if node.kind == "number":
        return float(node.value)
    if node.kind == "ref":
        return float(env[str(node.value)])
    if node.kind == "unary":
        (arg,) = node.args
        value = evaluate_tree(arg, env)
        if node.value == "neg":
            return -value
        raise ValueError(f"unhandled unary op: {node.value!r}")
    if node.kind == "binary":
        left = evaluate_tree(node.args[0], env)
        right = evaluate_tree(node.args[1], env)
        if node.value == "add":
            return left + right
        if node.value == "sub":
            return left - right
        if node.value == "mul":
            return left * right
        if node.value == "div":
            return left / right
        raise ValueError(f"unhandled binary op: {node.value!r}")
    if node.kind == "call":
        fn = str(node.value)
        if fn == "sigmoid":
            return _sigmoid(evaluate_tree(node.args[0], env))
        if fn == "exp":
            return math.exp(evaluate_tree(node.args[0], env))
        if fn == "abs":
            return abs(evaluate_tree(node.args[0], env))
        raise ValueError(f"unhandled function: {fn!r}")
    raise ValueError(f"unhandled node kind: {node.kind!r}")


def collect_tensor_names(node: ExprNode) -> list[str]:
    """Return the distinct tensor references (DFS, stable order)."""
    seen: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind == "ref":
            name = str(current.value)
            if name not in seen:
                seen.append(name)
        stack.extend(reversed(current.args))
    return seen


def _require_same_length(labels: Sequence[str], lengths: Mapping[str, int]) -> int:
    total = -1
    for name in labels:
        length = lengths.get(name, 0)
        if total == -1:
            total = length
        elif length != total:
            raise OpSpecError(
                t("expr.evaluate.length", name=name, got=length),
                hint=t("expr.evaluate.same_shape.hint"),
            )
    return total


def evaluate_tensors(tree: ExprNode, tensors: Mapping[str, Sequence[float]]) -> list[float]:
    """Evaluate a whole vector (element-wise) expression.

    :param tree: expression AST (references input tensors);
    :param tensors: name -> flat list of float values (same length each);
    :returns: flat list with the result (length == input length).
    :raises OpSpecError: On missing tensors or length mismatch.
    """
    names = collect_tensor_names(tree)
    for name in names:
        if name not in tensors:
            raise OpSpecError(t("expr.evaluate.missing", name=name))
    total = _require_same_length(names, {n: len(tensors[n]) for n in names})

    result: list[float] = []
    for i in range(total):
        env = {n: tensors[n][i] for n in names}
        result.append(evaluate_tree(tree, env))
    return result
