"""cann_ophelper.expr.ast -- Frozen expression AST shared by parser/lower/evaluator.

Node kinds:
- ``number``: literal scalar (``value`` is a finite float);
- ``ref``:   reference to an input/output tensor by name (``value`` is the name);
- ``unary``: one-operand node (``value`` is the canonical op, e.g. ``neg``);
- ``binary``: two-operand node (``value`` in add/sub/mul/div);
- ``call``:  named function applied to ``args`` (``value`` in sigmoid/exp/abs).

This module is dependency-free (stdlib only) so the IR can be imported by every
layer, including the Python reference evaluator.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Tuple

#: Canonical unary op names understood by lower + evaluator.
UNARY_OPS = frozenset({"neg"})

#: Canonical binary op names understood by lower + evaluator.
BINARY_OPS = frozenset({"add", "sub", "mul", "div"})

#: Named (function-call) ops and their arity. Mirrors docs/expr-rules.md §1.
CALL_OPS = {"sigmoid": 1, "exp": 1, "abs": 1}

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ExprNode:
    """A single immutable expression node (value + children)."""

    kind: str
    value: Any
    args: Tuple["ExprNode", ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("number", "ref", "unary", "binary", "call"):
            raise ValueError(f"unknown ExprNode kind: {self.kind!r}")
        for child in self.args:
            if not isinstance(child, ExprNode):
                raise TypeError(f"ExprNode args must be ExprNode, got {type(child).__name__}")


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def number(value: float) -> ExprNode:
    """Literal scalar node. ``value`` must be a finite real number."""
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"literal must be finite, got {value!r}")
    return ExprNode("number", f)


def ref(name: str) -> ExprNode:
    """Tensor reference node. ``name`` must be a C-identifier."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid tensor reference name: {name!r}")
    return ExprNode("ref", name)


def unary(op: str, arg: ExprNode) -> ExprNode:
    """Unary operator node (canonical op names from UNARY_OPS)."""
    if op not in UNARY_OPS:
        raise ValueError(f"unknown unary op: {op!r}")
    return ExprNode("unary", op, (arg,))


def binary(op: str, left: ExprNode, right: ExprNode) -> ExprNode:
    """Binary operator node (canonical op names from BINARY_OPS)."""
    if op not in BINARY_OPS:
        raise ValueError(f"unknown binary op: {op!r}")
    return ExprNode("binary", op, (left, right))


def call(name: str, *args: ExprNode) -> ExprNode:
    """Function-call node. Arity is checked against CALL_OPS when known."""
    if name not in CALL_OPS:
        raise ValueError(f"unknown function: {name!r}")
    if len(args) != CALL_OPS[name]:
        raise ValueError(f"{name} expects {CALL_OPS[name]} arg(s), got {len(args)}")
    return ExprNode("call", name, args)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def fmt_num(value: float) -> str:
    """Human-friendly literal rendering: integral values drop the trailing '.0'."""
    if float(value).is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(float(value))


def canonical_text(node: ExprNode) -> str:
    """Fully-parenthesized canonical string of an AST (used as fingerprint/echo)."""
    if node.kind == "number":
        return fmt_num(node.value)
    if node.kind == "ref":
        return str(node.value)
    if node.kind == "call":
        return f"{node.value}({', '.join(canonical_text(a) for a in node.args)})"
    if node.kind == "unary":
        op, (arg,) = node.value, node.args
        if op == "neg":
            return f"(-{canonical_text(arg)})"
        raise ValueError(f"cannot render unary op: {op!r}")
    if node.kind == "binary":
        op, (left, right) = node.value, node.args
        if op == "add":
            return f"({canonical_text(left)} + {canonical_text(right)})"
        if op == "sub":
            return f"({canonical_text(left)} - {canonical_text(right)})"
        if op == "mul":
            return f"({canonical_text(left)} * {canonical_text(right)})"
        if op == "div":
            return f"({canonical_text(left)} / {canonical_text(right)})"
        raise ValueError(f"cannot render binary op: {op!r}")
    raise ValueError(f"cannot render node kind: {node.kind!r}")
