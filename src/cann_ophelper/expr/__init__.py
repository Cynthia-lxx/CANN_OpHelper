"""cann_ophelper.expr -- Element-wise expression parsing, rules and evaluation.

Public surface used by gen-op / fill-op / wizard:
- ``parse_expr``: text -> ParsedExpr (AST + optional output tensor name);
- ``resolve_preset_expr``: named presets (add/mul/sigmoid/...) -> expr text;
- ``evaluate_tensors``: deterministic pure-Python expected output;
- lower (statement planning) lives in :mod:`cann_ophelper.expr.lower`.
"""

from __future__ import annotations

from .ast import (
    CALL_OPS,
    ExprNode,
    binary,
    call,
    canonical_text,
    fmt_num,
    number,
    ref,
    unary,
)
from .evaluate import collect_tensor_names, evaluate_tensors, evaluate_tree
from .lower import ExprProgram, ProgramStmt, lower_expr
from .parse import MAX_EXPR_DEPTH, MAX_EXPR_NODES, ParsedExpr, latex_to_infix, parse_expr
from .presets import EXPR_PRESETS, OpRule, SYMBOL_RULES, resolve_preset_expr

__all__ = [
    # ast
    "ExprNode",
    "number",
    "ref",
    "unary",
    "binary",
    "call",
    "CALL_OPS",
    "canonical_text",
    "fmt_num",
    # parse
    "parse_expr",
    "ParsedExpr",
    "latex_to_infix",
    "MAX_EXPR_DEPTH",
    "MAX_EXPR_NODES",
    # presets
    "SYMBOL_RULES",
    "OpRule",
    "EXPR_PRESETS",
    "resolve_preset_expr",
    # evaluate
    "evaluate_tree",
    "evaluate_tensors",
    "collect_tensor_names",
    # lower
    "ExprProgram",
    "ProgramStmt",
    "lower_expr",
]
