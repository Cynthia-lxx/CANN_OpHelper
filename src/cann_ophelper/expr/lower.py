"""cann_ophelper.expr.lower -- AST -> statement plan (ExprProgram).

The plan is the single contract consumed by the C++ text generator (fillgen)
and, indirectly, by the verification assets. Statements are flat and ordered:
every statement computes one output slot from source slots, where slots are
either tensor names (from the spec) or scratch slots ``s<i>``.

Kernel plan rules (mirrors docs/expr-rules.md):

- A numeric leaf operand becomes one ``dup`` (AscendC::Duplicate) statement
  writing directly into its parent operator slot (no extra scratch).
- Every non-root operator node (add/sub/mul/div/neg/sigmoid/exp/abs) gets its
  own scratch slot ``s<i>``; the root operator writes straight into the output
  tensor slot (no final copy).
- Constant-only subtrees are folded here using the Python evaluator, so a
  numeric leaf never has two numeric siblings after folding.
- ``neg`` maps to a scalar ``Muls(..., (T)-1)`` statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..i18n import t
from ..model import OpSpecError
from .ast import ExprNode
from .evaluate import evaluate_tree

__all__ = ["ProgramStmt", "ExprProgram", "lower_expr", "fold_constants", "collect_scratch_slots"]


@dataclass(frozen=True)
class ProgramStmt:
    """One kernel vector statement."""

    op: str                          # dup/neg/sigmoid/exp/abs/add/sub/mul/div
    dst: str                         # slot token: tensor name or scratch 's<i>'
    srcs: tuple[str, ...] = ()
    scalar: Optional[float] = None   # operand of ``dup``


@dataclass(frozen=True)
class ExprProgram:
    """Flat ordered statement plan produced by lower_expr."""

    statements: tuple[ProgramStmt, ...]
    scratch_count: int = 0
    #: Distinct input tensor names referenced by the expression (DFS order).
    inputs: tuple[str, ...] = ()


def fold_constants(node: ExprNode) -> ExprNode:
    """Replace constant-only subtrees by a single number node (recursive)."""
    if node.kind in ("number", "ref"):
        return node
    children = tuple(fold_constants(c) for c in node.args)
    folded = ExprNode(node.kind, node.value, children)
    from .ast import canonical_text
    from .evaluate import collect_tensor_names

    if not collect_tensor_names(folded):
        # Pure-numeric subtree: evaluate with an empty environment and inline
        # the resulting literal. Evaluation cannot fail: only + - * / neg and
        # the supported unary functions appear here.
        try:
            return _NUMBER_FACTORY(evaluate_tree(folded, {}))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise OpSpecError(t("expr.lower.const_eval", text=canonical_text(folded), err=exc)) from exc
    return folded


def _number(value: float) -> ExprNode:
    from .ast import number

    return number(value)


_NUMBER_FACTORY = _number


def _slot_is_scratch(slot: str) -> bool:
    return slot.startswith("s") and slot[1:].isdigit()


def collect_scratch_slots(statements: list[ProgramStmt]) -> int:
    """Highest scratch slot index + 1 used by the statements."""
    top = -1
    for stmt in statements:
        for slot in (stmt.dst, *stmt.srcs):
            if _slot_is_scratch(slot):
                top = max(top, int(slot[1:]))
    return top + 1


class _Lowerer:
    def __init__(self) -> None:
        self.statements: list[ProgramStmt] = []
        self._temp_index = 0

    def _new_scratch(self) -> str:
        slot = f"s{self._temp_index}"
        self._temp_index += 1
        return slot

    def emit(self, node: ExprNode, force_dst: Optional[str] = None) -> str:
        """Emit statements producing ``node`` and return its slot token.

        :param force_dst: write the result into this slot instead of a new one
            (used for the root output and numeric leaves).
        """
        if node.kind == "number":
            if force_dst is None:
                raise OpSpecError(t("expr.lower.root_number"))
            self.statements.append(ProgramStmt("dup", force_dst, scalar=float(node.value)))
            return force_dst
        if node.kind == "ref":
            return str(node.value)

        if node.kind == "unary" or node.kind == "call":
            dst = force_dst if force_dst is not None else self._new_scratch()
            src = self.emit(node.args[0])
            op = "neg" if node.kind == "unary" else str(node.value)
            self.statements.append(ProgramStmt(op, dst, (src,)))
            return dst

        # binary
        left, right = node.args
        dst = force_dst if force_dst is not None else self._new_scratch()
        # A numeric leaf is materialised straight into this operator's own slot
        # (Duplicate + in-place combine), so it needs no separate scratch.
        left_force = dst if left.kind == "number" else None
        right_force = dst if right.kind == "number" else None
        left_slot = self.emit(left, force_dst=left_force)
        right_slot = self.emit(right, force_dst=right_force)
        self.statements.append(ProgramStmt(str(node.value), dst, (left_slot, right_slot)))
        return dst


def lower_expr(tree: ExprNode, output_name: str) -> ExprProgram:
    """Lower a validated expression tree into an ExprProgram.

    :param tree: already parsed/folded expression (number/ref nodes allowed);
    :param output_name: single output tensor slot (result destination);
    :raises OpSpecError: If the expression references nothing (pure constant).
    """
    from .evaluate import collect_tensor_names

    folded = fold_constants(tree)
    inputs = tuple(collect_tensor_names(folded))
    if not inputs:
        raise OpSpecError(t("expr.lower.no_input"), hint=t("expr.lower.no_input.hint"))

    lowerer = _Lowerer()
    root_slot = lowerer.emit(folded, force_dst=output_name)
    if root_slot != output_name:  # pragma: no cover - defensive; root always forced
        raise OpSpecError(t("expr.lower.internal", slot=root_slot, output=output_name))
    scratch_count = collect_scratch_slots(lowerer.statements)
    return ExprProgram(
        statements=tuple(lowerer.statements),
        scratch_count=scratch_count,
        inputs=inputs,
    )
