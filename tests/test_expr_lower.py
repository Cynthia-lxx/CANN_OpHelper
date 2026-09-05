"""Tests for the expression lowering planner (AST -> ExprProgram).

The planner is the single contract between parsed expressions and both the
generated kernel (fillgen) and the Python reference evaluator, so these tests
pin down its slot/scratch allocation semantics without requiring lark (the
AST factories are used directly).
"""

from __future__ import annotations

import pytest

from cann_ophelper.expr.ast import binary, call, number, ref, unary
from cann_ophelper.expr.lower import ExprProgram, fold_constants, lower_expr
from cann_ophelper.model import OpSpecError


def stmt_ops(program: ExprProgram) -> list[str]:
    return [s.op for s in program.statements]


def stmt_repr(program: ExprProgram) -> list[tuple[str, str, tuple, object]]:
    return [(s.op, s.dst, s.srcs, s.scalar) for s in program.statements]


def test_single_input_identity_via_add_zero_is_not_folded():
    # The planner does not do zero/special-value algebra; a ref + ref lowers to one statement.
    program = lower_expr(binary("add", ref("A"), ref("B")), "C")
    assert stmt_ops(program) == ["add"]
    assert stmt_repr(program) == [("add", "C", ("A", "B"), None)]
    assert program.scratch_count == 0
    assert program.inputs == ("A", "B")


def test_root_unary_call_writes_output_directly():
    program = lower_expr(call("sigmoid", ref("A")), "C")
    assert stmt_ops(program) == ["sigmoid"]
    assert stmt_repr(program) == [("sigmoid", "C", ("A",), None)]
    assert program.scratch_count == 0
    assert program.inputs == ("A",)


def test_root_neg_is_a_scalar_statement():
    # -A == C lowers to a single 'neg' statement; fillgen maps it to Muls(., ., -1).
    program = lower_expr(unary("neg", ref("A")), "C")
    assert stmt_repr(program) == [("neg", "C", ("A",), None)]
    assert program.scratch_count == 0


def test_nested_expression_uses_one_scratch_per_internal_node():
    # A + 2/sigmoid(B) == add(A, div(2, sigmoid(B)))
    tree = binary("add", ref("A"), binary("div", number(2), call("sigmoid", ref("B"))))
    program = lower_expr(tree, "C")
    # dup s0=2; sigmoid s1<-B; div s0<-s0/s1; add C<-A/s0
    assert stmt_repr(program) == [
        ("dup", "s0", (), 2.0),
        ("sigmoid", "s1", ("B",), None),
        ("div", "s0", ("s0", "s1"), None),
        ("add", "C", ("A", "s0"), None),
    ]
    assert program.scratch_count == 2
    assert program.inputs == ("A", "B")


def test_numeric_leaf_reuses_parent_slot_in_place():
    # A * 2 == mul(A, 2): the literal fills the parent's slot, then an in-place mul.
    program = lower_expr(binary("mul", ref("A"), number(2)), "C")
    assert stmt_repr(program) == [
        ("dup", "C", (), 2.0),
        ("mul", "C", ("A", "C"), None),
    ]
    assert program.scratch_count == 0


def test_constant_on_left_side_uses_scratch_for_internal_and_reuses_at_root():
    # 2 - A at the root: 2 is the left operand, so it must fill a scratch that is
    # then subtracted, i.e. sub(C, dup(2), A) -- but dup targets the root slot only
    # for the right operand? v1 lowers literals into the *operator* slot; for a
    # left literal that slot is a scratch below the root.
    tree = binary("sub", number(2), ref("A"))
    # nested under an outer node to force the scratch path
    outer = binary("add", ref("B"), tree)
    program = lower_expr(outer, "C")
    assert stmt_repr(program) == [
        ("dup", "s0", (), 2.0),
        ("sub", "s0", ("s0", "A"), None),
        ("add", "C", ("B", "s0"), None),
    ]
    assert program.scratch_count == 1


def test_constant_root_raises_no_input():
    with pytest.raises(OpSpecError):
        lower_expr(number(5), "C")


def test_constant_only_subexpression_is_folded():
    tree = binary("add", ref("A"), binary("div", number(4), number(2)))
    folded = fold_constants(tree)
    assert folded.kind == "binary"
    assert folded.args[1].kind == "number"
    assert folded.args[1].value == 2.0


def test_deep_chains_reuse_output_slot_and_scratch_sequential():
    # ((A - B) * C) + A with three refs through two internal nodes.
    # Scratch numbering follows pre-order DFS (a parent claims its slot before
    # descending), so the inner sub claims s1 while its parent mul already holds s0.
    tree = binary("add", binary("mul", binary("sub", ref("A"), ref("B")), ref("C")), ref("A"))
    program = lower_expr(tree, "D")
    assert stmt_repr(program) == [
        ("sub", "s1", ("A", "B"), None),
        ("mul", "s0", ("s1", "C"), None),
        ("add", "D", ("s0", "A"), None),
    ]
    assert program.inputs == ("A", "B", "C")
    assert program.scratch_count == 2


def test_duplicate_tensor_refs_appear_once_in_inputs():
    program = lower_expr(binary("add", ref("A"), ref("A")), "C")
    assert program.inputs == ("A",)
