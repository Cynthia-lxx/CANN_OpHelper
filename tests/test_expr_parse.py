"""cann_ophelper.expr.parse -- real grammar/transformer regression tests.

The infix/LaTeX path needs ``lark`` (declared in pyproject.toml, installed by
the user into the venv).  These tests are skipped when lark is missing so the
rest of the suite stays runnable; every other test module deliberately builds
ASTs through the factories and never requires lark.

Regression pin: the sum/product rules rely on ``keep_all_tokens=True`` (the
anonymous "+ - * /" terminals are otherwise filtered out of the tree and the
left-to-right fold loses every operator).
"""

import pytest

pytest.importorskip("lark")

from cann_ophelper.expr.parse import MAX_EXPR_DEPTH, parse_expr
from cann_ophelper.expr.ast import ExprNode
from cann_ophelper.model import OpSpecError

# convenience helpers ---------------------------------------------------------


def _kind(node: ExprNode) -> str:
    return node.kind


# infix math ------------------------------------------------------------------


def test_doc_example_parse_and_shape():
    parsed = parse_expr("A + 2/sigmoid(B) = C")
    assert parsed.output == "C"
    tree = parsed.tree
    assert _kind(tree) == "binary" and tree.value == "add"
    left, right = tree.args
    assert left == ExprNode(kind="ref", value="A")
    assert _kind(right) == "binary" and right.value == "div"
    _, denom = right.args
    assert _kind(denom) == "call" and denom.value == "sigmoid"


def test_infix_matches_latex_equivalent():
    infix = parse_expr("A + 2/sigmoid(B)").tree
    latex = parse_expr(r"A + \frac{2}{\mathrm{sigmoid}(B)}").tree
    assert latex == infix


def test_left_associativity_folds_left_to_right():
    tree = parse_expr("A-B-C").tree
    assert tree.value == "sub"
    inner, right = tree.args
    assert inner.value == "sub" and right.value == "C"


def test_parentheses_override_precedence():
    tree = parse_expr("(A+B)*2").tree
    assert tree.value == "mul"
    inner, factor = tree.args
    assert inner.value == "add" and factor.value == 2.0


def test_unary_neg_and_positive():
    assert parse_expr("-A").tree == ExprNode(kind="unary", value="neg", args=(ExprNode(kind="ref", value="A"),))
    assert parse_expr("+A").tree == ExprNode(kind="ref", value="A")


def test_chained_division_keeps_every_operator():
    tree = parse_expr("A/2/3").tree
    assert tree.value == "div"
    inner, right = tree.args
    assert inner.value == "div" and right.value == 3.0


def test_function_call_arity_and_aliases():
    assert parse_expr("exp(B)").tree.value == "exp"
    with pytest.raises(OpSpecError):
        parse_expr("sigmoid(B, B)")  # arity mismatch


def test_unknown_function_rejected():
    with pytest.raises(OpSpecError) as exc:
        parse_expr("foo(A)")
    assert "sigmoid" in str(exc.value)  # hints the known-function list


# output-name handling --------------------------------------------------------


def test_output_name_variants():
    assert parse_expr("A+1").output is None
    assert parse_expr("A+1=Z").output == "Z"
    assert parse_expr("A+1", output="Z").output == "Z"


def test_output_mismatch_rejected():
    with pytest.raises(OpSpecError):
        parse_expr("A+1=Z", output="C")


def test_empty_and_bad_output_rejected():
    with pytest.raises(OpSpecError):
        parse_expr("   ")
    with pytest.raises(OpSpecError):
        parse_expr("A=")


def test_power_syntax_rejected():
    with pytest.raises(OpSpecError):
        parse_expr("A^2")
    with pytest.raises(OpSpecError):
        parse_expr("A**2")


def test_unknown_latex_command_rejected():
    with pytest.raises(OpSpecError):
        parse_expr(r"\sqrt{A}")


def test_depth_limit_enforced():
    # parentheses vanish during the transform, so nest real AST nodes instead:
    # a long "-"-unary chain builds an AST whose depth exceeds the limit.
    deep = "-" * (MAX_EXPR_DEPTH + 4) + "A"
    with pytest.raises(OpSpecError):
        parse_expr(deep)
