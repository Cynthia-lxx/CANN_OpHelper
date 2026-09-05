"""cann_ophelper.expr.presets -- Symbol rules and named expression presets.

The symbol table mirrors ``docs/expr-rules.md`` (the authoritative source):
every canonical op maps to its display form, arity and kernel plan. Adding a
symbol means extending this table *and* the lower/fillgen branches.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..i18n import t
from ..model import OpSpecError

__all__ = ["OpRule", "SYMBOL_RULES", "EXPR_PRESETS", "resolve_preset_expr"]


@dataclass(frozen=True)
class OpRule:
    """One supported expression symbol."""

    name: str            # canonical op name used in the AST (add/sub/...)
    display: str         # human symbol, e.g. '+'
    kind: str            # "binary" | "unary" | "function"
    arity: int
    note: str = ""       # one-line hint shown in errors/help


#: Canonical-name -> rule. Kinds follow docs/expr-rules.md §1.
SYMBOL_RULES: dict[str, OpRule] = {
    "add": OpRule("add", "+", "binary", 2),
    "sub": OpRule("sub", "-", "binary", 2),
    "mul": OpRule("mul", "*", "binary", 2),
    "div": OpRule("div", "/", "binary", 2),
    "neg": OpRule("neg", "-", "unary", 1),
    "sigmoid": OpRule("sigmoid", "sigmoid", "function", 1),
    "exp": OpRule("exp", "exp", "function", 1),
    "abs": OpRule("abs", "abs", "function", 1),
}

#: Python names accepted for function-style calls, mapped to canonical names.
FUNCTION_ALIASES: dict[str, str] = {"sigmoid": "sigmoid", "exp": "exp", "abs": "abs"}

#: Display symbols understood by the infix grammar -> canonical names.
INFIX_OP_MAP: dict[str, str] = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


#: Named expression presets: rule-name -> (expr text, one-line description).
#: These reuse the default tensor naming convention A/B/C used by gen-op.
EXPR_PRESETS: dict[str, tuple[str, str]] = {
    "add": ("A + B = C", "element-wise add, two inputs"),
    "sub": ("A - B = C", "element-wise subtract, two inputs"),
    "mul": ("A * B = C", "element-wise multiply, two inputs"),
    "div": ("A / B = C", "element-wise divide, two inputs"),
    "sigmoid": ("sigmoid(A) = C", "element-wise sigmoid, one input"),
    "exp": ("exp(A) = C", "element-wise exp, one input"),
    "abs": ("abs(A) = C", "element-wise absolute value, one input"),
}


def resolve_preset_expr(name: str) -> tuple[str, str]:
    """Return ``(expr_text, description)`` for a named preset.

    :raises OpSpecError: If the preset is unknown (bilingual message).
    """
    key = str(name).strip().lower()
    if key not in EXPR_PRESETS:
        raise OpSpecError(
            t("expr.preset_unknown", value=name, values=", ".join(sorted(EXPR_PRESETS))),
            hint=t("expr.preset_unknown.hint"),
        )
    return EXPR_PRESETS[key]
