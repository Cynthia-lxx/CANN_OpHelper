"""cann_ophelper.expr.parse -- Text-to-AST parsing (infix math + LaTeX subset).

Two input dialects are supported:
- plain infix math: ``A + 2 / sigmoid(B) = C``;
- a small LaTeX subset on top of it: ``\\frac{a}{b}`` -> ``(a)/(b)``,
  ``\\cdot``/``\\times`` -> ``*``, ``\\mathrm{sigmoid}`` -> ``sigmoid``.

An optional trailing ``= OutputName`` assigns the expression result to an
output tensor; without it the caller decides the output tensor (wizard/CLI).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from ..i18n import t
from ..model import OpSpecError
from .ast import ExprNode, binary, call, number, ref, unary
from .presets import FUNCTION_ALIASES, INFIX_OP_MAP

__all__ = ["ParsedExpr", "parse_expr", "latex_to_infix", "MAX_EXPR_DEPTH", "MAX_EXPR_NODES"]

#: Hard limits that keep generated kernels and error messages bounded.
MAX_EXPR_DEPTH = 32
MAX_EXPR_NODES = 64

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: LaTeX commands this tool understands (anything else raises a clear error).
_LATEX_KNOWN = frozenset({"frac", "cdot", "times", "left", "right", "mathrm", "operatorname"})

try:  # pragma: no cover - import guard, exercised only when lark is absent
    from lark import Lark, Transformer  # type: ignore
    from lark.exceptions import LarkError  # type: ignore

    _LARK_AVAILABLE = True
    _TRANSFORMER_BASE = Transformer
except Exception:  # pragma: no cover - defensive; user installs lark separately
    _LARK_AVAILABLE = False
    _TRANSFORMER_BASE = object  # type: ignore[misc,assignment]

_GRAMMAR_FILE = None


def _grammar() -> str:
    global _GRAMMAR_FILE
    if _GRAMMAR_FILE is None:
        from pathlib import Path

        _GRAMMAR_FILE = Path(__file__).with_name("grammar.lark").read_text(encoding="utf-8")
    return _GRAMMAR_FILE


def _parser():  # type: ignore[no-untyped-def]
    if not _LARK_AVAILABLE:
        raise OpSpecError(t("expr.parse.lark_missing"), hint=t("expr.parse.lark_missing.hint"))
    return Lark(_grammar(), parser="earley", start="start", lexer="auto")


class _ToAst(_TRANSFORMER_BASE):  # type: ignore[misc,valid-type]
    """Lark tree -> ExprNode (bottom-up; every callback returns an ExprNode)."""

    def number(self, items):  # type: ignore[no-untyped-def]
        raw = str(items[0])
        try:
            value = float(raw)
        except ValueError as exc:
            raise OpSpecError(t("expr.parse.syntax", text=raw, err=exc)) from exc
        if not math.isfinite(value):
            raise OpSpecError(
                t("expr.parse.number_not_finite", value=raw),
                hint=t("expr.parse.number_not_finite.hint"),
            )
        return number(value)

    def var(self, items):  # type: ignore[no-untyped-def]
        return ref(str(items[0]))

    def group(self, items):  # type: ignore[no-untyped-def]
        return items[1]

    def call(self, items):  # type: ignore[no-untyped-def]
        fn = str(items[0])
        args = [x for x in items[1:] if isinstance(x, ExprNode)]
        canonical = FUNCTION_ALIASES.get(fn)
        if canonical is None:
            raise OpSpecError(
                t("expr.parse.unknown_function", name=fn, values=", ".join(sorted(FUNCTION_ALIASES))),
                hint=t("expr.parse.unknown_function.hint"),
            )
        from .ast import CALL_OPS

        if len(args) != CALL_OPS[canonical]:
            raise OpSpecError(
                t(
                    "expr.parse.function_arity",
                    name=canonical,
                    expect=CALL_OPS[canonical],
                    got=len(args),
                ),
                hint=t("expr.parse.function_arity.hint"),
            )
        return call(canonical, *args)

    def neg(self, items):  # type: ignore[no-untyped-def]
        return unary("neg", items[1])

    def pos(self, items):  # type: ignore[no-untyped-def]
        return items[1]

    def unary(self, items):  # type: ignore[no-untyped-def]
        return items[0]

    def product(self, items):  # type: ignore[no-untyped-def]
        return self._fold(items)

    def sum(self, items):  # type: ignore[no-untyped-def]
        return self._fold(items)

    @staticmethod
    def _fold(items):  # type: ignore[no-untyped-def]
        acc = items[0]
        idx = 1
        while idx < len(items):
            op = str(items[idx])
            operand = items[idx + 1]
            acc = binary(INFIX_OP_MAP[op], acc, operand)
            idx += 2
        return acc


# ---------------------------------------------------------------------------
# LaTeX subset -> infix translation (recursive group scanner)
# ---------------------------------------------------------------------------


def latex_to_infix(text: str) -> str:
    """Translate a small LaTeX subset into plain infix text.

    Handles ``\\frac{n}{d}`` (nesting-safe), ``\\cdot``/``\\times``,
    ``\\mathrm{name}``/``\\operatorname{name}`` and ``\\left/\\right`` guards.
    Unsupported backslash commands raise a bilingual OpSpecError; bare ``^``
    power notation is rejected (out of v1 scope).
    """
    return _scan(text, 0, len(text))[0]


def _scan(s: str, i: int, end: int) -> tuple[str, int]:
    out: list[str] = []
    while i < end:
        ch = s[i]
        if ch == "\\":
            i += 1
            if i >= end:
                raise OpSpecError(t("expr.parse.syntax", text=s, err="dangling backslash"))
            if s[i] == ",":
                # escaped comma in maths; treat as plain separator
                out.append(",")
                i += 1
                continue
            j = i
            while j < end and s[j].isalpha():
                j += 1
            cmd = s[i:j]
            i = j
            if cmd == "frac":
                num, i = _read_group(s, i, end)
                den, i = _read_group(s, i, end)
                out.append(f"({_scan(num, 0, len(num))[0]})/({_scan(den, 0, len(den))[0]})")
            elif cmd in ("cdot", "times"):
                out.append("*")
            elif cmd in ("left", "right"):
                # consume the following single (non-space) delimiter
                while i < end and s[i].isspace():
                    i += 1
                if i < end:
                    i += 1
            elif cmd in ("mathrm", "operatorname"):
                content, i = _read_group(s, i, end)
                out.append(_scan(content, 0, len(content))[0])
            else:
                raise OpSpecError(
                    t("expr.latex.unsupported_cmd", cmd=cmd),
                    hint=t("expr.latex.unsupported_cmd.hint"),
                )
        elif ch == "^":
            raise OpSpecError(t("expr.parse.pow_unsupported"), hint=t("expr.parse.pow_unsupported.hint"))
        elif ch == "*" and i + 1 < end and s[i + 1] == "*":
            raise OpSpecError(t("expr.parse.pow_unsupported"), hint=t("expr.parse.pow_unsupported.hint"))
        else:
            out.append(ch)
            i += 1
    return "".join(out), i


def _read_group(s: str, i: int, end: int) -> tuple[str, int]:
    """Read a braced LaTeX group ``{...}`` (nesting-safe). Returns (content, new_i)."""
    while i < end and s[i].isspace():
        i += 1
    if i >= end or s[i] != "{":
        raise OpSpecError(
            t("expr.parse.syntax", text=s, err="expected '{' after LaTeX command"),
            hint=t("expr.latex.brace.hint"),
        )
    depth = 0
    start = i + 1
    j = i
    while j < end:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:j], j + 1
        j += 1
    raise OpSpecError(
        t("expr.parse.syntax", text=s, err="unbalanced '{'"),
        hint=t("expr.latex.brace.hint"),
    )


# ---------------------------------------------------------------------------
# Public parsing entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedExpr:
    """Result of parsing an expression text."""

    tree: ExprNode          # the computed (element-wise) expression
    output: Optional[str]   # output tensor name from "= C", or None


def _validate_limits(node: ExprNode) -> None:
    nodes = 0
    depth = 0
    stack: list[tuple[ExprNode, int]] = [(node, 1)]
    while stack:
        current, level = stack.pop()
        nodes += 1
        depth = max(depth, level)
        for child in current.args:
            stack.append((child, level + 1))
    if depth > MAX_EXPR_DEPTH:
        raise OpSpecError(t("expr.parse.depth_limit", limit=MAX_EXPR_DEPTH))
    if nodes > MAX_EXPR_NODES:
        raise OpSpecError(t("expr.parse.node_limit", limit=MAX_EXPR_NODES))


def _parse_infix(text: str) -> ExprNode:
    stripped = text.strip()
    if not stripped:
        raise OpSpecError(t("expr.parse.empty"), hint=t("expr.parse.empty.hint"))
    source = latex_to_infix(stripped) if "\\" in stripped else stripped
    try:
        tree = _parser().parse(source)
        return _ToAst().transform(tree)  # type: ignore[return-value]
    except OpSpecError:
        raise
    except LarkError as exc:
        raise OpSpecError(
            t("expr.parse.syntax", text=text, err=exc),
            hint=t("expr.parse.syntax.hint"),
        ) from exc


def parse_expr(text: str, *, output: Optional[str] = None) -> ParsedExpr:
    """Parse an element-wise expression, optionally with ``= Output``.

    :param text: e.g. ``A + 2/sigmoid(B) = C`` (infix or LaTeX subset);
    :param output: explicit output name overriding any ``= C`` in ``text``;
    :returns: ParsedExpr(tree, output).
    :raises OpSpecError: On syntax, unknown symbol, output-name or limit errors.
    """
    raw = str(text).strip()
    if not raw:
        raise OpSpecError(t("expr.parse.empty"), hint=t("expr.parse.empty.hint"))

    equals = raw.count("=")
    if equals > 1:
        raise OpSpecError(t("expr.parse.equal_many", text=raw))
    lhs, _, rhs = raw.partition("=")
    result_output = output
    if equals == 1:
        rhs_name = rhs.strip()
        if not _NAME_RE.match(rhs_name):
            raise OpSpecError(
                t("expr.parse.output_invalid", name=rhs_name),
                hint=t("expr.parse.output_invalid.hint"),
            )
        if output is not None and output != rhs_name:
            raise OpSpecError(t("expr.parse.output_mismatch", explicit=output, given=rhs_name))
        result_output = rhs_name

    tree = _parse_infix(lhs)
    _validate_limits(tree)
    return ParsedExpr(tree=tree, output=result_output)
