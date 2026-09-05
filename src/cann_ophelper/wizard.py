"""cann_ophelper.wizard -- Guided interactive collection of an operator spec.

The ``new-op`` workflow uses this module to turn a series of questions (or a
built-in preset) into a validated :class:`OpSpec`, which is then shown back for
confirmation and persisted as YAML by the CLI layer.

Design notes:
- **Pure logic vs prompts**: ``collect_op_spec`` takes an ``asker`` callable
  defaulting to ``typer.prompt``. Tests inject a scripted answer queue instead,
  so collecting can be exercised without an interactive TTY.
- **Parallel arrays**: dtypes and formats are collected as two comma-separated
  lists, mirroring the official prototype JSON (entries with the same index form
  one supported dtype+format pair). A single format is broadcast to every dtype
  by ``TensorSpec`` itself.
- **Presets are factory functions** returning a fresh OpSpec on each call, so
  module import never shares mutable lists across specs.
- Every answer is validated immediately (red hint + re-ask); the assembled spec
  is validated once more before being returned.

All user-facing messages are resolved through ``cann_ophelper.i18n``.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from rich.console import Console
from rich.markup import escape as esc
from typer import prompt as _default_asker

from .i18n import t
from .model import (
    OpSpec,
    OpSpecError,
    ParamType,
    SUPPORTED_DTYPES,
    SUPPORTED_FORMATS,
    TensorSpec,
)

__all__ = [
    "PRESETS",
    "resolve_preset",
    "collect_op_spec",
    "collect_expr_answer",
    "parse_int",
    "parse_shape_text",
    "parse_dtype_csv",
    "parse_format_csv",
]

#: Per-kind key used to build both the label and the storage attribute.
_INPUTS = "inputs"
_OUTPUTS = "outputs"
_TENSOR_KINDS: tuple[str, ...] = (_INPUTS, _OUTPUTS)

_OP_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Fallbacks shown as prompt defaults when no preset supplies a value. typer's
#: prompt re-asks on an empty reply only when there is NO default, so every
#: question that may legitimately accept an empty/fallback answer must carry one.
DEFAULT_SOC = "ascend910b1"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _add_preset() -> OpSpec:
    """Official AddCustomTemplate skeleton (aligned with examples/add.yaml).

    Element-wise z = x + y; the msopgen prototype only needs the tensor
    layout, so this preset doubles as the blueprint for any 2-in/1-out
    element-wise operator (change the op_type and re-render when the Compute
    template family is extended).
    """
    def _tensor(name: str) -> TensorSpec:
        return TensorSpec(
            name=name,
            type=["float16", "float"],
            format=["ND", "ND"],
            shape=[8, 2048],
        )

    return OpSpec(
        op_type="AddCustomTemplate",
        soc_version="ascend910b1",
        description="逐元素加法 z = x + y（官方 AddCustomTemplate 样例骨架）",
        inputs=[_tensor("x"), _tensor("y")],
        outputs=[_tensor("z")],
    )


#: Built-in presets. Values are zero-argument factory functions so every call
#: returns an independent OpSpec (no shared mutable tensor lists).
PRESETS: Dict[str, Callable[[], OpSpec]] = {"add": _add_preset}


def resolve_preset(name: str) -> OpSpec:
    """Resolve a preset name to a fresh OpSpec; raise OpSpecError when unknown."""
    factory = PRESETS.get(str(name).strip().lower())
    if factory is None:
        raise OpSpecError(
            t("wizard.err.unknown_preset", value=name, values=", ".join(sorted(PRESETS)))
        )
    return factory()


# ---------------------------------------------------------------------------
# Parsers (pure; each raises OpSpecError with a localized message)
# ---------------------------------------------------------------------------

def _split_csv(text: str) -> List[str]:
    """Split on commas and drop empty tokens (also trims ``[]`` wrappers)."""
    text = str(text).strip().strip("[] ")
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int(text: str, lo: int, hi: int) -> int:
    """Parse an integer within ``[lo, hi]``."""
    value = str(text).strip()
    try:
        number = int(value)
    except ValueError as exc:
        raise OpSpecError(t("wizard.err.count", lo=lo, hi=hi)) from exc
    if not lo <= number <= hi:
        raise OpSpecError(t("wizard.err.count", lo=lo, hi=hi))
    return number


def parse_shape_text(text: str) -> Optional[List[int]]:
    """Parse a shape hint like ``8, 2048`` (``-1`` for dynamic dims).

    Empty text means "no shape" (None). Entries must be integers >= -1.
    """
    parts = _split_csv(text)
    if not parts:
        return None
    dims: List[int] = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError as exc:
            raise OpSpecError(t("wizard.err.shape")) from exc
        if dim < -1:
            raise OpSpecError(t("wizard.err.shape"))
        dims.append(dim)
    return dims or None


def parse_dtype_csv(text: str) -> List[str]:
    """Parse a comma-separated dtype list; empty text defaults to ``float``."""
    dtypes = _split_csv(text) or ["float"]
    for dtype in dtypes:
        if dtype not in SUPPORTED_DTYPES:
            raise OpSpecError(
                t("wizard.err.dtype", value=dtype, values=", ".join(sorted(SUPPORTED_DTYPES)))
            )
    return dtypes


def parse_format_csv(text: str) -> List[str]:
    """Parse a comma-separated format list (upper-cased); empty -> ``ND``."""
    formats = _split_csv(text) or ["ND"]
    for fmt in formats:
        if fmt.upper() not in SUPPORTED_FORMATS:
            raise OpSpecError(
                t("wizard.err.format", value=fmt, values=", ".join(sorted(SUPPORTED_FORMATS)))
            )
    return [fmt.upper() for fmt in formats]


def _parse_op_type(text: str) -> str:
    value = str(text).strip()
    if not _OP_TYPE_RE.match(value):
        raise OpSpecError(t("wizard.err.op_type", value=value))
    return value


def _parse_soc(text: str) -> str:
    value = str(text).strip()
    if not _SOC_RE.match(value):
        raise OpSpecError(t("wizard.err.soc", value=value))
    return value


def _parse_param_type(text: str) -> str:
    value = str(text).strip().lower() or ParamType.REQUIRED.value
    if value not in (ParamType.REQUIRED.value, ParamType.OPTIONAL.value):
        raise OpSpecError(
            t("check.param_type_invalid", value=value),
            hint=t("check.param_type_invalid.hint"),
        )
    return value


# ---------------------------------------------------------------------------
# Prompting helpers
# ---------------------------------------------------------------------------

def _prompt(asker: Callable[..., str], question: str, default: Optional[str]) -> str:
    """Call the asker and normalize the answer; an empty reply falls back to the default."""
    if default is None:
        raw = asker(question)
    else:
        raw = asker(question, default=default)
    text = str(raw or "").strip()
    return text if text else (default if default is not None else "")


def _ask_until(
    console: Console,
    asker: Callable[..., str],
    question: str,
    default: Optional[str],
    parse: Callable[[str], object],
) -> object:
    """Ask repeatedly until ``parse`` succeeds; red hint on every failure."""
    while True:
        try:
            return parse(_prompt(asker, question, default))
        except OpSpecError as exc:
            console.print(f"[bold red]{esc(t('cli.error.title'))}{esc(str(exc))}[/]")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _format_shape_default(shape: Optional[List[int]]) -> str:
    """Render a shape default; empty text stands for "no shape" (None)."""
    if not shape:
        return ""
    return ", ".join(str(dim) for dim in shape)


def collect_op_spec(
    seed: Optional[OpSpec] = None,
    *,
    console: Optional[Console] = None,
    asker: Callable[..., str] = _default_asker,
) -> OpSpec:
    """Collect an operator spec from a sequence of questions.

    :param seed: Optional preset OpSpec used to prefill every answer (typed
        fields confirm the default with an empty reply). May be None, in which
        case only built-in defaults are offered.
    :param console: Rich console used for validation error hints.
    :param asker: Prompt function with ``asker(question[, default]) -> str``;
        defaults to ``typer.prompt``. Tests inject a scripted answer queue.
    :returns: A validated OpSpec (raises on final validation failure).
    """
    console = console or Console()

    # -- Basics ------------------------------------------------------------
    op_type = _ask_until(console, asker, t("wizard.prompt.op_type"),
                         seed.op_type if seed else None, _parse_op_type)
    soc_version = _ask_until(console, asker, t("wizard.prompt.soc"),
                             seed.soc_version if seed else DEFAULT_SOC, _parse_soc)
    description = _prompt(asker, t("wizard.prompt.desc"),
                          seed.description if seed and seed.description else "")

    # -- Counts ------------------------------------------------------------
    console.print(t("wizard.combo.note"))
    n_inputs = _ask_until(console, asker, t("wizard.prompt.n_inputs"),
                          str(len(seed.inputs)) if seed else "0",
                          lambda v: parse_int(v, 0, 9))
    n_outputs = _ask_until(console, asker, t("wizard.prompt.n_outputs"),
                           str(len(seed.outputs)) if seed else "1",
                           lambda v: parse_int(v, 1, 9))

    # -- Tensors -----------------------------------------------------------
    gathered: Dict[str, List[TensorSpec]] = {_INPUTS: [], _OUTPUTS: []}
    seen_names: set[str] = set()
    for kind in _TENSOR_KINDS:
        count = n_inputs if kind == _INPUTS else n_outputs
        label = t(f"wizard.kind.{kind}")
        prior = (seed.inputs if kind == _INPUTS else seed.outputs) if seed else []
        for index in range(count):
            existing = prior[index] if index < len(prior) else None

            def _parse_name(text: str) -> str:
                name = str(text).strip()
                if not name:
                    raise OpSpecError(t("wizard.err.name_empty"))
                if not _NAME_RE.match(name):
                    raise OpSpecError(
                        t("check.name_invalid", what=label, value=name),
                        hint=t("check.name_invalid.hint"),
                    )
                if name in seen_names:
                    raise OpSpecError(t("wizard.err.name_dup", value=name))
                return name

            name = _ask_until(console, asker,
                              t("wizard.prompt.tensor_name", kind=label, index=index + 1),
                              existing.name if existing else None, _parse_name)
            seen_names.add(str(name))

            param_type = _ask_until(
                console, asker,
                t("wizard.prompt.param_type", name=name),
                existing.param_type if existing else ParamType.REQUIRED.value,
                _parse_param_type,
            )
            dtypes = _ask_until(console, asker,
                                t("wizard.prompt.dtype_csv", name=name),
                                ", ".join(existing.type) if existing else "float",
                                parse_dtype_csv)
            formats = _ask_until(console, asker,
                                 t("wizard.prompt.format_csv", name=name),
                                 ", ".join(existing.format) if existing else "ND",
                                 parse_format_csv)
            shape = _ask_until(console, asker,
                               t("wizard.prompt.shape", name=name),
                               _format_shape_default(existing.shape if existing else None),
                               parse_shape_text)
            gathered[kind].append(
                TensorSpec(name=str(name), type=dtypes, format=formats,
                           param_type=str(param_type), shape=shape)
            )

    spec = OpSpec(
        op_type=str(op_type),
        soc_version=str(soc_version),
        inputs=gathered[_INPUTS],
        outputs=gathered[_OUTPUTS],
        description=str(description or ""),
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------
# Element-wise expression collection (gen-op interactive path)
# ---------------------------------------------------------------------------


def _parse_expr_answer(text: str) -> str:
    """Validate an expression answer; a preset name expands to its expression.

    Empty text means "skip" and is accepted. Raises the bilingual
    ``expr.parse.*`` error via ``parse_expr`` on bad syntax, so the ask-until
    loop can re-ask with a red hint.
    """
    from .expr import EXPR_PRESETS, parse_expr, resolve_preset_expr

    raw = str(text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in EXPR_PRESETS:
        raw = resolve_preset_expr(low)[0]
    # Syntax-only validation; tensor-reference semantics are checked later
    # against the spec by gen-op / fill-op (the spec may not exist yet here).
    parse_expr(raw)
    return raw


def collect_expr_answer(
    console: Optional[Console] = None,
    *,
    asker: Callable[..., str] = _default_asker,
    default: str = "",
) -> str:
    """Ask for an element-wise expression (or a preset name) until it parses.

    Returns the accepted expression text (empty string when the user skips).
    Semantic checks (references belong to the spec inputs, output exists) are
    intentionally left to gen-op / fill-op, which know the tensor layout.
    """
    console = console or Console()
    return _ask_until(console, asker, t("wizard.prompt.expr"), default, _parse_expr_answer)
