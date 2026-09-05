"""Tests for cann_ophelper.wizard (presets, parsers, guided collection).

``collect_op_spec`` is exercised without a TTY: a scripted answer queue replaces
``typer.prompt``, so validation/re-ask loops can be asserted directly. The
autouse conftest pins the i18n language to zh.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from cann_ophelper.model import OpSpec, OpSpecError
from cann_ophelper.wizard import (
    PRESETS,
    collect_op_spec,
    parse_dtype_csv,
    parse_format_csv,
    parse_int,
    parse_shape_text,
    resolve_preset,
)


class _FakeAsker:
    """Scripted prompt double: yields the next canned answer per call.

    Mirrors ``typer.prompt(question, default=...)`` behaviour: when no answer is
    left the prompt "ends" (EOFError). Empty answers are forwarded verbatim, and
    ``_prompt`` turns them into the default -- exactly like pressing Enter.
    """

    def __init__(self, answers: List[str]) -> None:
        self.answers = list(answers)

    def __call__(self, question: str, default: Optional[str] = None) -> str:
        if not self.answers:
            raise EOFError("no more scripted answers")
        return self.answers.pop(0)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def test_presets_exported() -> None:
    assert PRESETS and "add" in PRESETS


def test_add_preset_is_valid_and_official_shaped() -> None:
    spec = resolve_preset("add")
    assert isinstance(spec, OpSpec)
    spec.validate()  # raises on any problem
    assert spec.op_type == "AddCustomTemplate"
    assert [t.name for t in spec.inputs] == ["x", "y"]
    assert [t.name for t in spec.outputs] == ["z"]
    assert spec.inputs[0].type == ["float16", "float"]
    assert spec.inputs[0].format == ["ND", "ND"]


def test_preset_factories_return_fresh_objects() -> None:
    first = resolve_preset("add")
    second = resolve_preset("add")
    assert first is not second
    assert first.inputs[0] is not second.inputs[0]
    first.inputs[0].shape = None  # mutate one instance; the other is unaffected
    assert second.inputs[0].shape == [8, 2048]


def test_resolve_unknown_preset_raises() -> None:
    with pytest.raises(OpSpecError) as excinfo:
        resolve_preset("bogus")
    message = str(excinfo.value)
    assert "未知预设" in message
    assert "add" in message


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_int_accepts_bounds() -> None:
    assert parse_int("0", 0, 9) == 0
    assert parse_int("9", 1, 9) == 9


@pytest.mark.parametrize("text", ["-1", "10", "abc", ""])
def test_parse_int_rejects_out_of_range(text: str) -> None:
    with pytest.raises(OpSpecError):
        parse_int(text, 1, 9)


def test_parse_shape_text_forms() -> None:
    assert parse_shape_text("") is None
    assert parse_shape_text("8, 2048") == [8, 2048]
    assert parse_shape_text("[1024, 1024]") == [1024, 1024]
    assert parse_shape_text("-1") == [-1]
    assert parse_shape_text("[]") is None


@pytest.mark.parametrize("text", ["8,2048,abc", "x", "1,-2"])
def test_parse_shape_text_rejects_bad_dims(text: str) -> None:
    with pytest.raises(OpSpecError):
        parse_shape_text(text)


def test_parse_dtype_csv_defaults_and_validation() -> None:
    assert parse_dtype_csv("") == ["float"]
    assert parse_dtype_csv(" float16 , float ") == ["float16", "float"]
    with pytest.raises(OpSpecError) as excinfo:
        parse_dtype_csv("float33")
    assert "不受支持" in str(excinfo.value)


def test_parse_format_csv_defaults_and_validation() -> None:
    assert parse_format_csv("") == ["ND"]
    assert parse_format_csv("nd, nchw") == ["ND", "NCHW"]
    with pytest.raises(OpSpecError) as excinfo:
        parse_format_csv("BOGUS")
    assert "不受支持" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Guided collection (scripted answers)
# ---------------------------------------------------------------------------

def test_collect_spec_from_scratch() -> None:
    answers = [
        "AddCustomTemplate",  # op_type (no default -> must type it)
        "ascend910b1",        # soc
        "",                   # description (optional)
        "2",                  # n_inputs
        "1",                  # n_outputs
        # -- input x --
        "x",                  # name
        "",                   # param_type -> required
        "float16,float",      # dtypes
        "ND",                 # formats (single ND broadcast to both dtypes)
        "8, 2048",            # shape
        # -- input y --
        "y",
        "",
        "",                   # dtype -> default float
        "",                   # format -> default ND
        "",                   # shape -> None
        # -- output z --
        "z",
        "",
        "float16,float",
        "ND",
        "1024",               # shape single dim
    ]
    spec = collect_op_spec(asker=_FakeAsker(answers))
    spec.validate()

    assert spec.op_type == "AddCustomTemplate"
    assert spec.soc_version == "ascend910b1"
    x = spec.inputs[0]
    assert x.name == "x" and x.param_type == "required"
    assert x.type == ["float16", "float"]
    assert x.format == ["ND", "ND"]  # broadcast happened
    assert x.shape == [8, 2048]
    y = spec.inputs[1]
    assert y.type == ["float"] and y.format == ["ND"] and y.shape is None
    z = spec.outputs[0]
    assert z.shape == [1024]


def test_collect_reasks_after_invalid_dtype() -> None:
    """A wrong dtype is rejected with a hint, then the question is asked again."""
    answers = [
        "Add",        # op_type
        "ascend910b1",
        "",
        "1",          # one input
        "1",          # one output
        "src",
        "",
        "float33",    # invalid -> hint + re-ask
        "float16",    # valid replacement
        "",
        "",           # shape
        "dst",
        "",
        "",
        "",
        "",
    ]
    spec = collect_op_spec(asker=_FakeAsker(answers))
    assert spec.inputs[0].type == ["float16"]


def test_collect_reasks_after_duplicate_name() -> None:
    answers = [
        "Add",        # op_type
        "ascend910b1",
        "",
        "1",
        "1",
        "x",
        "",
        "float",
        "",
        "",
        "x",          # duplicate -> hint + re-ask
        "dst",
        "",
        "",
        "",
        "",
    ]
    spec = collect_op_spec(asker=_FakeAsker(answers))
    assert [t.name for t in spec.outputs] == ["dst"]


def test_collect_with_add_preset_prefill() -> None:
    """Empty replies accept every prefilled default from the 'add' preset."""
    seed = resolve_preset("add")
    answers = [""] * 20  # accept all defaults (see field count in the wizard)
    spec = collect_op_spec(seed=seed, asker=_FakeAsker(answers))
    spec.validate()
    assert spec.op_type == "AddCustomTemplate"
    assert [t.name for t in spec.inputs] == ["x", "y"]
    assert spec.inputs[0].type == ["float16", "float"]
    assert spec.inputs[0].shape == [8, 2048]
