"""Tests for cann_ophelper.proto (OpSpec -> official msopgen prototype JSON).

The golden reference is the official sample prototype JSON shipped with the
local docs (add_custom.json). Structure equivalence (json.loads deep equality)
is the contract: msopgen only parses JSON structure, so text layout may differ
as long as values and key sets match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cann_ophelper.model import AttrSpec, OpSpec, OpSpecError, TensorSpec
from cann_ophelper.proto import dump_prototype_json, prototype_json_text
from cann_ophelper.yamlio import load_op_spec

EXAMPLES_YAML = Path(__file__).resolve().parent.parent / "examples" / "add.yaml"
OFFICIAL_ADD_JSON = (
    Path(__file__).resolve().parents[2]
    / "Documentation_for_Developers"
    / "ascendc_operator_development"
    / "03_intermediate_vector_operator_development"
    / "src"
    / "add_custom.json"
)


def _official_dict() -> dict:
    assert OFFICIAL_ADD_JSON.is_file(), f"missing official golden: {OFFICIAL_ADD_JSON}"
    with OFFICIAL_ADD_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def _spec_with_attrs() -> OpSpec:
    return OpSpec(
        op_type="DummyWithAttr",
        soc_version="ascend910b1",
        inputs=[TensorSpec(name="x")],
        outputs=[TensorSpec(name="y")],
        attrs=[AttrSpec(name="mode", type="int", value=1)],
    )


def test_examples_spec_matches_official_prototype() -> None:
    """The built-in example spec must translate exactly into the official JSON."""
    spec = load_op_spec(EXAMPLES_YAML)
    payload = json.loads(prototype_json_text(spec))
    assert payload == _official_dict()


def test_output_has_no_shape_or_soc_metadata() -> None:
    """shape/soc/description are project metadata, never msopgen inputs."""
    spec = load_op_spec(EXAMPLES_YAML)
    assert spec.inputs[0].shape  # the fixture carries a shape hint...
    text = prototype_json_text(spec)
    assert "shape" not in text
    assert "soc" not in text
    assert "AddCustomTemplate" in text
    assert "float16" in text


def test_key_order_matches_official() -> None:
    spec = load_op_spec(EXAMPLES_YAML)
    payload = json.loads(prototype_json_text(spec))
    operator = payload[0]
    assert list(operator) == ["op", "input_desc", "output_desc"]
    for group in ("input_desc", "output_desc"):
        for entry in operator[group]:
            assert list(entry) == ["name", "param_type", "format", "type"]


def test_dump_writes_readable_json(tmp_path: Path) -> None:
    spec = load_op_spec(EXAMPLES_YAML)
    target = tmp_path / "nested" / "my_op.json"  # parents are created
    returned = dump_prototype_json(spec, target)
    assert returned == target
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == _official_dict()


def test_attrs_are_rejected() -> None:
    with pytest.raises(OpSpecError) as excinfo:
        prototype_json_text(_spec_with_attrs())
    message = str(excinfo.value)
    assert "attrs" in message
    assert "暂不支持" in message


def test_invalid_spec_error_is_passed_through() -> None:
    bad = OpSpec(
        op_type="Good_1",
        soc_version="9bad start",
        inputs=[TensorSpec(name="x")],
        outputs=[TensorSpec(name="y")],
    )
    with pytest.raises(OpSpecError):
        prototype_json_text(bad)


def test_duplicate_tensor_names_are_rejected() -> None:
    spec = OpSpec(
        op_type="Dup",
        inputs=[TensorSpec(name="x"), TensorSpec(name="x")],
        outputs=[TensorSpec(name="y")],
    )
    with pytest.raises(OpSpecError):
        prototype_json_text(spec)
