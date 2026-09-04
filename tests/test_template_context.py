"""Tests for build_render_context (OpSpec -> render context contract).

Checks that the context exposes exactly the keys the templates need, keeps
spec order, produces deterministic output and fails with field-context errors
when a value cannot be mapped.
"""

from __future__ import annotations

import pytest

from cann_ophelper.model import OpSpec, OpSpecError, TensorSpec
from cann_ophelper.template.context import build_render_context


def _spec(**overrides):
    """An AddCustomTemplate spec: two float16/float inputs x/y, one output z."""
    params = dict(
        op_type="AddCustomTemplate",
        soc_version="ascend910b1",
        inputs=[
            TensorSpec(name="x", type=["float16", "float"], format=["ND", "ND"]),
            TensorSpec(name="y", type=["float16", "float"], format=["ND", "ND"]),
        ],
        outputs=[TensorSpec(name="z", type=["float16", "float"], format=["ND", "ND"])],
    )
    params.update(overrides)
    return OpSpec(**params)


class TestTopLevelKeys:
    def test_all_expected_keys_present(self):
        ctx = build_render_context(_spec())
        for key in (
            "op_type",
            "op_snake",
            "kernel_class",
            "tiling_struct",
            "tiling_guard",
            "tiling_header_file",
            "soc_config",
            "inputs",
            "outputs",
        ):
            assert key in ctx

    def test_derived_identifiers(self):
        ctx = build_render_context(_spec())
        assert ctx["op_type"] == "AddCustomTemplate"
        assert ctx["op_snake"] == "add_custom_template"
        assert ctx["kernel_class"] == "KernelAdd"
        assert ctx["tiling_struct"] == "TilingDataTemplate"
        assert ctx["tiling_guard"] == "ADD_CUSTOM_TEMPLATE_TILING_H"
        assert ctx["tiling_header_file"] == "add_custom_template_tiling.h"
        assert ctx["soc_config"] == "ascend910b"

    def test_tensor_lists_in_spec_order(self):
        ctx = build_render_context(_spec())
        assert [t["name"] for t in ctx["inputs"]] == ["x", "y"]
        assert [t["name"] for t in ctx["outputs"]] == ["z"]


class TestTensorEntry:
    def test_input_token_fields(self):
        ctx = build_render_context(_spec())
        x = ctx["inputs"][0]
        assert x["name"] == "x"
        assert x["cap"] == "X"
        assert x["param_enum"] == "REQUIRED"
        assert x["dtype_alias"] == "dtypeX"
        assert x["macro_alias"] == "DTYPE_X"
        # comma-joined ge enum lists keep the parallel-array order
        assert x["ge_types"] == "ge::DT_FLOAT16, ge::DT_FLOAT"
        assert x["ge_formats"] == "ge::FORMAT_ND, ge::FORMAT_ND"

    def test_optional_param_enum(self):
        ctx = build_render_context(
            _spec(outputs=[TensorSpec(name="z", param_type="optional", type=["float16", "float"])])
        )
        z = ctx["outputs"][0]
        assert z["param_enum"] == "OPTIONAL"

    def test_output_alias_uses_cap(self):
        ctx = build_render_context(_spec())
        z = ctx["outputs"][0]
        assert z["dtype_alias"] == "dtypeZ"
        assert z["macro_alias"] == "DTYPE_Z"


class TestDeterminism:
    def test_two_calls_render_same_context(self):
        assert build_render_context(_spec()) == build_render_context(_spec())


class TestErrorContract:
    def test_unmapped_dtype_raises_with_field_context(self):
        spec = _spec()
        spec.inputs[0].type = ["double", "float"]
        spec.inputs[0].format = ["ND", "ND"]
        with pytest.raises(OpSpecError, match=r"inputs\[0\]\.x\.type\[0\]"):
            build_render_context(spec)

    def test_invalid_spec_rejected_before_mapping(self):
        spec = _spec()
        spec.outputs = []
        with pytest.raises(OpSpecError, match="outputs 不能为空"):
            build_render_context(spec)
