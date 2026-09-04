"""OpSpec/TensorSpec/AttrSpec 模型与校验的最小测试集。"""

import pytest

from cann_ophelper.model import (
    AttrSpec,
    OpSpec,
    OpSpecError,
    ParamType,
    TensorSpec,
)


def _add_spec(**overrides):
    """构造一个合法的 Add 元信息；overrides 直接覆盖 OpSpec 顶层字段。"""
    base = dict(
        op_type="AddCustomTemplate",
        soc_version="ascend910b1",
        inputs=[
            TensorSpec(name="x", type=["float16", "float"], format=["ND", "ND"]),
            TensorSpec(name="y", type=["float16", "float"], format=["ND", "ND"]),
        ],
        outputs=[TensorSpec(name="z", type=["float16", "float"], format=["ND", "ND"])],
    )
    base.update(overrides)
    return OpSpec(**base)


class TestOpSpecValidation:
    def test_defaults_are_sane(self):
        spec = _add_spec()
        spec.validate()
        assert spec.op_type == "AddCustomTemplate"
        assert spec.soc_version == "ascend910b1"
        assert spec.language == "cpp"
        assert spec.attrs == []
        assert spec.tiling == {}

    def test_missing_outputs_rejected(self):
        spec = _add_spec(outputs=[])
        with pytest.raises(OpSpecError, match="outputs 不能为空"):
            spec.validate()

    def test_empty_op_type_rejected(self):
        spec = _add_spec(op_type="   ")
        with pytest.raises(OpSpecError, match="不能为空"):
            spec.validate()

    def test_illegal_op_type_rejected(self):
        spec = _add_spec(op_type="Add Op")
        with pytest.raises(OpSpecError, match="不合法"):
            spec.validate()

    def test_empty_soc_rejected(self):
        spec = _add_spec(soc_version="")
        with pytest.raises(OpSpecError, match="soc_version 不能为空"):
            spec.validate()

    def test_soc_can_carry_prefix(self):
        # 允许带前缀的 soc 值（不报错，交给 msopgen.format_soc_for_msopgen 去重）
        spec = _add_spec(soc_version="ai_core-ascend910b1")
        spec.validate()

    def test_op_name_snake_case_convention(self):
        # 官方 op_type → 文件名/函数名采用 snake_case（AddCustomTemplate → add_custom_template）
        assert _add_spec().op_name_snake == "add_custom_template"


class TestTensorSpecValidation:
    def test_duplicate_names_rejected(self):
        spec = _add_spec(
            inputs=[TensorSpec(name="x"), TensorSpec(name="x")],
            outputs=[TensorSpec(name="z")],
        )
        with pytest.raises(OpSpecError, match="重复"):
            spec.validate()

    def test_mismatched_type_format_arrays_rejected(self):
        # 两边都 >1 且不等长才算不匹配（单侧为 1 会被广播，属合法简写）
        tensor = TensorSpec(name="x", type=["float16", "float", "int32"], format=["ND", "ND"])
        with pytest.raises(OpSpecError, match="长度"):
            tensor.validate()

    def test_bad_dtype_rejected(self):
        tensor = TensorSpec(name="x", type=["float16", "fp99"])
        with pytest.raises(OpSpecError, match="dtype 集合"):
            tensor.validate()

    def test_bad_format_rejected(self):
        tensor = TensorSpec(name="x", type=["float16"], format=["FANCY"])
        with pytest.raises(OpSpecError, match="format 集合"):
            tensor.validate()

    def test_singleton_side_is_broadcast(self):
        # 只给多 dtype、省略 format → format 自动广播；反之亦然
        a = TensorSpec(name="a", type=["float16", "float"], format="ND")
        a.validate()
        assert a.format == ["ND", "ND"]
        b = TensorSpec(name="b", type="float16", format=["ND", "NZ"])
        b.validate()
        assert b.type == ["float16", "float16"]

    def test_bad_param_type_rejected(self):
        tensor = TensorSpec(name="x", type=["float16"], param_type="sometimes")
        with pytest.raises(OpSpecError, match="param_type"):
            tensor.validate()

    def test_scalar_strings_normalized_to_lists(self):
        tensor = TensorSpec(name="x", type="float", format="ND")
        tensor.validate()
        assert tensor.type == ["float"]
        assert tensor.format == ["ND"]
        assert tensor.dtypes == ["float"]  # 便捷只读属性


class TestAttrSpecValidation:
    def test_attr_required_fields(self):
        attr = AttrSpec(name="dst_type", type="int", value=0)
        attr.validate()
        assert attr.param_type == ParamType.REQUIRED.value

    def test_attr_duplicate_with_tensor_rejected(self):
        spec = _add_spec(
            attrs=[AttrSpec(name="z", type="int")],  # z 是输出名，应冲突
        )
        with pytest.raises(OpSpecError, match="不得与输入/输出张量重名"):
            spec.validate()


class TestSerializationRoundTrip:
    def test_to_dict_order_is_stable(self):
        spec = _add_spec(description="加法")
        d = spec.to_dict()
        # 顶层字段序保持稳定，便于 diff
        assert list(d)[:2] == ["op_type", "soc_version"]
        assert "inputs" in d and "outputs" in d
        # 默认 language=cpp 不输出，空 attrs/tiling 不输出
        assert "language" not in d
        assert "attrs" not in d
        assert "tiling" not in d

    def test_from_dict_roundtrip(self):
        spec = _add_spec(description="加法")
        clone = OpSpec.from_dict(spec.to_dict())
        assert clone == spec  # dataclass 值相等（source/repr 字段不影响比较）
