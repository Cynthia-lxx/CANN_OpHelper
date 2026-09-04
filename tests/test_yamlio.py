"""yamlio：YAML ↔ OpSpec 往返、非法输入错误信息的最小测试集。"""

import pytest

from cann_ophelper.model import OpSpec, OpSpecError, TensorSpec
from cann_ophelper.yamlio import (
    dump_op_spec,
    load_op_spec,
    op_spec_to_yaml_text,
    yaml_text_to_op_spec,
)

ADD_YAML = """\
op_type: AddCustomTemplate
soc_version: ascend910b1
description: 逐元素加法 z = x + y
inputs:
  - name: x
    type: [float16, float]
    format: [ND, ND]
  - name: y
    type: [float16, float]
    format: [ND, ND]
outputs:
  - name: z
    type: [float16, float]
    format: [ND, ND]
"""


class TestLoad:
    def test_parse_minimal_yaml(self):
        spec = yaml_text_to_op_spec(ADD_YAML)
        assert spec.op_type == "AddCustomTemplate"
        assert spec.soc_version == "ascend910b1"
        assert len(spec.inputs) == 2
        assert len(spec.outputs) == 1
        assert spec.outputs[0].name == "z"

    def test_syntax_error_reports_clear_message(self):
        with pytest.raises(OpSpecError, match="YAML 语法错误"):
            yaml_text_to_op_spec("inputs:\n  - name: 'x\n    type: ['bad'\n")  # 引号不配对

    def test_empty_yaml_rejected(self):
        with pytest.raises(OpSpecError, match="YAML 内容为空"):
            yaml_text_to_op_spec("")

    def test_missing_op_type_rejected(self):
        with pytest.raises(OpSpecError, match="op_type"):
            yaml_text_to_op_spec("soc_version: ascend910b1\ninputs: []\noutputs: []\n")

    def test_non_mapping_root_rejected(self):
        with pytest.raises(OpSpecError, match="顶层应为"):
            yaml_text_to_op_spec("- a\n- b\n")

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(OpSpecError, match="文件不存在"):
            load_op_spec(tmp_path / "nope.yaml")

    def test_load_sets_source(self, tmp_path):
        f = tmp_path / "add.yaml"
        f.write_text(ADD_YAML, encoding="utf-8")
        spec = load_op_spec(f)
        assert spec.source == str(f)

    def test_load_dump_roundtrip_on_disk(self, tmp_path):
        f1 = tmp_path / "add.yaml"
        f2 = tmp_path / "roundtrip.yaml"
        f1.write_text(ADD_YAML, encoding="utf-8")
        load_op_spec(f1)  # 合法加载不抛错
        spec = load_op_spec(f1)
        dump_op_spec(spec, f2)
        assert load_op_spec(f2) == spec


class TestDump:
    def test_dump_defaults_are_omitted(self):
        spec = OpSpec(
            op_type="Add",
            soc_version="ascend910b1",
            inputs=[TensorSpec(name="x", type="float16")],
            outputs=[TensorSpec(name="y", type="float16")],
        )
        text = op_spec_to_yaml_text(spec)
        # 默认 language=cpp 不写、空 attrs/tiling 不写、无 description 不写
        assert "language:" not in text
        assert "attrs:" not in text
        assert "tiling:" not in text
        assert "description:" not in text
        # 单元素 type/format 以块式数组输出
        assert "- float16" in text
        assert "- ND" in text

    def test_dump_utf8_unicode_preserved(self):
        spec = yaml_text_to_op_spec(ADD_YAML)  # description 含中文
        text = op_spec_to_yaml_text(spec)
        assert "逐元素加法" in text
