"""CLI tests for cann-ophelper.

Drives the typer app with ``typer.testing.CliRunner``. The autouse conftest
fixture pins the active i18n language to Chinese (zh); English output is tested
explicitly with ``--lang en``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cann_ophelper import __version__
from cann_ophelper.cli import DEFAULT_OUT, DEFAULT_PROTO, app
from cann_ophelper.proto import prototype_json_text
from cann_ophelper.template import render
from cann_ophelper.yamlio import load_op_spec

runner = CliRunner()

EXAMPLES_YAML = Path(__file__).resolve().parent.parent / "examples" / "add.yaml"

#: A minimal spec whose dtype passes the model checks but is not in the
#: template maps (only float16/float are registered) -> render must fail loudly.
UNMAPPED_DTYPE_YAML = """\
op_type: Int32Dup
inputs:
  - name: x
    type: [int32]
    format: [ND]
outputs:
  - name: y
    type: [int32]
    format: [ND]
"""


def invoke(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


def test_help_lists_commands() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    assert "gen-msopgen" in result.output
    assert "new-op" in result.output
    assert "render" in result.output
    assert "quickstart" in result.output
    assert "--lang" in result.output


def test_quickstart_zh_lists_whole_flow() -> None:
    result = invoke("quickstart")
    assert result.exit_code == 0, result.output
    text = result.output
    assert "从零到云端 CANN 工程" in text
    assert "new-op --from add --yes --out add.yaml" in text
    assert "gen-msopgen myop.yaml --proto-out myop.json" in text
    assert "render myop.yaml --out" in text


def test_quickstart_english() -> None:
    result = invoke("--lang", "en", "quickstart")
    assert result.exit_code == 0, result.output
    text = result.output
    assert "From zero to a cloud-ready CANN project" in text
    assert "gen-msopgen myop.yaml --proto-out myop.json" in text


def test_new_op_help_shows_flags() -> None:
    result = invoke("new-op", "--help")
    assert result.exit_code == 0
    assert "--from" in result.output
    assert "--yes" in result.output
    assert "--out" in result.output


def test_gen_msopgen_help_shows_proto_out() -> None:
    result = invoke("gen-msopgen", "--help")
    assert result.exit_code == 0
    assert "--proto-out" in result.output


def test_version_flag() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_invalid_lang_is_usage_error() -> None:
    result = invoke("--lang", "xx", "gen-msopgen", str(EXAMPLES_YAML))
    assert result.exit_code != 0
    assert "choose from zh, en" in result.output


def test_gen_msopgen_defaults_zh() -> None:
    result = invoke("gen-msopgen", str(EXAMPLES_YAML))
    assert result.exit_code == 0, result.output
    text = result.output
    assert "算子元信息概览" in text
    assert "AddCustomTemplate" in text
    assert "msopgen 命令" in text
    assert "msopgen gen -i" in text
    assert "ai_core-ascend910b1" in text
    assert DEFAULT_PROTO in text
    assert DEFAULT_OUT in text
    # tensor tables render name/dtype/format entries
    assert "x" in text and "y" in text and "z" in text
    assert "float16" in text


def test_gen_msopgen_custom_flags() -> None:
    result = invoke(
        "gen-msopgen", str(EXAMPLES_YAML), "--proto", "p.json", "--out", "cloud/out"
    )
    assert result.exit_code == 0, result.output
    assert "msopgen gen -i p.json -c ai_core-ascend910b1 -lan cpp -out cloud/out" in result.output


def test_gen_msopgen_english() -> None:
    result = invoke("--lang", "en", "gen-msopgen", str(EXAMPLES_YAML))
    assert result.exit_code == 0, result.output
    text = result.output
    assert "Operator metadata overview" in text
    assert "msopgen command" in text
    assert "Input tensors" in text


def test_gen_msopgen_missing_file_fails() -> None:
    result = invoke("gen-msopgen", "no_such_file.yaml")
    assert result.exit_code == 1
    assert "文件不存在" in result.output


def test_render_without_out_only_previews(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = invoke("render", str(EXAMPLES_YAML))
    assert result.exit_code == 0, result.output
    text = result.output
    assert "add_custom_template.cpp" in text
    assert "add_custom_template_tiling.h" in text
    assert "op_kernel" in text and "op_host" in text
    assert "未写盘" in text
    assert "out/AddCustomTemplate" in text
    # nothing may be created in the working directory
    assert not (tmp_path / "op_kernel").exists()
    assert not (tmp_path / "op_host").exists()


def test_render_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "proj"
    result = invoke("render", str(EXAMPLES_YAML), "--out", str(out), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "仅预览" in result.output
    assert not out.exists()


def test_render_writes_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "proj"
    result = invoke("render", str(EXAMPLES_YAML), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert "已写盘" in result.output

    expected = render(load_op_spec(EXAMPLES_YAML))
    assert expected  # three produced files
    for relpath, text in expected.items():
        target = out / relpath
        assert target.is_file()
        assert target.read_text(encoding="utf-8") == text
    # a second run overwrites existing files
    second = invoke("render", str(EXAMPLES_YAML), "--out", str(out))
    assert second.exit_code == 0
    assert "覆盖" in second.output


def test_render_unmapped_dtype_fails(tmp_path: Path) -> None:
    yaml_file = tmp_path / "int32.yaml"
    yaml_file.write_text(UNMAPPED_DTYPE_YAML, encoding="utf-8")
    result = invoke("render", str(yaml_file), "--out", str(tmp_path / "proj"))
    assert result.exit_code == 1
    assert "未收录" in result.output


def test_render_missing_file_fails() -> None:
    result = invoke("render", "no_such_file.yaml")
    assert result.exit_code == 1
    assert "文件不存在" in result.output


# ---------------------------------------------------------------------------
# new-op
# ---------------------------------------------------------------------------

#: Number of prompts answered by pressing Enter when the 'add' preset pre-fills
#: every field: 3 basics + 2 counts + 5 per tensor x 3 tensors.
PRESET_PROMPT_COUNT = 3 + 2 + 5 * 3


def test_new_op_from_preset_writes_yaml(tmp_path: Path) -> None:
    target = tmp_path / "add_spec.yaml"
    result = invoke(
        "new-op", "--from", "add", "--yes", "--out", str(target),
        input="\n" * PRESET_PROMPT_COUNT,
    )
    assert result.exit_code == 0, result.output
    assert "已写入算子描述" in result.output
    spec = load_op_spec(target)
    assert spec.op_type == "AddCustomTemplate"
    assert [t.name for t in spec.inputs] == ["x", "y"]
    assert spec.inputs[0].type == ["float16", "float"]


def test_new_op_from_preset_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --out the YAML lands as <snake_case>.yaml in the working dir."""
    monkeypatch.chdir(tmp_path)
    result = invoke(
        "new-op", "--from", "add", "--yes", input="\n" * PRESET_PROMPT_COUNT
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "add_custom_template.yaml").is_file()


def test_new_op_cancel_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "cancelled.yaml"
    # answer all collection prompts, then decline the confirmation
    result = invoke(
        "new-op", "--from", "add", "--out", str(target),
        input="\n" * PRESET_PROMPT_COUNT + "n\n",
    )
    assert result.exit_code == 0, result.output
    assert "已取消" in result.output
    assert not target.exists()


def test_new_op_unknown_preset_fails() -> None:
    result = invoke("new-op", "--from", "bogus", "--yes")
    assert result.exit_code == 1
    assert "未知预设" in result.output
    assert "add" in result.output


def test_new_op_interactive_custom_input(tmp_path: Path) -> None:
    """A fully interactive run: scripted non-default answers, no preset.

    Per tensor the wizard asks, in order: name, param_type, dtype list,
    format list, shape. An empty answer means "keep the default" (or the
    parser's own fallback such as required/ND/float).
    """
    target = tmp_path / "custom.yaml"
    answers = "\n".join(
        [
            "MulSimple",   # op_type
            "ascend910b1",  # soc
            "",             # description
            "2",            # number of inputs
            "1",            # number of outputs
            # input a
            "a", "", "float16", "", "64",
            # input b
            "b", "", "float16", "", "64",
            # output c
            "c", "", "float16", "", "64",
        ]
    )
    result = invoke(
        "new-op", "--yes", "--out", str(target), input=answers + "\n",
    )
    assert result.exit_code == 0, result.output
    spec = load_op_spec(target)
    assert spec.op_type == "MulSimple"
    assert [t.name for t in spec.inputs] == ["a", "b"]
    assert [t.name for t in spec.outputs] == ["c"]
    assert spec.inputs[0].type == ["float16"]
    assert spec.inputs[0].format == ["ND"]
    assert spec.outputs[0].shape == [64]


# ---------------------------------------------------------------------------
# gen-msopgen --proto-out
# ---------------------------------------------------------------------------

def test_gen_msopgen_proto_out_writes_prototype(tmp_path: Path) -> None:
    proto = tmp_path / "nested" / "add_custom.json"
    result = invoke("gen-msopgen", str(EXAMPLES_YAML), "--proto-out", str(proto))
    assert result.exit_code == 0, result.output
    assert "原型 JSON 已写入" in result.output

    spec = load_op_spec(EXAMPLES_YAML)
    expected = json.loads(prototype_json_text(spec))
    assert json.loads(proto.read_text(encoding="utf-8")) == expected


def test_gen_msopgen_without_proto_out_writes_nothing(tmp_path: Path) -> None:
    """Backward compatibility: no --proto-out means no filesystem writes."""
    result = invoke("gen-msopgen", str(EXAMPLES_YAML))
    assert result.exit_code == 0
    assert "原型 JSON 已写入" not in result.output


def test_gen_msopgen_without_proto_warns_demo_sample() -> None:
    """Neither --proto nor --proto-out: warn that -i points at the add demo
    sample, not at the current operator's prototype."""
    result = invoke("gen-msopgen", str(EXAMPLES_YAML))
    assert result.exit_code == 0, result.output
    text = result.output
    assert "内置 Add 演示样例" in text
    assert DEFAULT_PROTO in text
    assert "--proto-out" in text
    # the command still references the demo default, clearly flagged
    assert f"-i {DEFAULT_PROTO}" in text


def test_gen_msopgen_proto_out_repoints_proto_to_export(tmp_path: Path) -> None:
    """--proto-out alone: the msopgen '-i' automatically points at the exported
    file name, so the user never sees a stale demo JSON path."""
    target = tmp_path / "nested" / "sub" / "asc_try.json"
    result = invoke("gen-msopgen", str(EXAMPLES_YAML), "--proto-out", str(target))
    assert result.exit_code == 0, result.output
    text = result.output
    assert "-i asc_try.json" in text
    assert "已自动让命令中的 -i" in text
    assert DEFAULT_PROTO not in text
    assert target.is_file()


def test_gen_msopgen_proto_out_with_explicit_proto_keeps_proto(tmp_path: Path) -> None:
    """Both --proto-out and a *different* explicit --proto: the command keeps
    the explicit -i value and warns about the mismatch."""
    target = tmp_path / "exported.json"
    result = invoke(
        "gen-msopgen", str(EXAMPLES_YAML),
        "--proto", "cloud/proto.json", "--proto-out", str(target),
    )
    assert result.exit_code == 0, result.output
    text = result.output
    assert "-i cloud/proto.json" in text
    assert "不一致" in text
    assert target.is_file()


def test_gen_msopgen_proto_out_english(tmp_path: Path) -> None:
    target = tmp_path / "asc_try.json"
    result = invoke(
        "--lang", "en", "gen-msopgen", str(EXAMPLES_YAML),
        "--proto-out", str(target),
    )
    assert result.exit_code == 0, result.output
    text = result.output
    assert "-i asc_try.json" in text
    assert "pointed the command '-i' at the file exported" in text.lower()
