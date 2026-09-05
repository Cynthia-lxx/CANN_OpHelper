"""CLI tests for cann-ophelper.

Drives the typer app with ``typer.testing.CliRunner``. The autouse conftest
fixture pins the active i18n language to Chinese (zh); English output is tested
explicitly with ``--lang en``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cann_ophelper import __version__
from cann_ophelper.cli import DEFAULT_OUT, DEFAULT_PROTO, app
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


def invoke(*args: str):
    return runner.invoke(app, list(args))


def test_help_lists_commands() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    assert "gen-msopgen" in result.output
    assert "render" in result.output
    assert "--lang" in result.output


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
