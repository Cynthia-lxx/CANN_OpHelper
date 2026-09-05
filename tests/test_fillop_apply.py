"""Tests for the fill-op / apply stage (empty-shell inspection + three-file write).

Covers:
- inspect_project: reading a msopgen empty-shell profile (kernel entry, host
  Input/Output names/dtypes, AddConfig soc, tiling struct);
- check_shell / apply error paths (not-a-shell, missing files, entry/tensor/
  dtype/soc mismatch, wrong file set);
- apply writes exactly three files and leaves every other project file
  byte-for-byte untouched (dry-run included);
- fill-op CLI wiring (help + end-to-end dry-run and real write with the
  parser/lower stage stubbed, because lark is installed by the user later).

These tests build the lowering plan from the AST factories directly, so they
never require lark.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cann_ophelper.apply import apply, check_shell, expected_relpaths, inspect_project
from cann_ophelper.cli import app
from cann_ophelper.expr.ast import binary, ref
from cann_ophelper.expr.lower import lower_expr
from cann_ophelper.fillgen import build_three_files, profile_from_spec
from cann_ophelper.model import OpSpec, OpSpecError, TensorSpec
from cann_ophelper.yamlio import dump_op_spec

SHELL = Path(__file__).resolve().parent / "fixtures" / "shell_asc_try"

#: Files the apply stage must never touch.
NON_TARGETS = (
    "build.sh",
    "CMakePresets.json",
    "framework/CMakeLists.txt",
    "test/main.cpp",
)

runner = CliRunner()


def spec_2in(
    op_type: str = "AscTry",
    out: str = "C",
    soc: str = "ascend910b4",
    expr: str = "",
) -> OpSpec:
    return OpSpec(
        op_type=op_type,
        soc_version=soc,
        inputs=[
            TensorSpec("A", ["float"], ["ND"]),
            TensorSpec("B", ["float"], ["ND"]),
        ],
        outputs=[TensorSpec(out, ["float"], ["ND"])],
        expr=expr,
    )


def copy_shell(tmp_path: Path) -> Path:
    target = tmp_path / "shell"
    shutil.copytree(SHELL, target)
    return target


def snapshot(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = path.read_bytes()
    return out


def build_add_files(profile):
    program = lower_expr(binary("add", ref("A"), ref("B")), profile.outputs[0].name)
    return program, build_three_files(profile, program)


# ---------------------------------------------------------------------------
# inspect_project: profile extraction
# ---------------------------------------------------------------------------


def test_inspect_project_reads_shell_profile(tmp_path):
    shell = inspect_project(copy_shell(tmp_path), expected_entry="asc_try")
    assert shell.entry == "asc_try"
    assert shell.op_pascal == "AscTry"
    assert shell.tiling_struct == "TilingDataTemplate"
    assert shell.soc == "ascend910b"
    assert [(t.name, sorted(t.dtypes)) for t in shell.inputs] == [
        ("A", ["float"]),
        ("B", ["float"]),
    ]
    assert [(t.name, sorted(t.dtypes)) for t in shell.outputs] == [("C", ["float"])]


def test_inspect_project_scans_without_expected_entry(tmp_path):
    shell = inspect_project(copy_shell(tmp_path))
    assert shell.entry == "asc_try"
    assert shell.op_pascal == "AscTry"


def test_inspect_project_not_a_shell(tmp_path):
    with pytest.raises(OpSpecError, match="不是 msopgen 空壳工程"):
        inspect_project(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(OpSpecError, match="不是 msopgen 空壳工程"):
        inspect_project(empty)


def test_inspect_project_missing_expected_entry(tmp_path):
    shell_dir = copy_shell(tmp_path)
    with pytest.raises(OpSpecError, match="找不到 msopgen 空壳工程"):
        inspect_project(shell_dir, expected_entry="other_op")


# ---------------------------------------------------------------------------
# check_shell / apply error paths
# ---------------------------------------------------------------------------


def test_check_shell_accepts_matching_profile(tmp_path):
    profile = profile_from_spec(spec_2in())
    shell = check_shell(copy_shell(tmp_path), profile)
    assert shell.entry == profile.entry


def test_check_shell_rejects_other_entry(tmp_path):
    profile = profile_from_spec(spec_2in(op_type="OtherOp"))
    with pytest.raises(OpSpecError, match="找不到 msopgen 空壳工程"):
        check_shell(copy_shell(tmp_path), profile)


def test_check_shell_rejects_tensor_mismatch(tmp_path):
    bad = profile_from_spec(
        OpSpec(
            op_type="AscTry",
            soc_version="ascend910b4",
            inputs=[
                TensorSpec("A", ["float"], ["ND"]),
                TensorSpec("X", ["float"], ["ND"]),
            ],
            outputs=[TensorSpec("C", ["float"], ["ND"])],
        )
    )
    with pytest.raises(OpSpecError, match="张量与 spec 不一致"):
        check_shell(copy_shell(tmp_path), bad)


def test_check_shell_rejects_dtype_mismatch(tmp_path):
    shell_dir = copy_shell(tmp_path)
    host_path = shell_dir / "op_host" / "asc_try.cpp"
    text = host_path.read_text(encoding="utf-8")
    marker = 'this->Input("B")'
    data_type = ".DataType({ge::DT_FLOAT})"
    pos = text.find(marker)
    assert pos != -1
    dpos = text.find(data_type, pos)
    assert dpos != -1
    text = text[:dpos] + ".DataType({ge::DT_FLOAT16})" + text[dpos + len(data_type):]
    host_path.write_text(text, encoding="utf-8", newline="")

    profile = profile_from_spec(spec_2in())
    with pytest.raises(OpSpecError, match="未声明 spec 所需的 dtype"):
        check_shell(shell_dir, profile)


def test_check_shell_rejects_soc_mismatch(tmp_path):
    shell_dir = copy_shell(tmp_path)
    host_path = shell_dir / "op_host" / "asc_try.cpp"
    text = host_path.read_text(encoding="utf-8")
    text = text.replace('.AddConfig("ascend910b")', '.AddConfig("ascend910c")')
    host_path.write_text(text, encoding="utf-8", newline="")

    profile = profile_from_spec(spec_2in())
    with pytest.raises(OpSpecError, match="与 spec 的 ascend910b 不一致"):
        check_shell(shell_dir, profile)


# ---------------------------------------------------------------------------
# apply: three-file write, nothing else touched
# ---------------------------------------------------------------------------


def test_apply_writes_exactly_three_files(tmp_path):
    shell_dir = copy_shell(tmp_path)
    before = snapshot(shell_dir)
    profile = profile_from_spec(spec_2in())
    _program, files = build_add_files(profile)

    written = apply(shell_dir, profile, files)
    assert written == expected_relpaths("asc_try")

    after = snapshot(shell_dir)
    for rel in NON_TARGETS:
        assert after[rel] == before[rel], f"{rel} must stay untouched"
    assert set(after) == set(before)  # no extra files created

    kernel = after["op_kernel/asc_try.cpp"].decode("utf-8")
    assert kernel != before["op_kernel/asc_try.cpp"].decode("utf-8")
    assert "class KernelAscTry {" in kernel
    assert "AscendC::Add(cLocal, aLocal, bLocal, this->tileLength);" in kernel
    assert "REGISTER_TILING_DEFAULT(AscTryTilingData);" in kernel
    assert "void asc_try(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)" in kernel

    tiling = after["op_kernel/asc_try_tiling.h"].decode("utf-8")
    assert "struct AscTryTilingData {" in tiling

    host = after["op_host/asc_try.cpp"].decode("utf-8")
    assert 'this->Input("A")' in host
    assert '.AddConfig("ascend910b");' in host


def test_apply_dry_run_touches_nothing(tmp_path):
    shell_dir = copy_shell(tmp_path)
    before = snapshot(shell_dir)
    profile = profile_from_spec(spec_2in())
    _program, files = build_add_files(profile)

    written = apply(shell_dir, profile, files, dry_run=True)
    assert written == expected_relpaths("asc_try")
    assert snapshot(shell_dir) == before


def test_apply_rejects_wrong_file_set(tmp_path):
    shell_dir = copy_shell(tmp_path)
    profile = profile_from_spec(spec_2in())
    _program, files = build_add_files(profile)
    del files["op_host/asc_try.cpp"]
    with pytest.raises(OpSpecError, match="文件集合与约定不符"):
        apply(shell_dir, profile, files)


# ---------------------------------------------------------------------------
# CLI fill-op wiring
# ---------------------------------------------------------------------------


def _write_expr_spec(tmp_path: Path, name: str = "asctry.yaml") -> Path:
    path = tmp_path / name
    dump_op_spec(spec_2in(expr="A + B = C"), path)
    return path


def test_fill_op_help_lists_args() -> None:
    result = runner.invoke(app, ["fill-op", "--help"])
    assert result.exit_code == 0, result.output
    assert "yaml_path" in result.output
    assert "project_dir" in result.output
    assert "--dry-run" in result.output


def test_fill_op_writes_files_end_to_end(tmp_path, monkeypatch) -> None:
    import cann_ophelper.cli as cli_mod

    def fake_files_from_spec(spec):
        profile = profile_from_spec(spec)
        program, files = build_add_files(profile)
        return profile, program, files

    monkeypatch.setattr(cli_mod, "files_from_spec", fake_files_from_spec)
    yaml_path = _write_expr_spec(tmp_path)
    shell_dir = copy_shell(tmp_path)
    before = snapshot(shell_dir)

    result = runner.invoke(
        app, ["fill-op", str(yaml_path), str(shell_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "表达式：A + B = C" in result.output
    for rel in expected_relpaths("asc_try"):
        assert rel in result.output
        assert "覆盖" in result.output

    after = snapshot(shell_dir)
    for rel in NON_TARGETS:
        assert after[rel] == before[rel], f"{rel} must stay untouched"
    assert "AscendC::Add(" in after["op_kernel/asc_try.cpp"].decode("utf-8")

    # fill-op additionally generated the cloud verify bundle (new files only).
    verify_dir = shell_dir / "verify"
    assert verify_dir.is_dir()
    for asset in ("input_A.bin", "input_B.bin", "golden.bin",
                  "aclnn_asc_try.cpp", "run_verify.sh", "verify_result.py"):
        assert (verify_dir / asset).is_file(), asset


def test_fill_op_dry_run_leaves_files_alone(tmp_path, monkeypatch) -> None:
    import cann_ophelper.cli as cli_mod

    def fake_files_from_spec(spec):
        profile = profile_from_spec(spec)
        program, files = build_add_files(profile)
        return profile, program, files

    monkeypatch.setattr(cli_mod, "files_from_spec", fake_files_from_spec)
    yaml_path = _write_expr_spec(tmp_path)
    shell_dir = copy_shell(tmp_path)
    before = snapshot(shell_dir)

    result = runner.invoke(
        app, ["fill-op", str(yaml_path), str(shell_dir), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "未写盘" in result.output
    assert snapshot(shell_dir) == before


def test_fill_op_rejects_wrong_directory(tmp_path, monkeypatch) -> None:
    import cann_ophelper.cli as cli_mod

    def fake_files_from_spec(spec):
        profile = profile_from_spec(spec)
        program, files = build_add_files(profile)
        return profile, program, files

    monkeypatch.setattr(cli_mod, "files_from_spec", fake_files_from_spec)
    yaml_path = _write_expr_spec(tmp_path)
    not_a_shell = tmp_path / "not_a_shell"
    not_a_shell.mkdir()

    result = runner.invoke(
        app, ["fill-op", str(yaml_path), str(not_a_shell)]
    )
    assert result.exit_code == 1
    assert "不是 msopgen 空壳工程" in result.output
