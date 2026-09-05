"""Tests for verifygen: deterministic expected data + aclnn runner bundle.

The bundle is purely textual/binary assets generated for the cloud; nothing is
compiled or executed here. Assertions cover:

- determinism of the generated data;
- binary sizes / formats (float32 LE, DATA_LENGTH elements);
- golden correctness for a simple add (independent elementwise re-compute);
- aclnn runner / run script / verify_result.py content markers grounded in the
  official sample;
- write_verify_assets actually creates exactly the promised verify/ files and
  is idempotent.

These tests build the lowering plan from AST factories and need no lark.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from cann_ophelper.expr.ast import binary, ref
from cann_ophelper.expr.lower import lower_expr
from cann_ophelper.fillgen import profile_from_spec
from cann_ophelper.model import OpSpec, TensorSpec
from cann_ophelper.verifygen import (
    DATA_LENGTH,
    DATA_SEED,
    verify_files,
    write_verify_assets,
)


def spec_2in(expr: str = "A + B = C") -> OpSpec:
    return OpSpec(
        op_type="AscTry",
        soc_version="ascend910b4",
        inputs=[
            TensorSpec("A", ["float"], ["ND"]),
            TensorSpec("B", ["float"], ["ND"]),
        ],
        outputs=[TensorSpec("C", ["float"], ["ND"])],
        expr=expr,
    )


def _load_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    assert len(raw) % 4 == 0
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _add_program_and_profile():
    profile = profile_from_spec(spec_2in())
    program = lower_expr(binary("add", ref("A"), ref("B")), "C")
    return program, profile


def test_verify_files_are_deterministic(tmp_path) -> None:
    program, profile = _add_program_and_profile()
    first = verify_files(program, profile)
    second = verify_files(program, profile)
    assert set(first) == set(second)
    for relpath in first:
        assert first[relpath] == second[relpath]
    assert DATA_SEED == 20260905


def test_input_and_golden_files_layout() -> None:
    program, profile = _add_program_and_profile()
    files = verify_files(program, profile)

    # A/B inputs + golden + three text assets = 6 files.
    assert set(files) == {
        "verify/input_A.bin",
        "verify/input_B.bin",
        "verify/golden.bin",
        "verify/aclnn_asc_try.cpp",
        "verify/run_verify.sh",
        "verify/verify_result.py",
    }
    for name in ("input_A.bin", "input_B.bin", "golden.bin"):
        assert len(files[f"verify/{name}"]) == DATA_LENGTH * 4


def test_golden_matches_elementwise_add(tmp_path) -> None:
    program, profile = _add_program_and_profile()
    files = verify_files(program, profile)

    data_dir = tmp_path / "verify"
    data_dir.mkdir(parents=True)
    for rel, content in files.items():
        (tmp_path / rel).write_bytes(content)

    a = _load_f32(data_dir / "input_A.bin")
    b = _load_f32(data_dir / "input_B.bin")
    golden = _load_f32(data_dir / "golden.bin")
    assert len(a) == len(b) == len(golden) == DATA_LENGTH
    for i in range(0, DATA_LENGTH, 1024):  # spot-check every 1024th element
        assert golden[i] == _f32(a[i] + b[i])


def test_runner_source_markers() -> None:
    program, profile = _add_program_and_profile()
    runner = verify_files(program, profile)["verify/aclnn_asc_try.cpp"].decode("utf-8")
    assert '#include "acl/acl.h"' in runner
    assert '#include "aclnn_asc_try.h"' in runner
    assert "aclnnAscTryGetWorkspaceSize(inputA, inputB, outputC, &workspaceSize, &executor);" in runner
    assert "aclnnAscTry(workspaceAddr, workspaceSize, executor, stream);" in runner
    assert 'WriteBinFile(dataDir + "/output.bin", resultData)' in runner
    assert 'ReadBinFile(dataDir + "/input_A.bin", inputAHostData)' in runner
    assert 'ReadBinFile(dataDir + "/input_B.bin", inputBHostData)' in runner
    assert "aclrtMemcpy" in runner
    assert runner.count("\r") == 0  # LF only, cloud-friendly


def test_run_script_uses_lf_and_markers() -> None:
    program, profile = _add_program_and_profile()
    run = verify_files(program, profile)["verify/run_verify.sh"].decode("utf-8")
    assert "#!/usr/bin/env bash" in run
    assert "bash build.sh" in run
    assert 'custom_opp_*.run' in run
    assert "aclnn_asc_try.cpp" in run
    assert "execute_op" in run
    assert "verify_result.py" in run
    assert run.count("\r") == 0


def test_run_script_has_soc_alias_fallback() -> None:
    # Cloud Lab (910B family, CANN 9.0.0) reports socVersion ascend910_93 while the
    # op installs under ascend910b -> NNOP lookup fails with 161001 unless the
    # installed dirs are mirrored under the device soc name before running.
    program, profile = _add_program_and_profile()
    run = verify_files(program, profile)["verify/run_verify.sh"].decode("utf-8")
    assert "ascend910b" in run
    assert "ascend910_93" in run
    assert 'cp -r "${KERNEL_ROOT}/ascend910b" "${KERNEL_ROOT}/ascend910_93"' in run
    assert 'cp -r "${KERNEL_ROOT}/config/ascend910b" "${KERNEL_ROOT}/config/ascend910_93"' in run
    # alias block is emitted after the install step, before running the op
    assert run.index("Deploying operator package") < run.index("ascend910_93") < run.index("[4/4]")
    assert run.count("\r") == 0


def test_verify_result_py_is_stdlib_only() -> None:
    program, profile = _add_program_and_profile()
    verify_py = verify_files(program, profile)["verify/verify_result.py"].decode("utf-8")
    assert "import os" in verify_py
    assert "import struct" in verify_py
    assert "TEST PASSED" in verify_py
    assert "import numpy" not in verify_py


def test_write_verify_assets_creates_bundle_and_is_idempotent(tmp_path) -> None:
    program, profile = _add_program_and_profile()
    project = tmp_path / "project"
    project.mkdir()

    written = write_verify_assets(project, program, profile)
    assert written == sorted(verify_files(program, profile))

    first_snapshot = {
        path.relative_to(project).as_posix(): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert set(first_snapshot) == set(verify_files(program, profile))

    # Re-running overwrites deterministically; no extra files accumulate.
    write_verify_assets(project, program, profile)
    second_snapshot = {
        path.relative_to(project).as_posix(): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    assert first_snapshot == second_snapshot
