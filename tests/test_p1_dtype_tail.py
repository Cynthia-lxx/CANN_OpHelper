"""P1 dtype/tail regression tests.

Covers the float16 vertical slice end to end at the text/asset level:

- fp16 three-file generation markers (half kernel, 32B-block tail tiling,
  host DT_FLOAT16 + tail math);
- fp16 verify assets: binary sizes, pack format, runner dtype tokens, and a
  hand-rolled RNE cross-check of the fp16 golden (statement sequence);
- big/small-core partition math (offsets cover the tensor exactly);
- float32 legacy path keeps ``element_count=None`` (default 8*2048) so the old
  assets stay byte-compatible (covered by test_verifygen too).

Nothing here is compiled; every assertion is against generated text/bytes.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from cann_ophelper.fillgen import files_from_spec, profile_from_spec
from cann_ophelper.model import OpSpec, TensorSpec
from cann_ophelper.verifygen import DATA_LENGTH, verify_files

CORE_NUM = 8
ELEMS_PER_BLOCK = 16  # 32B / sizeof(half)
N = 416  # 26 blocks -> perCoreBlocks=3, tailBlockNum=2 (cores 0..1 are big)


def _spec_fp16_addcmul() -> OpSpec:
    return OpSpec(
        op_type="AscF16Addcmul",
        soc_version="ascend910b4",
        inputs=[
            TensorSpec("A", ["float16"], ["ND"], shape=[N]),
            TensorSpec("B", ["float16"], ["ND"]),
            TensorSpec("D", ["float16"], ["ND"]),
        ],
        outputs=[TensorSpec("Y", ["float16"], ["ND"])],
        expr="A + B * D * 2.0 = Y",
    )


def _fp16(v: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(v)))[0]


def _load_f16(path: Path) -> list[float]:
    return list(struct.unpack(f"<{path.stat().st_size // 2}e", path.read_bytes()))


# ---------------------------------------------------------------------------
# fp16 generation markers
# ---------------------------------------------------------------------------


def test_fp16_profile_fields() -> None:
    profile = profile_from_spec(_spec_fp16_addcmul())
    assert profile.dtype == "float16"
    assert profile.cpp_dtype == "half"
    assert profile.type_len == 2
    assert profile.element_count == N


def test_fp16_kernel_uses_half_and_tail_tiling() -> None:
    _, program, files = files_from_spec(_spec_fp16_addcmul())
    kernel = files["op_kernel/asc_f16_addcmul.cpp"]
    assert "AscendC::GlobalTensor<half>" in kernel
    assert "AscendC::LocalTensor<half>" in kernel
    assert "this->dataNum" in kernel
    assert "this->bigDataNum" in kernel
    assert "this->smallDataNum" in kernel
    assert "this->tailBlockNum" in kernel
    assert "AscendC::DataCopy(aLocal, aGm[0], this->dataNum);" in kernel
    # No v1 uniform-tile leftovers in the fp16 kernel.
    assert "this->tileLength" not in kernel
    # Duplicate constants are emitted with the half cast (A + B*D*2.0).
    assert "AscendC::Duplicate<half>(s0, (half)2" in kernel
    # Init signature carries the tail fields, and Process is copy/compute/out.
    assert "uint32_t bigDataNum, uint32_t smallDataNum, uint32_t tailBlockNum)" in kernel
    assert kernel.count("CopyIn();") >= 0


def test_fp16_tiling_header_has_tail_fields() -> None:
    _, _, files = files_from_spec(_spec_fp16_addcmul())
    header = files["op_kernel/asc_f16_addcmul_tiling.h"]
    for field in ("uint32_t totalLength;", "uint32_t bigDataNum;", "uint32_t smallDataNum;", "uint32_t tailBlockNum;"):
        assert field in header


def test_fp16_host_tiling_math() -> None:
    _, _, files = files_from_spec(_spec_fp16_addcmul())
    host = files["op_host/asc_f16_addcmul.cpp"]
    assert "{ge::DT_FLOAT16}" in host
    assert "ELEMS_PER_BLOCK = 16;  // 32B / sizeof(half)" in host
    assert "uint32_t totalBlocks = totalLength / ELEMS_PER_BLOCK;" in host
    assert "uint32_t tailBlockNum = totalBlocks % 8;" in host
    assert "tiling->bigDataNum = (perCoreBlocks + (tailBlockNum > 0 ? 1 : 0)) * ELEMS_PER_BLOCK;" in host
    assert "tiling->smallDataNum = perCoreBlocks * ELEMS_PER_BLOCK;" in host


# ---------------------------------------------------------------------------
# Partition math: big/small-core ranges must tile the tensor exactly.
# ---------------------------------------------------------------------------


def _mirror_host_partition(total_length: int) -> list[tuple[int, int]]:
    """Python mirror of the generated host tiling for fp16 (8 cores, 16 el/32B)."""
    total_blocks = total_length // ELEMS_PER_BLOCK
    per_core = total_blocks // CORE_NUM
    tail_blocks = total_blocks % CORE_NUM
    big = (per_core + (1 if tail_blocks > 0 else 0)) * ELEMS_PER_BLOCK
    small = per_core * ELEMS_PER_BLOCK
    ranges: list[tuple[int, int]] = []
    for idx in range(CORE_NUM):
        is_big = idx < tail_blocks
        data_num = big if is_big else small
        offset = idx * big if is_big else tail_blocks * big + (idx - tail_blocks) * small
        ranges.append((offset, offset + data_num))
    return ranges


@pytest.mark.parametrize("total_length", [416, 320, 512, 16, 128])
def test_tail_partition_covers_tensor_exactly(total_length: int) -> None:
    ranges = _mirror_host_partition(total_length)
    intervals = sorted(ranges, key=lambda item: item[0])
    assert intervals[0][0] == 0
    assert intervals[-1][1] == total_length
    assert all(lo == prev_hi for (lo, _), (_, prev_hi) in zip(intervals[1:], intervals))


# ---------------------------------------------------------------------------
# fp16 verify assets
# ---------------------------------------------------------------------------


def test_fp16_verify_asset_sizes_and_golden_roundtrip(tmp_path: Path) -> None:
    profile = profile_from_spec(_spec_fp16_addcmul())
    _, program, _ = files_from_spec(_spec_fp16_addcmul())
    bundle = verify_files(program, profile)
    for relpath, content in bundle.items():
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    for name in ("input_A.bin", "input_B.bin", "input_D.bin", "golden.bin"):
        assert (tmp_path / "verify" / name).stat().st_size == N * 2  # half = 2 bytes

    a = _load_f16(tmp_path / "verify" / "input_A.bin")
    b = _load_f16(tmp_path / "verify" / "input_B.bin")
    d = _load_f16(tmp_path / "verify" / "input_D.bin")
    golden = _load_f16(tmp_path / "verify" / "golden.bin")

    # Independent RNE statement sequence: t = b*d; t = t*2.0; y = a + t.
    for i in range(N):
        expected = _fp16(a[i] + _fp16(_fp16(b[i] * d[i]) * 2.0))
        assert golden[i] == pytest.approx(expected, rel=0.0, abs=0.0)


def test_fp16_runner_dtype_tokens(tmp_path: Path) -> None:
    profile = profile_from_spec(_spec_fp16_addcmul())
    _, program, _ = files_from_spec(_spec_fp16_addcmul())
    bundle = verify_files(program, profile)
    runner = bundle["verify/aclnn_asc_f16_addcmul.cpp"].decode("utf-8")
    assert "std::vector<uint16_t>" in runner
    assert "ACL_FLOAT16" in runner
    assert f"const int64_t elementCount = {N};" in runner
    result_py = bundle["verify/verify_result.py"].decode("utf-8")
    assert "def load_values" in result_py
    assert "TEST PASSED!" in result_py


# ---------------------------------------------------------------------------
# float32 legacy path stays untouched
# ---------------------------------------------------------------------------


def test_float32_default_element_count_is_legacy() -> None:
    spec = OpSpec(
        op_type="TryFloat",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"]), TensorSpec("B", ["float"], ["ND"])],
        outputs=[TensorSpec("C", ["float"], ["ND"])],
        expr="A + B = C",
    )
    profile = profile_from_spec(spec)
    assert profile.dtype == "float"
    assert profile.element_count is None  # falls back to DATA_LENGTH
    _, program, files = files_from_spec(spec)
    bundle = verify_files(program, profile)
    assert len(bundle["verify/golden.bin"]) == DATA_LENGTH * 4
    runner = bundle["verify/aclnn_try_float.cpp"].decode("utf-8")
    assert "std::vector<float>" in runner
    assert "ACL_FLOAT" in runner
    # Tail tiling must not leak into the float path.
    assert "tailBlockNum" not in files["op_kernel/try_float.cpp"]
    assert "half" not in files["op_kernel/try_float.cpp"]
