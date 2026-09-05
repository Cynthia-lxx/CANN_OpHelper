"""cann_ophelper.verifygen -- Deterministic expected data + one-key cloud verify.

``fill-op`` writes the three kernel files and then drops a self-contained
``verify/`` asset bundle into the same msopgen project:

- ``input_<TENSOR>.bin``  -- one deterministic float32 file per declared input
  (fixed random seed, values in (-1, 1), count = 8 * 2048 so the official
  Add-teaching tiling assumption holds: 8 cores x equal blocks);
- ``golden.bin``          -- expected output computed by *interpreting the same
  lowered ExprProgram* that generated the kernel (per-statement float32
  rounding), i.e. kernel and expectation share one semantics;
- ``aclnn_<entry>.cpp``   -- aclnn single-op host runner generated from the
  msopgen op profile; mirrors the official minimal sample API calls
  (DOC ``09_course_practice/src/09.01_testcase/testcase_6/aclnn_test.cpp``);
- ``run_verify.sh``       -- one-key: build + deploy the op -> compile the
  runner -> run on device -> numeric compare (docs/expr-rules.md section 3);
- ``verify_result.py``    -- pure-stdlib comparator with the official tolerance
  ``rtol=atol=1e-3`` (no numpy needed on the cloud box).

Nothing here is ever compiled or executed locally; every AscendC/aclnn call is
copied from the official patterns locked in docs/expr-rules.md.
"""

from __future__ import annotations

import math
import random
import struct
from pathlib import Path
from typing import Mapping, Sequence

from .fillgen import FileProfile, TensorRef
from .i18n import t
from .model import OpSpecError

__all__ = [
    "DATA_SEED",
    "DATA_LENGTH",
    "verify_files",
    "write_verify_assets",
]

#: Deterministic data seed and element count (8 cores x 2048 elements each).
DATA_SEED = 20260905
DATA_LENGTH = 8 * 2048

#: Official compare tolerance (verify_result.py mirrors np.isclose(rtol/atol=1e-3)).
_ATOL = 1e-3
_RTOL = 1e-3

_FLOAT32 = struct.Struct("<f")


def _f32(value: float) -> float:
    """Round a Python float to IEEE-754 float32 (kernel precision)."""
    return _FLOAT32.unpack(_FLOAT32.pack(float(value)))[0]


def _pascal_var(name: str) -> str:
    """Deterministic C++ identifier suffix for a tensor name (a -> A)."""
    return name[:1].upper() + name[1:]


def _interp_program(program, tensors: Mapping[str, Sequence[float]], out_name: str) -> list[float]:
    """Interpret the lowered statement plan over float vectors.

    Executes the exact same ordered statement sequence the kernel runs, one
    element at a time, rounding every statement result to float32 -- the same
    semantics the NPU vector ops have at the official comparison tolerance.
    """
    # Valid symbol buffers = declared inputs plus the output tensor: the
    # lowered plan may write directly into the output symbol (e.g. constants
    # are dup'ed into the destination slot) exactly like the kernel does.
    names = set(tensors) | {out_name}
    slots: dict[str, list[float]] = {name: list(values) for name, values in tensors.items()}
    if out_name not in slots:
        slots[out_name] = [0.0] * DATA_LENGTH
    scratch: list[list[float]] = [[0.0] * DATA_LENGTH for _ in range(program.scratch_count)]

    def buf(slot: str) -> list[float]:
        if slot in names:
            return slots[slot]
        return scratch[int(slot[1:])]

    for stmt in program.statements:
        dst = buf(stmt.dst)
        if stmt.op == "dup":
            scalar = _f32(stmt.scalar if stmt.scalar is not None else 0.0)
            for i in range(DATA_LENGTH):
                dst[i] = scalar
            continue
        srcs = [buf(s) for s in stmt.srcs]
        for i in range(DATA_LENGTH):
            if stmt.op == "neg":
                value = -srcs[0][i]
            elif stmt.op == "add":
                value = srcs[0][i] + srcs[1][i]
            elif stmt.op == "sub":
                value = srcs[0][i] - srcs[1][i]
            elif stmt.op == "mul":
                value = srcs[0][i] * srcs[1][i]
            elif stmt.op == "div":
                value = srcs[0][i] / srcs[1][i]
            elif stmt.op == "sigmoid":
                value = 1.0 / (1.0 + math.exp(-srcs[0][i]))
            elif stmt.op == "exp":
                value = math.exp(srcs[0][i])
            elif stmt.op == "abs":
                value = abs(srcs[0][i])
            else:  # pragma: no cover - guarded by fillgen.validate_program rules
                raise OpSpecError(t("verifygen.err.op_unsupported", op=stmt.op))
            dst[i] = _f32(value)
    return slots[out_name]


def _binary_data(values: Sequence[float]) -> bytes:
    return b"".join(_FLOAT32.pack(value) for value in values)


def _inputs_binary(refs: Sequence[TensorRef]) -> dict[str, bytes]:
    """Deterministic input data files for every declared input tensor."""
    rng = random.Random(DATA_SEED)
    out: dict[str, bytes] = {}
    for ref in refs:
        values = [_f32(rng.uniform(-1.0, 1.0)) for _ in range(DATA_LENGTH)]
        out[f"verify/input_{ref.name}.bin"] = _binary_data(values)
    return out


# ---------------------------------------------------------------------------
# aclnn single-op runner (generated from the profile; official sample shape)
# ---------------------------------------------------------------------------


def _runner_source(profile: FileProfile) -> str:
    pascal = profile.op_pascal
    entry = profile.entry
    inputs = profile.inputs
    output = profile.outputs[0]
    in_c = [_pascal_var(ref.name) for ref in inputs]
    out_c = _pascal_var(output.name)
    n = DATA_LENGTH

    lines: list[str] = []

    def add(*items: str) -> None:
        lines.extend(items)

    add(
        "// Generated by CANN_OpHelper verifygen -- aclnn single-op host runner.",
        "// API pattern mirrors the official minimal sample",
        "// (DOC 09_course_practice/src/09.01_testcase/testcase_6/aclnn_test.cpp).",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <fstream>",
        "#include <numeric>",
        "#include <string>",
        "#include <vector>",
        "",
        '#include "acl/acl.h"',
        f'#include "aclnn_{entry}.h"',
        "",
        "#define SUCCESS 0",
        "#define FAILED 1",
        "",
        "#define CHECK_RET(cond, return_expr) \\",
        "    do {                             \\",
        "        if (!(cond)) {               \\",
        "            return_expr;             \\",
        "        }                            \\",
        "    } while (0)",
        "",
        "#define LOG_PRINT(message, ...)         \\",
        "    do {                                \\",
        "        printf(message, ##__VA_ARGS__); \\",
        "    } while (0)",
        "",
        "int64_t GetShapeSize(const std::vector<int64_t> &shape)",
        "{",
        "    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());",
        "}",
        "",
        "template <typename T>",
        "bool ReadBinFile(const std::string &fileName, std::vector<T> &data)",
        "{",
        "    std::ifstream input(fileName, std::ios::binary);",
        "    if (!input) {",
        '        LOG_PRINT("open %s failed\\n", fileName.c_str());',
        "        return false;",
        "    }",
        "    input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));",
        "    if (input.gcount() != static_cast<std::streamsize>(data.size() * sizeof(T))) {",
        '        LOG_PRINT("read %s failed, expected %zu bytes, got %lld bytes\\n", fileName.c_str(),',
        "                  data.size() * sizeof(T),",
        "                  static_cast<long long>(input.gcount()));",
        "        return false;",
        "    }",
        "    return true;",
        "}",
        "",
        "template <typename T>",
        "bool WriteBinFile(const std::string &fileName, const std::vector<T> &data)",
        "{",
        "    std::ofstream output(fileName, std::ios::binary);",
        "    if (!output) {",
        '        LOG_PRINT("open %s failed\\n", fileName.c_str());',
        "        return false;",
        "    }",
        "    output.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));",
        "    return output.good();",
        "}",
        "",
        "int Init(int32_t deviceId, aclrtStream *stream)",
        "{",
        "    auto ret = aclInit(nullptr);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\\n", ret); return FAILED);',
        "    ret = aclrtSetDevice(deviceId);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\\n", ret); return FAILED);',
        "    ret = aclrtCreateStream(stream);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\\n", ret); return FAILED);',
        "    return SUCCESS;",
        "}",
        "",
        "template <typename T>",
        "int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,",
        "                    aclDataType dataType, aclTensor **tensor)",
        "{",
        "    auto size = GetShapeSize(shape) * sizeof(T);",
        "    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\\n", ret); return FAILED);',
        "",
        "    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\\n", ret); return FAILED);',
        "",
        "    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),",
        "                              shape.size(), *deviceAddr);",
        '    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\\n"); return FAILED);',
        "    return SUCCESS;",
        "}",
        "",
        "void DestroyResources(const std::vector<void *> &tensors, const std::vector<void *> &deviceAddrs, aclrtStream stream,",
        "                      int32_t deviceId, void *workspaceAddr = nullptr)",
        "{",
        "    for (size_t i = 0; i < std::min(tensors.size(), deviceAddrs.size()); i++) {",
        "        if (tensors[i] != nullptr) {",
        "            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));",
        "        }",
        "        if (deviceAddrs[i] != nullptr) {",
        "            aclrtFree(deviceAddrs[i]);",
        "        }",
        "    }",
        "    if (workspaceAddr != nullptr) {",
        "        aclrtFree(workspaceAddr);",
        "    }",
        "    aclrtDestroyStream(stream);",
        "    aclrtResetDevice(deviceId);",
        "    aclFinalize();",
        "}",
        "",
    )

    ptr_list = ", ".join([*(f"input{c}" for c in in_c), f"output{out_c}"])
    addr_list = ", ".join(
        [*(f"input{c}DeviceAddr" for c in in_c), f"output{out_c}DeviceAddr"]
    )
    # Declare every pointer up front so any failure path can run one cleanup().
    add(
        "int main(int argc, char **argv)",
        "{",
        '    const std::string dataDir = argc > 1 ? argv[1] : ".";',
        "",
        "    int32_t deviceId = 0;",
        "    aclrtStream stream = nullptr;",
        "    auto ret = Init(deviceId, &stream);",
        '    CHECK_RET(ret == SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\\n", ret); return FAILED);',
        "",
        f"    const int64_t elementCount = {n};",
        f"    const std::vector<int64_t> tensorShape = {{{n}}};",
        "",
        f"    std::vector<float> output{out_c}HostData(elementCount, 0.0f);",
    )
    for c in in_c:
        add(f"    std::vector<float> input{c}HostData(elementCount);")
    for c in in_c:
        add(
            f'    CHECK_RET(ReadBinFile(dataDir + "/input_{_input_file_name(inputs, c)}.bin", input{c}HostData),',
            "              LOG_PRINT(\"read input file failed\\n\"); return FAILED);",
        )
    add("")
    for c in in_c:
        add(f"    void *input{c}DeviceAddr = nullptr;")
        add(f"    aclTensor *input{c} = nullptr;")
    add(f"    void *output{out_c}DeviceAddr = nullptr;")
    add(f"    aclTensor *output{out_c} = nullptr;")
    add("    void *workspaceAddr = nullptr;")
    add(
        "",
        "    auto cleanup = [&]() {",
        f"        DestroyResources({{{ptr_list}}}, {{{addr_list}}}, stream, deviceId, workspaceAddr);",
        "    };",
        "",
    )

    for ref, c in zip(inputs, in_c):
        add(
            f"    ret = CreateAclTensor(input{c}HostData, tensorShape, &input{c}DeviceAddr, ACL_FLOAT, &input{c});",
            f'    CHECK_RET(ret == SUCCESS, LOG_PRINT("create input tensor {ref.name} failed. ERROR: %d\\n", ret);',
            "              cleanup(); return FAILED);",
        )
    add(
        f"    ret = CreateAclTensor(output{out_c}HostData, tensorShape, &output{out_c}DeviceAddr, ACL_FLOAT, &output{out_c});",
        '    CHECK_RET(ret == SUCCESS, LOG_PRINT("create output tensor failed. ERROR: %d\\n", ret);',
        "              cleanup(); return FAILED);",
        "",
        "    uint64_t workspaceSize = 0;",
        "    aclOpExecutor *executor = nullptr;",
        f"    ret = aclnn{pascal}GetWorkspaceSize({ptr_list}, &workspaceSize, &executor);",
        f'    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnn{pascal}GetWorkspaceSize failed. ERROR: %d\\n", ret);',
        "              cleanup(); return FAILED);",
        "",
        "    if (workspaceSize > 0) {",
        "        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);",
        '        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\\n", ret);',
        "                  cleanup(); return FAILED);",
        "    }",
        "",
        f"    ret = aclnn{pascal}(workspaceAddr, workspaceSize, executor, stream);",
        f'    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnn{pascal} failed. ERROR: %d\\n", ret);',
        "              cleanup(); return FAILED);",
        "",
        "    ret = aclrtSynchronizeStream(stream);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\\n", ret);',
        "              cleanup(); return FAILED);",
        "",
        f"    std::vector<float> resultData(elementCount);",
        f"    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), output{out_c}DeviceAddr,",
        "                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);",
        '    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\\n", ret);',
        "              cleanup(); return FAILED);",
        "",
        '    CHECK_RET(WriteBinFile(dataDir + "/output.bin", resultData),',
        "              LOG_PRINT(\"write output.bin failed\\n\"); cleanup(); return FAILED);",
        "",
        "    cleanup();",
        '    LOG_PRINT("write output to %s/output.bin success\\n", dataDir.c_str());',
        "    return SUCCESS;",
        "}",
        "",
    )
    return "\n".join(lines)


def _input_file_name(inputs: Sequence[TensorRef], c: str) -> str:
    """Look up the original tensor name that maps to identifier suffix ``c``."""
    for ref in inputs:
        if _pascal_var(ref.name) == c:
            return ref.name
    raise OpSpecError(t("verifygen.err.internal", tag=c))  # pragma: no cover


def _verify_py_source() -> str:
    return """\
#!/usr/bin/env python3
# Generated by CANN_OpHelper verifygen -- pure-stdlib numeric comparator.
# Mirrors the official verify_result.py (rtol=atol=1e-3, equal_nan=True) but
# uses only the Python standard library so the cloud box needs no numpy.
import os
import struct
import sys
from pathlib import Path

ATOL = 1e-3
RTOL = 1e-3


def load_f32(path):
    raw = Path(path).read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit("FAILED! bad file size: %s (%d bytes)" % (path, len(raw)))
    count = len(raw) // 4
    return list(struct.unpack("<%df" % count, raw))


def close(a, b):
    if a != a and b != b:  # NaN == NaN for both
        return True
    return abs(a - b) <= ATOL + RTOL * abs(b)


def verify_result():
    golden = load_f32("golden.bin")
    output = load_f32("output.bin")
    if len(golden) != len(output):
        print("FAILED! size mismatch, output=%d, golden=%d" % (len(output), len(golden)))
        return False
    diff = [abs(a - b) for a, b in zip(output, golden)]
    bad = [i for i, ok in enumerate(close(a, b) for a, b in zip(output, golden)) if not ok]
    if not bad:
        print("TEST PASSED!")
        return True
    print("TEST FAILED!")
    print("mismatch count: %d/%d" % (len(bad), len(golden)))
    print("max abs diff: %g" % (max(diff) if diff else 0.0))
    for idx in bad[:10]:
        print("index: %d, output: %g, golden: %g, diff: %g" % (idx, output[idx], golden[idx], diff[idx]))
    return False


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    sys.exit(0 if verify_result() else 1)
"""


def _run_sh_source(profile: FileProfile) -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by CANN_OpHelper verifygen -- one-key build + run + numeric compare.\n"
        "# Run from the cloud CANN environment at the project root:\n"
        "#     bash verify/run_verify.sh\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        'VERIFY_DIR="$(pwd)"\n'
        'PROJECT_DIR="$(dirname "${VERIFY_DIR}")"\n'
        'cd "${PROJECT_DIR}"\n'
        '\n'
        'echo "[1/4] Building operator (bash build.sh)"\n'
        "bash build.sh\n"
        '\n'
        "RUN_PKG=\"$(find build_out -maxdepth 1 -name 'custom_opp_*.run' | head -n 1 || true)\"\n"
        'if [ -z "${RUN_PKG}" ]; then\n'
        '    echo "FAILED: no build_out/custom_opp_*.run package was produced" >&2\n'
        "    exit 1\n"
        "fi\n"
        'echo "[2/4] Deploying operator package"\n'
        '"${RUN_PKG}" --install-path="${HOME}/"\n'
        '\n'
        '# Cloud Lab (910B family, CANN 9.0.0) quirk: the NPU reports socVersion\n'
        '# "ascend910_93" while the op is compiled/installed under "ascend910b", so\n'
        '# NNOP cannot find kernel/config/ascend910_93/binary_info_config.json and the\n'
        '# first aclnn call fails with 161001. The two .o sets are interchangeable\n'
        '# (same DAV_2201 arch), so mirror the installed dirs under the device soc\n'
        '# name. Done on every run because build.sh + reinstall wipes the copy.\n'
        'KERNEL_ROOT="${HOME}/vendors/customize/op_impl/ai_core/tbe/kernel"\n'
        'if [ -d "${KERNEL_ROOT}/ascend910b" ] && [ ! -d "${KERNEL_ROOT}/ascend910_93" ]; then\n'
        '    cp -r "${KERNEL_ROOT}/ascend910b" "${KERNEL_ROOT}/ascend910_93"\n'
        '    echo "      soc alias: kernel/ascend910b -> kernel/ascend910_93"\n'
        'fi\n'
        'if [ -d "${KERNEL_ROOT}/config/ascend910b" ] && [ ! -d "${KERNEL_ROOT}/config/ascend910_93" ]; then\n'
        '    cp -r "${KERNEL_ROOT}/config/ascend910b" "${KERNEL_ROOT}/config/ascend910_93"\n'
        '    echo "      soc alias: kernel/config/ascend910b -> kernel/config/ascend910_93"\n'
        'fi\n'
        '\n'
        'echo "[3/4] Compiling the aclnn single-op runner"\n'
        'g++ -I"${VERIFY_DIR}" -I"${ASCEND_TOOLKIT_HOME}/include" -I"${HOME}/vendors/customize/op_api/include" \\\n'
        '    -L"${ASCEND_TOOLKIT_HOME}/lib64" -L"${HOME}/vendors/customize/op_api/lib" \\\n'
        f'    "${{VERIFY_DIR}}/aclnn_{profile.entry}.cpp" -lcust_opapi -lnnopbase -lacl_rt -o "${{VERIFY_DIR}}/execute_op"\n'
        'set +u\n'
        'source "${HOME}/vendors/customize/bin/set_env.bash"\n'
        'set -u\n'
        '\n'
        'echo "[4/4] Running the op and comparing output.bin with golden.bin"\n'
        '"${VERIFY_DIR}/execute_op" "${VERIFY_DIR}"\n'
        'python3 "${VERIFY_DIR}/verify_result.py"\n'
    )


def verify_files(program, profile: FileProfile) -> dict[str, bytes]:
    """Build the complete ``verify/`` asset bundle for one expression op.

    :returns: ``{relative_path: bytes}``; binary files are float32 LE, text
        files are LF-encoded UTF-8 so the bundle can be uploaded as-is.
    """
    files = _inputs_binary(profile.inputs)

    # Expected output: interpret the *same* lowered program the kernel runs.
    tensors: dict[str, list[float]] = {}
    for key, raw in files.items():  # key == "verify/input_<NAME>.bin"
        name = key[len("verify/input_"):-len(".bin")]
        tensors[name] = list(struct.unpack(f"<{DATA_LENGTH}f", raw))
    golden = _interp_program(program, tensors, profile.outputs[0].name)

    files["verify/golden.bin"] = _binary_data(golden)
    files[f"verify/aclnn_{profile.entry}.cpp"] = _runner_source(profile).encode("utf-8")
    files["verify/run_verify.sh"] = _run_sh_source(profile).encode("utf-8")
    files["verify/verify_result.py"] = _verify_py_source().encode("utf-8")
    return files


def write_verify_assets(project_dir: Path, program, profile: FileProfile) -> list[str]:
    """Write the verify bundle under ``project_dir/verify/``.

    :returns: sorted list of relative paths that were (re)written.
    """
    root = Path(project_dir)
    if not root.is_dir():
        raise OpSpecError(t("fill_op.err.not_a_shell"), hint=t("fill_op.err.not_a_shell.hint"))
    files = verify_files(program, profile)
    written: list[str] = []
    for relpath, content in files.items():
        target = root / relpath
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as exc:
            raise OpSpecError(
                t("verifygen.err.write_fail", path=target, reason=exc.strerror or str(exc)),
                hint=t("verifygen.err.write_fail.hint"),
            ) from exc
        written.append(relpath)
    return sorted(written)
