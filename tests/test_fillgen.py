"""Tests for the fillgen three-file text generator + profile validation.

focuses on:
- profile_from_spec v1 contract (single float output, identifier safety);
- build_three_files shape/content (kernel/tiling/host text);
- validate_program rejecting undeclared references.

The full-shell apply path (inspect_project/apply) is tested separately in
test_fillop_apply.py using the AscTry shell fixture.
"""

from __future__ import annotations

import pytest

from cann_ophelper.expr.ast import binary, call, number, ref
from cann_ophelper.expr.lower import lower_expr
from cann_ophelper.fillgen import build_three_files, profile_from_spec
from cann_ophelper.model import OpSpec, OpSpecError, TensorSpec


def spec_2in(op_type="AscTry", out="C", soc="ascend910b4"):
    return OpSpec(
        op_type=op_type,
        soc_version=soc,
        inputs=[
            TensorSpec("A", ["float"], ["ND"]),
            TensorSpec("B", ["float"], ["ND"]),
        ],
        outputs=[TensorSpec(out, ["float"], ["ND"])],
    )


def spec_1in(op_type="AscSigmoid", out="C"):
    return OpSpec(
        op_type=op_type,
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"])],
        outputs=[TensorSpec(out, ["float"], ["ND"])],
    )


# ---------------------------------------------------------------------------
# profile_from_spec validation
# ---------------------------------------------------------------------------


def test_profile_from_spec_basic():
    profile = profile_from_spec(spec_2in())
    assert profile.entry == "asc_try"
    assert profile.op_type == "AscTry"
    assert profile.op_pascal == "AscTry"
    assert profile.kernel_class == "KernelAscTry"
    assert profile.tiling_struct == "AscTryTilingData"
    assert profile.tiling_guard == "ASC_TRY_TILING_H"
    assert [r.name for r in profile.inputs] == ["A", "B"]
    assert [r.name for r in profile.outputs] == ["C"]
    assert profile.soc == "ascend910b"


def test_profile_entry_from_snake_op_type():
    profile = profile_from_spec(
        OpSpec(
            op_type="my_op_abc",
            soc_version="ascend910b4",
            inputs=[TensorSpec("A", ["float"], ["ND"])],
            outputs=[TensorSpec("C", ["float"], ["ND"])],
        )
    )
    assert profile.entry == "my_op_abc"
    assert profile.op_pascal == "MyOpAbc"
    assert profile.tiling_struct == "MyOpAbcTilingData"
    assert profile.op_type == "my_op_abc"


def test_profile_rejects_multiple_outputs():
    spec = OpSpec(
        op_type="TryTwo",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"])],
        outputs=[TensorSpec("C", ["float"], ["ND"]), TensorSpec("D", ["float"], ["ND"])],
    )
    with pytest.raises(OpSpecError, match="单输出"):
        profile_from_spec(spec)


def test_profile_rejects_no_inputs():
    spec = OpSpec(op_type="TryNo", soc_version="ascend910b4", outputs=[TensorSpec("C", ["float"], ["ND"])])
    with pytest.raises(OpSpecError, match="缺少输入"):
        profile_from_spec(spec)


def test_profile_rejects_float16_v1():
    spec = OpSpec(
        op_type="TryF16",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float16"], ["ND"]), TensorSpec("B", ["float16"], ["ND"])],
        outputs=[TensorSpec("C", ["float16"], ["ND"])],
    )
    with pytest.raises(OpSpecError, match="v1 仅支持 float"):
        profile_from_spec(spec)


def test_profile_rejects_mixed_dtypes():
    spec = OpSpec(
        op_type="TryMix",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"]), TensorSpec("B", ["float16"], ["ND"])],
        outputs=[TensorSpec("C", ["float"], ["ND"])],
    )
    with pytest.raises(OpSpecError, match="dtype 不一致"):
        profile_from_spec(spec)


def test_profile_rejects_identifier_collision():
    spec = OpSpec(
        op_type="TryColl",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"]), TensorSpec("a", ["float"], ["ND"])],
        outputs=[TensorSpec("C", ["float"], ["ND"])],
    )
    with pytest.raises(OpSpecError, match="碰撞"):
        profile_from_spec(spec)


def test_profile_rejects_bad_tensor_name():
    spec = OpSpec(
        op_type="TryName",
        soc_version="ascend910b4",
        inputs=[TensorSpec("1X", ["float"], ["ND"])],
        outputs=[TensorSpec("C", ["float"], ["ND"])],
    )
    with pytest.raises(OpSpecError):
        profile_from_spec(spec)


def test_soc_family_cleanup():
    assert profile_from_spec(spec_2in(soc="ai_core-ascend910b4")).soc == "ascend910b"
    assert profile_from_spec(spec_2in(soc="ascend910b4")).soc == "ascend910b"
    assert profile_from_spec(spec_2in(soc="ascend910b")).soc == "ascend910b"


# ---------------------------------------------------------------------------
# build_three_files content
# ---------------------------------------------------------------------------


def test_simple_add_files_shape():
    program = lower_expr(binary("add", ref("A"), ref("B")), "C")
    files = build_three_files(profile_from_spec(spec_2in()), program)
    assert set(files) == {"op_kernel/asc_try.cpp", "op_kernel/asc_try_tiling.h", "op_host/asc_try.cpp"}
    for text in files.values():
        assert text.endswith("\n")
        assert text.startswith("/**\n* Copyright")
    # kernel text
    kernel = files["op_kernel/asc_try.cpp"]
    assert "class KernelAscTry {" in kernel
    assert "AscendC::TPipe pipe;" in kernel  # regression: pipe must be a member
    assert "AscendC::Add(cLocal, aLocal, bLocal, this->tileLength);" in kernel
    assert "REGISTER_TILING_DEFAULT(AscTryTilingData);" in kernel
    assert "void asc_try(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)" in kernel
    # tiling header
    tiling = files["op_kernel/asc_try_tiling.h"]
    assert "struct AscTryTilingData {" in tiling
    assert "uint32_t totalLength;" in tiling
    assert "uint32_t tileNum;" in tiling
    # host
    host = files["op_host/asc_try.cpp"]
    assert 'this->Input("A")' in host
    assert 'this->Input("B")' in host
    assert 'this->Output("C")' in host
    assert '.DataType({ge::DT_FLOAT})' in host
    assert '.Format({ge::FORMAT_ND});' in host
    assert '.AddConfig("ascend910b");' in host
    assert "OP_ADD(AscTry);" in host


def test_sigmoid_files_no_scratch():
    program = lower_expr(call("sigmoid", ref("A")), "C")
    files = build_three_files(profile_from_spec(spec_1in()), program)
    kernel = files["op_kernel/asc_sigmoid.cpp"]
    assert "AscendC::Sigmoid(cLocal, aLocal, this->tileLength);" in kernel
    assert "TBuf" not in kernel
    # single input GM param list
    assert "void asc_sigmoid(GM_ADDR a, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)" in kernel


def test_nested_expression_scratch_buffers():
    tree = binary("add", ref("A"), binary("div", number(2), call("sigmoid", ref("B"))))
    program = lower_expr(tree, "C")
    files = build_three_files(profile_from_spec(spec_2in()), program)
    kernel = files["op_kernel/asc_try.cpp"]
    assert "pipe.InitBuffer(tmp0, this->tileLength * sizeof(float));" in kernel
    assert "pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));" in kernel
    assert "AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0;" in kernel
    assert "AscendC::Duplicate<float>(s0, (float)2, this->tileLength);" in kernel
    assert "AscendC::Sigmoid(s1, bLocal, this->tileLength);" in kernel
    assert "AscendC::Div(s0, s0, s1, this->tileLength);" in kernel
    assert "AscendC::Add(cLocal, aLocal, s0, this->tileLength);" in kernel


def test_three_input_kernel_naming():
    spec = OpSpec(
        op_type="TryThree",
        soc_version="ascend910b4",
        inputs=[TensorSpec("A", ["float"], ["ND"]), TensorSpec("B", ["float"], ["ND"]), TensorSpec("C", ["float"], ["ND"])],
        outputs=[TensorSpec("D", ["float"], ["ND"])],
    )
    program = lower_expr(binary("add", ref("A"), binary("mul", ref("B"), ref("C"))), "D")
    files = build_three_files(profile_from_spec(spec), program)
    kernel = files["op_kernel/try_three.cpp"]
    assert "class KernelTryThree {" in kernel
    assert "void try_three(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR workspace, GM_ADDR tiling)" in kernel
    assert "AscendC::Mul" in kernel
    assert "AscendC::Add(cLocal, aLocal, s0, this->tileLength);" in kernel or "AscendC::Add(" in kernel
    assert "inQueueC" in kernel


def test_constant_left_nested_scratch_semantics():
    # (2 - A) + B: the numeric 2 fills scratch s0 which then feeds Sub.
    tree = binary("add", binary("sub", number(2), ref("A")), ref("B"))
    program = lower_expr(tree, "C")
    files = build_three_files(profile_from_spec(spec_2in()), program)
    kernel = files["op_kernel/asc_try.cpp"]
    assert "AscendC::Duplicate<float>(s0, (float)2, this->tileLength);" in kernel
    assert "AscendC::Sub(s0, s0, aLocal, this->tileLength);" in kernel
    assert "AscendC::Add(cLocal, s0, bLocal, this->tileLength);" in kernel


def test_neg_lowers_to_muls_statement_text():
    # -A + B == C
    tree = binary("add", _neg(ref("A")), ref("B"))
    program = lower_expr(tree, "C")
    files = build_three_files(profile_from_spec(spec_2in()), program)
    kernel = files["op_kernel/asc_try.cpp"]
    assert "AscendC::Muls(s0, aLocal, (float)-1, this->tileLength);" in kernel
    assert "AscendC::Add(cLocal, s0, bLocal, this->tileLength);" in kernel


def _neg(arg):
    from cann_ophelper.expr.ast import unary

    return unary("neg", arg)


def test_validate_rejects_undeclared_dst():
    profile = profile_from_spec(spec_2in())
    program = lower_expr(binary("add", ref("A"), ref("B")), "ZZ")
    with pytest.raises(OpSpecError, match="未声明"):
        build_three_files(profile, program)


def test_validate_rejects_undeclared_input():
    profile = profile_from_spec(spec_2in())
    program = lower_expr(binary("add", ref("A"), ref("X")), "C")
    with pytest.raises(OpSpecError, match="未声明"):
        build_three_files(profile, program)
