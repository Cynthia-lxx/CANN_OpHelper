"""cann_ophelper.fillgen -- Three-file text generator for expression kernels.

Given a lowering plan (``ExprProgram``) plus a naming profile (read from a
msopgen empty shell or derived from the op spec), produces the three text
files that replace the corresponding files inside a msopgen project:

- ``op_kernel/<entry>.cpp``     full kernel (official teaching style);
- ``op_kernel/<entry>_tiling.h`` tiling data struct;
- ``op_host/<entry>.cpp``        host (Tiling/InferShape/OpDef) mirroring the
                                 msopgen skeleton layout.

v1 constraints enforced here: single output, single dtype ``float`` (DT_FLOAT,
ND format), all tensors sharing the same type/format lists, and the official
Add-teaching tiling assumption (total length divisible by the core count).
These are validated before any text is generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .expr import ExprProgram, lower_expr, parse_expr
from .expr.ast import fmt_num
from .i18n import t
from .model import OpSpec, OpSpecError

__all__ = [
    "TensorRef",
    "FileProfile",
    "profile_from_spec",
    "build_three_files",
    "files_from_spec",
]

#: ASCEND Open Software license block (kept byte-identical with official outputs).
LICENSE = (
    "/**\n"
    "* Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
    "* This program is free software, you can redistribute it and/or modify it under the terms and conditions of\n"
    "* CANN Open Software License Agreement Version 2.0 (the \"License\").\n"
    "* Please refer to the License for details. You may not use this file except in compliance with the License.\n"
    "* THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n"
    "* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
    "* See LICENSE in the root of the software repository for the full text of the License.\n"
    "*/"
)

#: Canonical CANN dtype/format label maps (v1 subset, docs/expr-rules.md).
_DTYPE_GE = {"float": "ge::DT_FLOAT", "float16": "ge::DT_FLOAT16"}
_FORMAT_GE = {"ND": "ge::FORMAT_ND"}

_OP_FN = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "sigmoid": "Sigmoid",
    "exp": "Exp",
    "abs": "Abs",
}

_BUFFER_NUM = 1
_QUEUE_DEPTH = 1


@dataclass(frozen=True)
class TensorRef:
    """One tensor as it appears in the generated host/kernel."""

    name: str
    dtype: str = "float"   # canonical dtype label: 'float' (v1)
    dtype_ge: str = "ge::DT_FLOAT"
    format_ge: str = "ge::FORMAT_ND"


@dataclass(frozen=True)
class FileProfile:
    """Naming + dtype profile for the three generated files."""

    op_type: str                    # e.g. "AscTry"
    entry: str                      # kernel entry + file stem, e.g. "asc_try"
    tiling_struct: str              # e.g. "AscTryTilingData"
    inputs: tuple[TensorRef, ...]
    outputs: tuple[TensorRef, ...]
    soc: str = "ascend910b"
    block_dim: int = 8
    cpp_dtype: str = "float"        # concrete C++ scalar type (v1: float)

    @property
    def op_pascal(self) -> str:
        """PascalCase class name (original op_type when it is already Pascal)."""
        return _pascal_of(self.op_type, self.entry)

    @property
    def kernel_class(self) -> str:
        return f"Kernel{self.op_pascal}"

    @property
    def tiling_guard(self) -> str:
        return f"{self.entry.upper()}_TILING_H"


def _pascal(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _pascal_of(op_type: str, entry: str) -> str:
    """Prefer the user's op_type when it is already PascalCase (e.g. ``AscTry``);
    otherwise derive it from the snake-case entry (``asc_try`` -> ``AscTry``)."""
    candidate = str(op_type).strip()
    if (
        candidate
        and candidate[0].isupper()
        and "_" not in candidate
        and candidate.replace("_", "a").isalnum()
    ):
        return candidate
    return _pascal(entry)


def _snake_of(op_type: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(op_type):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _clean_soc(soc: str) -> str:
    """Reduce a msopgen soc label to its OpDef platform family.

    ``ai_core-ascend910b4`` / ``ascend910b4`` -> ``ascend910b`` (matches the
    msopgen-generated skeleton, which registers e.g. ``AddConfig("ascend910b")``).
    """
    base = str(soc or "").strip()
    if base.startswith("ai_core-"):
        base = base[len("ai_core-"):]
    head = ""
    for ch in base:
        if ch.isdigit() or ch.isalpha():
            head += ch
        else:
            break
    while head and head[-1].isdigit():
        head = head[:-1]
    return head or "ascend910b"


def _tensor_ref(name: str, dtypes: Sequence[str], formats: Sequence[str]) -> TensorRef:
    dtype = str(dtypes[0]).strip().lower() if dtypes else "float"
    if dtype not in _DTYPE_GE:
        raise OpSpecError(
            t("fillgen.err.dtype_unsupported", value=dtype, values=", ".join(sorted(_DTYPE_GE))),
            hint=t("fillgen.err.dtype_unsupported.hint"),
        )
    fmt = str(formats[0]).strip() if formats else "ND"
    if fmt not in _FORMAT_GE:
        raise OpSpecError(
            t("fillgen.err.format_unsupported", value=fmt, values=", ".join(sorted(_FORMAT_GE)))
        )
    return TensorRef(
        name=str(name),
        dtype=dtype,
        dtype_ge=_DTYPE_GE[dtype],
        format_ge=_FORMAT_GE[fmt],
    )


def profile_from_spec(spec: OpSpec) -> FileProfile:
    """Build a FileProfile from an OpSpec (used by gen-op / fill-op).

    Validates the v1 contract (single dtype float, ND, one output, identifier
    safety, no ambiguous identifier collisions).
    """
    if not spec.outputs or len(spec.outputs) != 1:
        raise OpSpecError(t("fillgen.err.single_output"), hint=t("fillgen.err.single_output.hint"))
    if not spec.inputs:
        raise OpSpecError(t("fillgen.err.need_input"), hint=t("fillgen.err.need_input.hint"))

    entry = _snake_of(spec.op_type)
    if not entry or not all(c.isalnum() or c == "_" for c in entry):
        raise OpSpecError(t("fillgen.err.op_type_invalid", value=spec.op_type))

    inputs = tuple(
        _tensor_ref(tens.name, tens.type, tens.format) for tens in spec.inputs
    )
    outputs = tuple(
        _tensor_ref(tens.name, tens.type, tens.format) for tens in spec.outputs
    )

    # All tensors must share the same dtype (DTYPE_X single-float contract).
    dtype_set = {ref.dtype for ref in (*inputs, *outputs)}
    if len(dtype_set) != 1:
        raise OpSpecError(t("fillgen.err.dtype_uniform", values=", ".join(sorted(dtype_set))))
    if {"float"} != dtype_set:
        raise OpSpecError(
            t("fillgen.err.dtype_v1", values=", ".join(sorted(dtype_set))),
            hint=t("fillgen.err.dtype_v1.hint"),
        )

    bases = [t.name[:1].lower() + t.name[1:] for t in (*inputs, *outputs)]
    if len(set(bases)) != len(bases):
        raise OpSpecError(t("fillgen.err.ident_collision"))
    for ref in (*inputs, *outputs):
        if not ref.name or not ref.name.replace("_", "a").isalnum() or not ref.name[0].isalpha():
            raise OpSpecError(t("fillgen.err.name_invalid", value=ref.name))

    return FileProfile(
        op_type=spec.op_type,
        entry=entry,
        tiling_struct=_pascal_of(spec.op_type, entry) + "TilingData",
        inputs=inputs,
        outputs=outputs,
        soc=_clean_soc(spec.soc_version),
        block_dim=8,
        cpp_dtype="float",
    )


# ---------------------------------------------------------------------------
# Token helpers (shared by kernel generation)
# ---------------------------------------------------------------------------


def tensor_var(name: str) -> str:
    """C++ local/global base name for a tensor (A -> a)."""
    return name[:1].lower() + name[1:]


def _all_tensor_names(profile: FileProfile) -> set[str]:
    return {ref.name for ref in (*profile.inputs, *profile.outputs)}


def _var_for(slot: str, names: set[str]) -> str:
    """Map a plan slot to the C++ variable holding that tensor/buffer."""
    if slot in names:
        return f"{tensor_var(slot)}Local"
    return slot  # scratch slots are already s<i>


# ---------------------------------------------------------------------------
# Kernel text
# ---------------------------------------------------------------------------


def _statement_lines(stmt, names: set[str]) -> str:
    dst = _var_for(stmt.dst, names)
    if stmt.op == "dup":
        assert stmt.scalar is not None
        return f"        AscendC::Duplicate<float>({dst}, (float){fmt_num(stmt.scalar)}, this->tileLength);"
    if stmt.op == "neg":
        src = _var_for(stmt.srcs[0], names)
        return f"        AscendC::Muls({dst}, {src}, (float)-1, this->tileLength);"
    fn = _OP_FN[stmt.op]
    srcs = ", ".join(_var_for(s, names) for s in stmt.srcs)
    return f"        AscendC::{fn}({dst}, {srcs}, this->tileLength);"


def _scratch_indices(program: ExprProgram) -> list[int]:
    out: set[int] = set()
    for stmt in program.statements:
        for slot in (stmt.dst, *stmt.srcs):
            if slot.startswith("s") and slot[1:].isdigit():
                out.add(int(slot[1:]))
    return sorted(out)


def _build_kernel(profile: FileProfile, program: ExprProgram) -> str:
    names = _all_tensor_names(profile)
    scratch_idx = _scratch_indices(program)
    entry = profile.entry
    cls = profile.kernel_class
    st = profile.tiling_struct

    gm_params = ", ".join(tensor_var(r.name) for r in (*profile.inputs, *profile.outputs))

    # ---- method bodies -----------------------------------------------------
    inits: list[str] = []
    gms: list[str] = []
    queues: list[str] = []
    members: list[str] = []
    copy_in: list[str] = []
    compute_heads: list[str] = []
    compute_tails: list[str] = []
    stmts: list[str] = []
    copy_out: list[str] = []

    inits.append("        this->blockLength = totalLength / AscendC::GetBlockNum();")
    inits.append("        this->tileNum = tileNum;")
    inits.append("        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;")

    for ref in profile.inputs:
        base = tensor_var(ref.name)
        inits.append(
            f"        {base}Gm.SetGlobalBuffer((__gm__ float *){base} + this->blockLength * AscendC::GetBlockIdx(), "
            f"this->blockLength);"
        )
        inits.append(f"        pipe.InitBuffer(inQueue{ref.name}, BUFFER_NUM, this->tileLength * sizeof(float));")
        queues.append(f"    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueue{ref.name};")
        gms.append(f"    AscendC::GlobalTensor<float> {base}Gm;")
        copy_in.append(f"        AscendC::LocalTensor<float> {base}Local = inQueue{ref.name}.AllocTensor<float>();")
        copy_in.append(
            f"        AscendC::DataCopy({base}Local, {base}Gm[progress * this->tileLength], this->tileLength);"
        )
        copy_in.append(f"        inQueue{ref.name}.EnQue({base}Local);")
        compute_heads.append(f"        AscendC::LocalTensor<float> {base}Local = inQueue{ref.name}.DeQue<float>();")

    for ref in profile.outputs:
        base = tensor_var(ref.name)
        inits.append(
            f"        {base}Gm.SetGlobalBuffer((__gm__ float *){base} + this->blockLength * AscendC::GetBlockIdx(), "
            f"this->blockLength);"
        )
        inits.append(f"        pipe.InitBuffer(outQueue{ref.name}, BUFFER_NUM, this->tileLength * sizeof(float));")
        queues.append(f"    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueue{ref.name};")
        gms.append(f"    AscendC::GlobalTensor<float> {base}Gm;")
        compute_heads.append(f"        AscendC::LocalTensor<float> {base}Local = outQueue{ref.name}.AllocTensor<float>();")
        compute_tails.append(f"        outQueue{ref.name}.EnQue({base}Local);")
        copy_out.append(f"        AscendC::LocalTensor<float> {base}Local = outQueue{ref.name}.DeQue<float>();")
        copy_out.append(
            f"        AscendC::DataCopy({base}Gm[progress * this->tileLength], {base}Local, this->tileLength);"
        )
        copy_out.append(f"        outQueue{ref.name}.FreeTensor({base}Local);")

    # Free input tensors after enqueuing outputs (must free all dequeued tensors).
    for ref in profile.inputs:
        compute_tails.append(f"        inQueue{ref.name}.FreeTensor({tensor_var(ref.name)}Local);")

    for idx in scratch_idx:
        inits.append(f"        pipe.InitBuffer(tmp{idx}, this->tileLength * sizeof(float));")
        members.append(f"    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp{idx};")
        stmts.append(f"        AscendC::LocalTensor<float> s{idx} = tmp{idx}.Get<float>();")

    for stmt in program.statements:
        stmts.append(_statement_lines(stmt, names))

    return "\n".join(
        [
            LICENSE,
            "",
            "",
            f'#include "kernel_operator.h"',
            f'#include "{entry}_tiling.h"',
            f"constexpr int32_t BUFFER_NUM = {_BUFFER_NUM};  // tensor num for each queue",
            f"constexpr int32_t QUEUE_DEPTH = {_QUEUE_DEPTH};",
            "",
            f"class {cls} {{",
            "public:",
            f"    __aicore__ inline {cls}() {{}}",
            "    __aicore__ inline void Init(GM_ADDR " + ", GM_ADDR ".join(
                tensor_var(r.name) for r in (*profile.inputs, *profile.outputs)
            ) + ", uint32_t totalLength, uint32_t tileNum)",
            "    {",
            *inits,
            "    }",
            "",
            "    __aicore__ inline void Process()",
            "    {",
            "        int32_t loopCount = this->tileNum * BUFFER_NUM;",
            "        for (int32_t i = 0; i < loopCount; i++) {",
            "            CopyIn(i);",
            "            Compute(i);",
            "            CopyOut(i);",
            "        }",
            "    }",
            "",
            "private:",
            "    __aicore__ inline void CopyIn(int32_t progress)",
            "    {",
            *copy_in,
            "    }",
            "    __aicore__ inline void Compute(int32_t progress)",
            "    {",
            *compute_heads,
            *stmts,
            *compute_tails,
            "    }",
            "    __aicore__ inline void CopyOut(int32_t progress)",
            "    {",
            *copy_out,
            "    }",
            "",
            "private:",
            "    AscendC::TPipe pipe;",
            *queues,
            *gms,
            *members,
            "    uint32_t blockLength;",
            "    uint32_t tileNum;",
            "    uint32_t tileLength;",
            "};",
            "",
            "",
            f"extern \"C\" __global__ __aicore__ void {entry}(" + ", ".join(
                f"GM_ADDR {tensor_var(r.name)}" for r in (*profile.inputs, *profile.outputs)
            ) + ", GM_ADDR workspace, GM_ADDR tiling)",
            "{",
            f"    REGISTER_TILING_DEFAULT({st});",
            f"    GET_TILING_DATA_WITH_STRUCT({st}, tiling_data, tiling);",
            f"    {cls} op;",
            "    op.Init(" + ", ".join(tensor_var(r.name) for r in (*profile.inputs, *profile.outputs))
            + ", tiling_data.totalLength, tiling_data.tileNum);",
            "    op.Process();",
            "}",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Tiling + host text
# ---------------------------------------------------------------------------


def _build_tiling(profile: FileProfile) -> str:
    st = profile.tiling_struct
    return "\n".join(
        [
            LICENSE,
            "",
            "",
            f"#ifndef {profile.tiling_guard}",
            f"#define {profile.tiling_guard}",
            "#include <cstdint>",
            "",
            f"struct {st} {{",
            "    uint32_t totalLength;",
            "    uint32_t tileNum;",
            "};",
            f"#endif // {profile.tiling_guard}",
            "",
        ]
    )


def _build_host(profile: FileProfile) -> str:
    entry = profile.entry
    st = profile.tiling_struct
    pascal = profile.op_pascal
    #: op_def_registry methods: plural spec section -> singular method name.
    decl_method = {"inputs": "Input", "outputs": "Output"}

    def decl(kind: str) -> list[str]:
        lines: list[str] = []
        for ref in getattr(profile, kind):
            ge_type = f"{{{ref.dtype_ge}}}"
            lines.append(f"        this->{decl_method[kind]}(\"{ref.name}\")")
            lines.append("            .ParamType(REQUIRED)")
            lines.append(f"            .DataType({ge_type})")
            lines.append(f"            .Format({{{ref.format_ge}}});")
        return lines

    input_block = decl("inputs")
    output_block = decl("outputs")

    return "\n".join(
        [
            LICENSE,
            "",
            "",
            '#include "register/op_def_registry.h"',
            f'#include "../op_kernel/{entry}_tiling.h"',
            "",
            "namespace optiling {",
            "",
            "static ge::graphStatus TilingFunc(gert::TilingContext *context)",
            "{",
            "    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();",
            f"    context->SetBlockDim({profile.block_dim});",
            f"    {st} *tiling = context->GetTilingData<{st}>();",
            "    tiling->totalLength = totalLength;",
            "    tiling->tileNum = 1;",
            "    return ge::GRAPH_SUCCESS;",
            "}",
            "}  // namespace optiling",
            "",
            "namespace ge {",
            "static graphStatus InferShape(gert::InferShapeContext *context)",
            "{",
            "    const gert::Shape *inputShape = context->GetInputShape(0);",
            "    gert::Shape *outputShape = context->GetOutputShape(0);",
            "    *outputShape = *inputShape;",
            "    return GRAPH_SUCCESS;",
            "}",
            "",
            "static graphStatus InferDataType(gert::InferDataTypeContext *context)",
            "{",
            "    context->SetOutputDataType(0, context->GetInputDataType(0));",
            "    return ge::GRAPH_SUCCESS;",
            "}",
            "}  // namespace ge",
            "",
            "namespace ops {",
            f"class {pascal} : public OpDef {{",
            "public:",
            f"    explicit {pascal}(const char *name) : OpDef(name)",
            "    {",
            *input_block,
            "",
            *output_block,
            "",
            "        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);",
            "        this->AICore()",
            "            .SetTiling(optiling::TilingFunc)",
            f"            .AddConfig(\"{profile.soc}\");",
            "    }",
            "};",
            f"OP_ADD({pascal});",
            "}  // namespace ops",
            "",
        ]
    )


def _slot_is_scratch(slot: str) -> bool:
    return slot.startswith("s") and slot[1:].isdigit()


def validate_program(profile: FileProfile, program: ExprProgram) -> None:
    """Cross-check a lowering plan against a naming profile.

    Every slot referenced by the plan must be either a scratch slot ``s<i>`` or
    a tensor declared in the profile; expression inputs must be profile inputs
    (an output used as input, or an undeclared tensor, raises a bilingual
    error before any C++ text is produced).
    """
    in_names = {ref.name for ref in profile.inputs}
    out_names = {ref.name for ref in profile.outputs}
    tensor_names = in_names | out_names

    for name in program.inputs:
        if name not in in_names:
            raise OpSpecError(
                t("fillgen.err.unknown_ref", name=name),
                hint=t("fillgen.err.unknown_ref.hint", output=", ".join(sorted(out_names))),
            )

    for stmt in program.statements:
        if stmt.op == "dup" and stmt.scalar is None:
            raise OpSpecError(t("fillgen.err.dup_missing"))
        for slot in (stmt.dst, *stmt.srcs):
            if _slot_is_scratch(slot):
                continue
            if slot not in tensor_names:
                raise OpSpecError(
                    t("fillgen.err.unknown_ref", name=slot),
                    hint=t("fillgen.err.unknown_ref.hint", output=", ".join(sorted(out_names))),
                )


def build_three_files(profile: FileProfile, program: ExprProgram) -> dict[str, str]:
    """Return {relative_path: text} for the three replaceable files.

    Relative paths follow the msopgen layout so ``apply`` can drop them straight
    into an empty-shell project directory.

    :raises OpSpecError: If the plan references tensors the profile does not
        declare, or the plan is otherwise inconsistent (validated first).
    """
    validate_program(profile, program)
    entry = profile.entry
    return {
        f"op_kernel/{entry}.cpp": _build_kernel(profile, program),
        f"op_kernel/{entry}_tiling.h": _build_tiling(profile),
        f"op_host/{entry}.cpp": _build_host(profile),
    }


def files_from_spec(spec: OpSpec) -> tuple[FileProfile, ExprProgram, dict[str, str]]:
    """One-call pipeline used by the gen-op / fill-op CLI commands.

    Validates the v1 contract, parses ``spec.expr`` (infix, LaTeX subset or a
    preset name -- see expr.parse) into an AST, lowers it into a statement plan
    and generates the three replacement texts.

    :param spec: an OpSpec with a non-empty ``expr`` and exactly one output;
    :returns: ``(profile, program, {relative_path: text})``;
    :raises OpSpecError: Bilingual error at the first failing stage (missing
        expr, output-name conflict, undeclared tensor reference, ...).
    """
    profile = profile_from_spec(spec)  # enforces single float ND output
    expr_text = spec.expr.strip()
    if not expr_text:
        raise OpSpecError(t("fillgen.err.expr_missing"), hint=t("fillgen.err.expr_missing.hint"))

    parsed = parse_expr(expr_text)
    declared = profile.outputs[0].name
    if parsed.output is not None and parsed.output != declared:
        raise OpSpecError(
            t("fillgen.err.output_mismatch", given=parsed.output, declared=declared),
            hint=t("fillgen.err.output_mismatch.hint"),
        )
    program = lower_expr(parsed.tree, declared)
    files = build_three_files(profile, program)
    return profile, program, files
