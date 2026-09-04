"""Tests for the Jinja2 template engine (whole-file rendering).

Two independent regression layers:
1. self-contained golden snapshots under tests/fixtures/golden (no large
   directories involved), and
2. alignment against the real official S1-S3 sample files (three precise,
   read-only paths) using a documented normalizer that fixes the official
   file's own typos/whitespace quirks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cann_ophelper.model import OpSpec, TensorSpec
from cann_ophelper.template.engine import TEMPLATE_OUTPUTS, TemplateEngine

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "fixtures" / "golden"
OFFICIAL_DIR = Path(
    r"p:\Dev\CANN_Learning_Refs\Documentation_for_Developers"
    r"\ascendc_operator_development\03_intermediate_vector_operator_development\src\custom_op"
)

#: Rendered relpath -> (golden fixture name, official sample relpath)
EXPECTED = {
    "op_kernel/add_custom_template.cpp": (
        "add_custom_template_op_kernel.cpp",
        "op_kernel/add_custom_template.cpp",
    ),
    "op_kernel/add_custom_template_tiling.h": (
        "add_custom_template_op_kernel_tiling.h",
        "op_kernel/add_custom_template_tiling.h",
    ),
    "op_host/add_custom_template.cpp": (
        "add_custom_template_op_host.cpp",
        "op_host/add_custom_template.cpp",
    ),
}


def _spec() -> OpSpec:
    """The reference spec: official AddCustomTemplate, float16/float, ND."""
    tensor = dict(type=["float16", "float"], format=["ND", "ND"])
    return OpSpec(
        op_type="AddCustomTemplate",
        soc_version="ascend910b1",
        inputs=[TensorSpec(name="x", **tensor), TensorSpec(name="y", **tensor)],
        outputs=[TensorSpec(name="z", **tensor)],
    )


def normalize_official(text: str) -> str:
    """Apply the documented corrections to an official sample so it matches the
    canonical template output (see docs/official-patterns.md SS7):
    - host typo ``intputShape`` -> ``inputShape``;
    - kernel extra space before ``__global__`` removed;
    - kernel printf double space after the comma removed;
    - every file ends with exactly one newline.
    """
    text = text.replace("intputShape", "inputShape")
    text = text.replace("\n __global__", "\n__global__")
    text = text.replace(",  AscendC::GetBlockIdx()", ", AscendC::GetBlockIdx()")
    if not text.endswith("\n"):
        text += "\n"
    return text


class TestRenderShape:
    def test_renders_exactly_three_outputs(self):
        files = TemplateEngine().render(_spec())
        assert set(files) == set(EXPECTED)

    def test_output_paths_use_snake_case(self):
        files = TemplateEngine().render(_spec())
        for relpath in files:
            assert "add_custom_template" in relpath

    def test_every_file_is_nonempty_and_ends_with_newline(self):
        files = TemplateEngine().render(_spec())
        for relpath, text in files.items():
            assert text.strip(), f"{relpath} rendered empty"
            assert text.endswith("\n"), f"{relpath} missing trailing newline"

    def test_template_output_mapping_contract(self):
        # every (template, relpath pattern) is unique and points into op_*/
        names = [t for t, _ in TEMPLATE_OUTPUTS]
        assert len(names) == len(set(names)) == 3


class TestDeterminism:
    def test_two_renders_are_byte_identical(self):
        assert TemplateEngine().render(_spec()) == TemplateEngine().render(_spec())


class TestGoldenSnapshots:
    @pytest.mark.parametrize("relpath", sorted(EXPECTED))
    def test_matches_golden_fixture(self, relpath):
        golden_name = EXPECTED[relpath][0]
        text = TemplateEngine().render(_spec())[relpath]
        golden = (GOLDEN_DIR / golden_name).read_text(encoding="utf-8")
        assert text == golden


class TestOfficialAlignment:
    @pytest.mark.parametrize("relpath", sorted(EXPECTED))
    def test_rendered_text_matches_normalized_official_sample(self, relpath):
        golden_name, official_rel = EXPECTED[relpath]
        official_path = OFFICIAL_DIR / official_rel
        if not official_path.is_file():
            pytest.skip(f"official sample not available: {official_path}")
        text = TemplateEngine().render(_spec())[relpath]
        normalized = normalize_official(official_path.read_text(encoding="utf-8"))
        assert text == normalized


class TestContentSmoke:
    def test_kernel_contains_official_identifiers(self):
        text = TemplateEngine().render(_spec())["op_kernel/add_custom_template.cpp"]
        for fragment in (
            "class KernelAdd",
            "template <class dtypeX, class dtypeY, class dtypeZ>",
            "__global__ __aicore__ void add_custom_template",
            "REGISTER_TILING_DEFAULT(TilingDataTemplate)",
            "KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op",
            "AscendC::Add(zLocal, xLocal, yLocal, this->tileLength)",
        ):
            assert fragment in text

    def test_host_contains_official_identifiers(self):
        text = TemplateEngine().render(_spec())["op_host/add_custom_template.cpp"]
        for fragment in (
            "static ge::graphStatus TilingFunc",
            "context->SetBlockDim(8)",
            ".DataType({ge::DT_FLOAT16, ge::DT_FLOAT})",
            ".Format({ge::FORMAT_ND, ge::FORMAT_ND})",
            '.AddConfig("ascend910b")',
            "class AddCustomTemplate : public OpDef",
            "OP_ADD(AddCustomTemplate);",
        ):
            assert fragment in text

    def test_tiling_header_contains_guard_and_struct(self):
        text = TemplateEngine().render(_spec())["op_kernel/add_custom_template_tiling.h"]
        assert "#ifndef ADD_CUSTOM_TEMPLATE_TILING_H" in text
        assert "struct TilingDataTemplate {" in text
        assert "uint32_t totalLength;" in text
        assert "uint32_t tileNum;" in text
