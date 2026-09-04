"""Minimal tests for msopgen command assembly (argument order, soc prefix,
quoting and path passthrough).

Assertion fragments in this file match the Simplified Chinese templates of the
i18n catalog (the default language; see tests/conftest.py).
"""

from __future__ import annotations

import pytest

from cann_ophelper.model import OpSpec, OpSpecError, TensorSpec
from cann_ophelper.msopgen import (
    build_msopgen_command,
    format_soc_for_msopgen,
    shell_quote,
    show_cloud_instructions,
)

SOC_BASE = "ascend910b1"


def _spec(soc_version=SOC_BASE, language="cpp"):
    return OpSpec(
        op_type="AddCustomTemplate",
        soc_version=soc_version,
        language=language,
        inputs=[
            TensorSpec(name="x", type="float16"),
            TensorSpec(name="y", type="float16"),
        ],
        outputs=[TensorSpec(name="z", type="float16")],
    )


class TestFormatSoc:
    def test_prefix_added_for_bare_soc(self):
        assert format_soc_for_msopgen("ascend910b1") == "ai_core-ascend910b1"

    def test_prefix_not_duplicated(self):
        assert format_soc_for_msopgen("ai_core-ascend910b1") == "ai_core-ascend910b1"


class TestShellQuote:
    def test_plain_path_unchanged(self):
        assert shell_quote("Sources/03.02/add_custom.json") == "Sources/03.02/add_custom.json"

    def test_space_path_quoted(self):
        assert shell_quote("my dir/op.json") == "'my dir/op.json'"


class TestBuildCommand:
    def test_command_matches_official_layout(self):
        cmd = build_msopgen_command(_spec(), "add_custom.json", "custom_op")
        assert cmd == "msopgen gen -i add_custom.json -c ai_core-ascend910b1 -lan cpp -out custom_op"

    def test_absolute_paths_passed_through(self):
        cmd = build_msopgen_command(_spec(), "C:/ops/add_custom.json", "C:/out/custom_op")
        assert "-i C:/ops/add_custom.json" in cmd
        assert "-out C:/out/custom_op" in cmd

    def test_path_with_spaces_gets_quoted(self):
        cmd = build_msopgen_command(_spec(), "path with space/add_custom.json", "out dir")
        assert "'path with space/add_custom.json'" in cmd
        assert "'out dir'" in cmd

    def test_parameter_order_stable(self):
        cmd = build_msopgen_command(_spec(), "p.json", "o")
        # follow the official example order: -i -> -c -> -lan -> -out
        assert cmd.index("-i") < cmd.index("-c") < cmd.index("-lan") < cmd.index("-out")

    def test_prefixed_soc_not_doubled(self):
        cmd = build_msopgen_command(_spec(soc_version="ai_core-ascend910b1"), "p.json", "o")
        assert "-c ai_core-ascend910b1" in cmd

    def test_non_cpp_language_rejected(self):
        with pytest.raises(OpSpecError, match="language"):
            build_msopgen_command(_spec(language="python"), "p.json", "o")

    def test_invalid_spec_rejected_before_output(self):
        bad = _spec()
        bad.outputs = []  # bypass dataclass construction to break validation
        with pytest.raises(OpSpecError, match="outputs 不能为空"):
            build_msopgen_command(bad, "p.json", "o")


class TestCloudInstructions:
    def test_contains_command_and_steps(self):
        text = show_cloud_instructions(_spec(), "add_custom.json", "custom_op")
        assert "msopgen gen" in text
        assert "ai_core-ascend910b1" in text
        assert "op_host" in text
