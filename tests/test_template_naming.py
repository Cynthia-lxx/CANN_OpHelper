"""Tests for the template naming helpers.

Templates are rendered from validated OpSpecs; these tests cover the pure
naming derivations (snake name, kernel class, tiling struct, dtype/macro
aliases, include guard) used by the render context.
"""

from __future__ import annotations

from cann_ophelper.template.naming import (
    TILING_STRUCT_NAME,
    dtype_alias,
    kernel_class,
    macro_alias,
    op_snake,
    tiling_guard,
    tiling_struct,
)


class TestOpSnake:
    def test_camel_to_snake_matches_official_rule(self):
        # AddCustomTemplate -> add_custom_template (msopgen file/function rule)
        assert op_snake("AddCustomTemplate") == "add_custom_template"


class TestKernelClass:
    def test_strips_customtemplate_suffix(self):
        # Official sample: AddCustomTemplate -> KernelAdd
        assert kernel_class("AddCustomTemplate") == "KernelAdd"

    def test_strips_custom_suffix(self):
        assert kernel_class("SigmoidCustom") == "KernelSigmoid"

    def test_no_known_suffix_keeps_full_type(self):
        assert kernel_class("Layernorm") == "KernelLayernorm"

    def test_suffix_only_type_not_stripped_to_empty(self):
        assert kernel_class("CustomTemplate").startswith("Kernel")


class TestTilingStruct:
    def test_constant_matches_official_sample(self):
        assert tiling_struct() == TILING_STRUCT_NAME == "TilingDataTemplate"


class TestAliases:
    def test_dtype_alias_single_letter(self):
        assert dtype_alias("x") == "dtypeX"

    def test_dtype_alias_preserves_rest_of_name(self):
        assert dtype_alias("xGm") == "dtypeXGm"

    def test_macro_alias(self):
        assert macro_alias("x") == "DTYPE_X"

    def test_macro_alias_multi_char(self):
        assert macro_alias("xLocal") == "DTYPE_XLOCAL"

    def test_tiling_guard(self):
        assert tiling_guard("add_custom_template") == "ADD_CUSTOM_TEMPLATE_TILING_H"
