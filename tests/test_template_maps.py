"""Tests for the dtype/format/soc mapping tables (maps.py).

Only combinations confirmed by the official chapter-03 sample are registered;
unknown values must raise OpSpecError. Assertion fragments match the Simplified
Chinese templates (the default language; see tests/conftest.py).
"""

from __future__ import annotations

import pytest

from cann_ophelper.model import OpSpecError
from cann_ophelper.template.maps import ge_dtype, ge_format, opdef_soc


class TestGeDtype:
    def test_maps_official_dtypes(self):
        assert ge_dtype("float16") == "ge::DT_FLOAT16"
        assert ge_dtype("float") == "ge::DT_FLOAT"

    def test_accepts_mixed_case_and_whitespace(self):
        assert ge_dtype(" Float16 ") == "ge::DT_FLOAT16"

    def test_unknown_dtype_raises_with_zh_message(self):
        with pytest.raises(OpSpecError, match="未收录到 ge::DT_\\* 映射表"):
            ge_dtype("double")

    def test_field_path_is_carried(self):
        with pytest.raises(OpSpecError, match=r"inputs\[0\]\.x"):
            ge_dtype("int32", field_path="inputs[0].x")


class TestGeFormat:
    def test_maps_official_format(self):
        assert ge_format("ND") == "ge::FORMAT_ND"

    def test_unknown_format_raises_with_zh_message(self):
        with pytest.raises(OpSpecError, match="未收录到 ge::FORMAT_\\* 映射表"):
            ge_format("NCHW")


class TestOpdefSoc:
    def test_maps_official_soc(self):
        # ascend910b1 -> AddConfig("ascend910b") per official-patterns SS3.3
        assert opdef_soc("ascend910b1") == "ascend910b"

    def test_unknown_soc_raises_with_zh_message(self):
        with pytest.raises(OpSpecError, match="未收录到 AddConfig 映射表"):
            opdef_soc("ascend310p")
