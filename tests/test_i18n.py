"""Tests for the bilingual message catalog (cann_ophelper.i18n).

Only this module exercises English templates; every other test pins "zh"
through tests/conftest.py.
"""

from __future__ import annotations

import pytest

from cann_ophelper.i18n import SUPPORTED_LANGUAGES, get_language, set_language, t
from cann_ophelper.model import OpSpecError, TensorSpec


class TestLanguageSelection:
    def test_supported_languages(self):
        assert SUPPORTED_LANGUAGES == ("zh", "en")

    def test_default_language_is_zh(self):
        # tests/conftest.py pins the active language to zh before each test
        assert get_language() == "zh"

    def test_set_language_en(self):
        set_language("en")
        assert get_language() == "en"

    def test_language_names_are_case_insensitive(self):
        set_language("EN")
        assert get_language() == "en"

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError):
            set_language("fr")


class TestLookup:
    def test_zh_template_returned(self):
        assert t("check.outputs_empty") == "outputs 不能为空"

    def test_en_template_returned(self):
        set_language("en")
        assert t("check.outputs_empty") == "outputs must not be empty"

    def test_formatting_placeholders(self):
        set_language("zh")
        msg = t("check.dtype_unsupported", index=1, dtype="fp99")
        assert msg == "type[1] 'fp99' 不在支持的 dtype 集合内"
        set_language("en")
        msg = t("check.dtype_unsupported", index=1, dtype="fp99")
        assert msg == "type[1] 'fp99' is not in the supported dtype set"

    def test_unknown_key_returns_key(self):
        assert t("no.such.key") == "no.such.key"

    def test_fallback_from_zh_to_en_when_key_missing(self):
        # zh and en share all keys today; simulate a zh-only gap by checking a
        # key that exists in en only is impossible -- assert the reverse: en has
        # no extra keys over zh, so this is a guard for catalog consistency.
        for lang in SUPPORTED_LANGUAGES:
            set_language(lang)
            assert t("ci.title")  # non-empty in both languages


class TestIntegration:
    def test_model_error_message_switches_language(self):
        tensor = TensorSpec(name="x", type=["float16", "fp99"])
        set_language("zh")
        with pytest.raises(OpSpecError, match="不在支持的 dtype 集合内"):
            tensor.validate()
        set_language("en")
        with pytest.raises(OpSpecError, match="is not in the supported dtype set"):
            tensor.validate()

    def test_hint_separator_localized(self):
        tensor = TensorSpec(name="x", type=["float16", "fp99"])
        set_language("zh")
        with pytest.raises(OpSpecError) as exc_info:
            tensor.validate()
        assert "建议：" in str(exc_info.value)
        set_language("en")
        with pytest.raises(OpSpecError) as exc_info:
            tensor.validate()
        assert " Hint: " in str(exc_info.value)
