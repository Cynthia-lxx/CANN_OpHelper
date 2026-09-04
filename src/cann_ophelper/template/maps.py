"""cann_ophelper.template.maps -- dtype/format/soc mapping tables for codegen.

The official samples write dtypes and formats differently on the host side
(``ge::DT_*`` / ``ge::FORMAT_*`` inside the ops::OpDef registration) and
register the AICore processor with an ``AddConfig`` soc string that differs
from the msopgen ``-c`` argument (``ai_core-ascend910b1`` vs ``"ascend910b"``,
see docs/official-patterns.md SS3.3). These tables convert the normalized
values stored in an :class:`OpSpec` into the exact identifiers the generated
C++ files expect.

The tables are deliberately small: every entry is confirmed by the chapter-03
official sample. Unknown values raise :class:`OpSpecError` with an i18n hint
instead of guessing, so unsupported specs fail loudly during context building.
"""

from __future__ import annotations

from typing import Dict

from ..i18n import t
from ..model import OpSpecError

__all__ = [
    "GE_DTYPE_TABLE",
    "GE_FORMAT_TABLE",
    "OPDEF_SOC_TABLE",
    "ge_dtype",
    "ge_format",
    "opdef_soc",
]

#: dtype -> ``ge::DT_*`` identifier used by the host OpDef registration.
#: Confirmed by the official add_custom.json + add_custom_template.cpp sample
#: (type ``float16`` -> ge::DT_FLOAT16, ``float`` -> ge::DT_FLOAT).
GE_DTYPE_TABLE: Dict[str, str] = {
    "float16": "ge::DT_FLOAT16",
    "float": "ge::DT_FLOAT",
}

#: format -> ``ge::FORMAT_*`` identifier used by the host OpDef registration.
GE_FORMAT_TABLE: Dict[str, str] = {
    "ND": "ge::FORMAT_ND",
}

#: base soc_version -> host ``AddConfig`` value. Kept explicit on purpose: the
#: version-tail drop seen in ``ascend910b1`` -> ``ascend910b`` is not a safe
#: general rule for every SoC.
OPDEF_SOC_TABLE: Dict[str, str] = {
    "ascend910b1": "ascend910b",
}


def ge_dtype(dtype: str, *, field_path: str = "") -> str:
    """Map a normalized dtype (e.g. ``float16``) to ``ge::DT_FLOAT16``.

    :raises OpSpecError: When the dtype has no entry in ``GE_DTYPE_TABLE``.
    """
    path = field_path or "type"
    try:
        return GE_DTYPE_TABLE[str(dtype).strip().lower()]
    except KeyError:
        raise OpSpecError(
            t("tmpl.dtype_unmapped", dtype=dtype),
            field_path=path,
            hint=t("tmpl.dtype_unmapped.hint"),
        ) from None


def ge_format(fmt: str, *, field_path: str = "") -> str:
    """Map a normalized format (e.g. ``ND``) to ``ge::FORMAT_ND``.

    :raises OpSpecError: When the format has no entry in ``GE_FORMAT_TABLE``.
    """
    path = field_path or "format"
    try:
        return GE_FORMAT_TABLE[str(fmt).strip().upper()]
    except KeyError:
        raise OpSpecError(
            t("tmpl.format_unmapped", fmt=fmt),
            field_path=path,
            hint=t("tmpl.format_unmapped.hint"),
        ) from None


def opdef_soc(soc_version: str, *, field_path: str = "") -> str:
    """Map the base ``soc_version`` to the host ``AddConfig`` string.

    E.g. ``ascend910b1`` -> ``ascend910b``.

    :raises OpSpecError: When the soc has no entry in ``OPDEF_SOC_TABLE``.
    """
    path = field_path or "soc_version"
    try:
        return OPDEF_SOC_TABLE[str(soc_version).strip().lower()]
    except KeyError:
        raise OpSpecError(
            t("tmpl.soc_unmapped", soc=soc_version),
            field_path=path,
            hint=t("tmpl.soc_unmapped.hint"),
        ) from None
