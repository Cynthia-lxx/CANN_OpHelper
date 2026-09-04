"""Test-wide fixtures.

User-facing messages are resolved via the i18n catalog. Existing assertions in
this suite match the Simplified Chinese templates, so pin the active language to
"zh" for every test regardless of the CANN_OPHELPER_LANG environment variable.
"""

from __future__ import annotations

import pytest

from cann_ophelper.i18n import set_language


@pytest.fixture(autouse=True)
def _force_zh_language():
    set_language("zh")
    yield
