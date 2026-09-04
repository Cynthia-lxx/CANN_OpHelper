"""cann_ophelper -- A Windows-local CLI toolkit that assists in generating CANN
Ascend C operator project templates.

This project never compiles or runs C++ locally; compilation validation happens
in the cloud CANN Lab.

Language policy: code comments/docstrings are written in English; user-facing
messages are resolved through the bilingual catalog in ``cann_ophelper.i18n``
(Simplified Chinese by default, switchable to English).
"""

__version__ = "0.1.0"

from .i18n import SUPPORTED_LANGUAGES, get_language, set_language, t
from .model import AttrSpec, OpSpec, OpSpecError, ParamType, TensorSpec

__all__ = [
    "__version__",
    "OpSpec",
    "TensorSpec",
    "AttrSpec",
    "ParamType",
    "OpSpecError",
    "SUPPORTED_LANGUAGES",
    "set_language",
    "get_language",
    "t",
]
