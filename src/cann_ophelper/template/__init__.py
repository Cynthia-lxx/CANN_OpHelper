"""cann_ophelper.template -- Jinja2 template engine for Ascend C operator code.

This package renders the "phase 2" artifacts of the CANN_OpHelper workflow:
given an :class:`OpSpec` it produces complete op_kernel / op_host file texts
that align with the official AddCustomTemplate sample (chapter 03, see
docs/official-patterns.md). Templates are shipped inside the package
(``templates/``) and loaded through Jinja2's PackageLoader.

Layering (no cycles): context -> naming/maps/model; engine -> context.
Each layer is stateless and independently testable. This module re-exports the
public surface so callers can ``from cann_ophelper.template import ...``.
"""

from __future__ import annotations

from .context import build_render_context
from .maps import ge_dtype, ge_format, opdef_soc
from .naming import (
    dtype_alias,
    kernel_class,
    macro_alias,
    op_snake,
    tiling_guard,
    tiling_struct,
)

__all__ = [
    "build_render_context",
    "dtype_alias",
    "ge_dtype",
    "ge_format",
    "kernel_class",
    "macro_alias",
    "op_snake",
    "opdef_soc",
    "tiling_guard",
    "tiling_struct",
]
