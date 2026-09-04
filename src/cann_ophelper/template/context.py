"""cann_ophelper.template.context -- Build the Jinja2 render context from an OpSpec.

The render context is the single contract between the data model and the .j2
templates: it pre-derives every identifier the templates need (naming tokens,
the host ``ge::DT_*`` / ``ge::FORMAT_*`` lists, the ``AddConfig`` soc) so the
templates stay free of logic and remain diff-able against the official sample
files. Future rounds that rework whole-file templates into snippets only touch
this module and the templates, never the data model.

Every value is derived from a validated spec; anything unmappable raises
:class:`OpSpecError` through template.maps, carrying a field context and an
i18n hint.
"""

from __future__ import annotations

from typing import Any, Dict

from ..model import OpSpec, TensorSpec
from .maps import ge_dtype, ge_format, opdef_soc
from .naming import (
    dtype_alias,
    kernel_class,
    macro_alias,
    op_snake,
    tiling_guard,
    tiling_struct,
)

__all__ = ["build_render_context"]


def _tensor_ctx(kind: str, idx: int, tensor: TensorSpec) -> Dict[str, Any]:
    """Derive the render entries for one input/output tensor.

    ``ge_types`` / ``ge_formats`` keep the parallel-array semantics of the
    official prototype: both are rendered as comma-joined lists of the same
    length (never deduplicated, so dtype+format pairs stay aligned).
    """
    base = f"{kind}[{idx}].{tensor.name}"
    types = [ge_dtype(dt, field_path=f"{base}.type[{j}]") for j, dt in enumerate(tensor.type)]
    formats = [ge_format(fm, field_path=f"{base}.format[{j}]") for j, fm in enumerate(tensor.format)]
    return {
        "name": tensor.name,
        "cap": tensor.name[:1].upper() + tensor.name[1:],
        "param_enum": tensor.param_type.upper(),
        "dtype_alias": dtype_alias(tensor.name),
        "macro_alias": macro_alias(tensor.name),
        "ge_types": ", ".join(types),
        "ge_formats": ", ".join(formats),
    }


def build_render_context(spec: OpSpec) -> Dict[str, Any]:
    """Build the template render context for ``spec``.

    The spec is validated first so identifier/mapping failures follow the
    existing :class:`OpSpecError` contract (field context + i18n hint). Lists
    keep the spec's order, so output is deterministic.
    """
    spec.validate()
    snake = op_snake(spec.op_type)
    return {
        "op_type": spec.op_type,
        "op_snake": snake,
        "kernel_class": kernel_class(spec.op_type),
        "tiling_struct": tiling_struct(),
        "tiling_guard": tiling_guard(snake),
        "tiling_header_file": f"{snake}_tiling.h",
        "soc_config": opdef_soc(spec.soc_version, field_path="soc_version"),
        "inputs": [_tensor_ctx("inputs", i, t) for i, t in enumerate(spec.inputs)],
        "outputs": [_tensor_ctx("outputs", i, t) for i, t in enumerate(spec.outputs)],
    }
