"""cann_ophelper.proto -- Serialize a validated OpSpec into the official msopgen
prototype JSON (the ``msopgen gen -i`` input).

Workflow position (docs/official-patterns.md SS1.3): the prototype JSON declares
the operator name plus its input/output descriptor entries (``name``,
``param_type`` and the parallel ``format[]`` / ``type[]`` arrays). ``OpSpec``
(model.py) mirrors exactly those fields, so a validated spec can be
*mechanically translated* into the prototype JSON -- the only path this tool
uses to produce such a file. Workspace rule 6.4 was relaxed accordingly: no
fields are ever fabricated without official backing.

Scope limits in this phase:
- ``shape`` / ``soc_version`` / ``description`` never enter the prototype JSON:
  shape is a runtime quantity, soc is an msopgen ``-c`` argument and description
  is project metadata.
- ``attrs`` are not exported yet: none of the official sample prototype files in
  the local documentation shows an attr entry layout, so exporting one would
  mean inventing a format. ``prototype_json_text`` therefore raises
  ``OpSpecError`` when ``spec.attrs`` is non-empty.

Serialization style: a top-level array holding one operator object; the entry
key order follows the official samples (``name`` -> ``param_type`` -> ``format``
-> ``type``); 4-space indentation via ``json.dumps`` (layout is cosmetic --
msopgen only parses the JSON structure).

All user-facing messages are resolved through ``cann_ophelper.i18n``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .i18n import t
from .model import OpSpec, OpSpecError, TensorSpec

__all__ = ["prototype_json_text", "dump_prototype_json"]


def _tensor_entry(tensor: TensorSpec) -> dict:
    """Official entry layout: name/param_type/format/type, never ``shape``.

    Built explicitly instead of reusing ``TensorSpec.to_dict`` so that the
    optional ``shape`` metadata can never leak into the msopgen input.
    """
    return {
        "name": tensor.name,
        "param_type": tensor.param_type,
        "format": tensor.format,
        "type": tensor.type,
    }


def prototype_json_text(spec: OpSpec) -> str:
    """Validate ``spec``, then serialize the official prototype JSON text.

    :raises OpSpecError: if the spec is invalid, or if it declares attrs (no
        official layout reference exists in this phase, so they are not
        exportable).
    """
    spec.validate()
    if spec.attrs:
        raise OpSpecError(
            t("proto.attr_unsupported", count=len(spec.attrs)),
            hint=t("proto.attr_unsupported.hint"),
        )
    payload: list[dict[str, Any]] = [
        {
            "op": spec.op_type,
            "input_desc": [_tensor_entry(tensor) for tensor in spec.inputs],
            "output_desc": [_tensor_entry(tensor) for tensor in spec.outputs],
        }
    ]
    return json.dumps(payload, indent=4, ensure_ascii=False)


def dump_prototype_json(spec: OpSpec, path: Union[str, Path]) -> Path:
    """Validate, then write the prototype JSON to ``path`` (parents created).

    Errors (validation, filesystem) surface as ``OpSpecError`` with an i18n hint,
    mirroring the error surface of ``yamlio.dump_op_spec``.
    """
    p = Path(path)
    text = prototype_json_text(spec)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise OpSpecError(
            t("proto.write_fail", path=p, reason=reason),
            hint=t("proto.write_fail.hint"),
        ) from exc
    return p
