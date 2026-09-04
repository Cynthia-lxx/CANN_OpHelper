"""cann_ophelper.yamlio -- Persistence between OpSpec and YAML.

Responsibilities:
- ``load_op_spec(path)``: read YAML -> validate -> return ``OpSpec``;
- ``dump_op_spec(spec, path)``: ``OpSpec`` -> validate -> block-style YAML to disk
  (UTF-8, stable field order).

Error handling: missing files, YAML syntax errors and structural/field problems
all raise ``OpSpecError`` (a ``ValueError`` subclass) carrying context and a fix
hint, ready for typer/rich rendering later.

Dependencies: PyYAML (declared in pyproject; only executed in a real environment).

All user-facing messages are resolved through ``cann_ophelper.i18n``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .i18n import t
from .model import OpSpec, OpSpecError

__all__ = ["load_op_spec", "dump_op_spec", "op_spec_to_yaml_text", "yaml_text_to_op_spec"]

#: YAML allows non-string keys off the root; restrict the root to a mapping for
#: friendlier errors.
_FILE_ENCODING = "utf-8"


def yaml_text_to_op_spec(text: str) -> OpSpec:
    """Parse YAML text into an OpSpec. Syntax and semantic errors become OpSpecError."""
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OpSpecError(t("yamlio.syntax", err=exc), hint=t("yamlio.syntax.hint")) from exc
    if raw is None:
        raise OpSpecError(t("yamlio.empty"), hint=t("yamlio.empty.hint"))
    try:
        return OpSpec.from_dict(raw)
    except OpSpecError:
        raise
    except Exception as exc:  # noqa: BLE001 -- defensive fallback, unified error surface
        raise OpSpecError(t("yamlio.parse", err=exc), hint=t("yamlio.parse.hint")) from exc


def op_spec_to_yaml_text(spec: OpSpec, *, sort_keys: bool = False) -> str:
    """Serialize an OpSpec to block-style YAML text.

    - Validation first: an invalid model is never written;
    - ``sort_keys=False`` keeps the stable field order from to_dict for easy
      diffs and human reading.
    """
    spec.validate()
    return yaml.safe_dump(
        spec.to_dict(),
        allow_unicode=True,
        sort_keys=sort_keys,
        default_flow_style=False,
        width=100,
    )


def load_op_spec(path: Union[str, Path]) -> OpSpec:
    """Load and validate an operator spec YAML from file."""
    p = Path(path)
    if not p.exists():
        raise OpSpecError(t("yamlio.file_missing", path=p), hint=t("yamlio.file_missing.hint"))
    if not p.is_file():
        raise OpSpecError(t("yamlio.not_file", path=p), hint=t("yamlio.not_file.hint"))
    try:
        text = p.read_text(encoding=_FILE_ENCODING)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise OpSpecError(
            t("yamlio.read_fail", path=p, reason=reason),
            hint=t("yamlio.read_fail.hint"),
        ) from exc
    spec = yaml_text_to_op_spec(text)
    spec.source = str(p)  # type: ignore[attr-defined]
    return spec


def dump_op_spec(spec: OpSpec, path: Union[str, Path]) -> Path:
    """Validate then write an OpSpec as block-style YAML; create parents if needed."""
    p = Path(path)
    text = op_spec_to_yaml_text(spec)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding=_FILE_ENCODING)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise OpSpecError(
            t("yamlio.write_fail", path=p, reason=reason),
            hint=t("yamlio.write_fail.hint"),
        ) from exc
    return p
