"""cann_ophelper.model -- Operator metadata data model and validation.

Field semantics follow the official msopgen operator prototype JSON (see
docs/official-patterns.md SS1.3/SS1.4):

- Each input_desc/output_desc entry carries ``name``, ``param_type``
  (required/optional), ``format`` (array) and ``type`` (array); ``format`` and
  ``type`` are *parallel arrays*, so the entries with the same index form one
  supported "format + dtype" combination (e.g. ["ND","ND"] and
  ["float16","float"]).
- The prototype JSON does **not** contain ``shape`` or ``soc``: shape is a
  runtime quantity and soc is an msopgen ``-c`` command line argument. They are
  therefore represented by this module's ``OpSpec`` (``soc_version``), while
  ``shape`` lives on ``TensorSpec.shape`` as an optional operator shape hint
  that never enters the msopgen command.

This module depends only on the standard library and is shared by yamlio /
msopgen / later template engines.

All user-facing messages are resolved through the bilingual catalog in
``cann_ophelper.i18n``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Mapping, Optional

from .i18n import t

__all__ = [
    "OpSpecError",
    "ParamType",
    "TensorSpec",
    "AttrSpec",
    "OpSpec",
    "SUPPORTED_DTYPES",
    "SUPPORTED_FORMATS",
    "normalize_dtype",
    "normalize_format",
    "as_list",
]


class OpSpecError(ValueError):
    """Raised when an operator spec is invalid.

    The message carries a field context (reason) and a fix hint. Both pieces
    are resolved through the i18n catalog at the call site.
    """

    def __init__(self, message: str, *, field_path: str = "", hint: str = "") -> None:
        self.field_path = field_path
        self.hint = hint
        full = message
        if field_path:
            full = f"{field_path}: {message}"
        if hint:
            full = f"{full}{t('msg.hint_join')}{hint}"
        super().__init__(full)


# ---------------------------------------------------------------------------
# Constants and normalization
# ---------------------------------------------------------------------------

class ParamType(str, Enum):
    """Whether a parameter is required; matches the official ``param_type`` value."""

    REQUIRED = "required"
    OPTIONAL = "optional"


#: Common dtypes (lowercase). Based on official samples using float/float16,
#: plus common numeric types. These are string aliases of ge::DT_*.
SUPPORTED_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float",  # "float" means FP32 in the official JSON (ge::DT_FLOAT)
        "double",
        "bfloat16",
        "complex64",
        "complex128",
        "string",
    }
)

#: Common formats (uppercase); string aliases of ge::FORMAT_*.
SUPPORTED_FORMATS = frozenset(
    {"ND", "NCHW", "NHWC", "NC1HWC0", "NDC1HWC0", "NZ", "FRACTAL_Z", "FRACTAL_NZ", "FRACTAL_ZN_L2"}
)

_OP_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ATTR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def as_list(value: Any) -> List[Any]:
    """Normalize a scalar to a one-element list; None -> []; lists/tuples pass through."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_dtype(value: Any) -> str:
    """Normalize a dtype: strip whitespace and lower-case. Unknown values are kept for validation."""
    return str(value).strip().lower()


def normalize_format(value: Any) -> str:
    """Normalize a format: strip whitespace and upper-case. Unknown values are kept for validation."""
    return str(value).strip().upper()


def _check_identifier(value: str, what: str) -> None:
    if not value:
        raise OpSpecError(t("check.name_empty", what=what), hint=t("check.name_empty.hint"))
    if not _OP_TYPE_RE.match(value):
        raise OpSpecError(
            t("check.name_invalid", what=what, value=value),
            hint=t("check.name_invalid.hint"),
        )


def camel_to_snake(name: str) -> str:
    """Convert PascalCase/camelCase to snake_case (official op_type -> file/function rule).

    E.g. AddCustomTemplate -> add_custom_template. Consecutive capitals are
    split at the last capital before a lower-case run.
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# TensorSpec / AttrSpec
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """A single input/output tensor descriptor; field names match the official prototype JSON entry.

    ``format`` and ``type`` are parallel arrays: e.g. format=["ND","ND"] and
    type=["float16","float"] mean the tensor supports both (ND, float16) and
    (ND, float) combinations. For convenience a single string is accepted for
    either side and is promoted to a one-element list automatically.
    """

    name: str
    type: List[str] = field(default_factory=lambda: ["float"])
    format: List[str] = field(default_factory=lambda: ["ND"])
    param_type: str = ParamType.REQUIRED.value
    #: Optional shape hint (e.g. [1024, 1024], -1 allowed for dynamic dims).
    #: Metadata only; never enters the msopgen command.
    shape: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.type = [normalize_dtype(x) for x in as_list(self.type)] or ["float"]
        self.format = [normalize_format(f) for f in as_list(self.format)] or ["ND"]
        self.param_type = str(self.param_type).strip().lower()
        if self.shape is not None:
            self.shape = list(self.shape)
        # Friendly broadcast: a singleton side is stretched to the other side's
        # length. E.g. type=[float16, float] without format -> format=[ND, ND].
        if len(self.type) == 1 and len(self.format) > 1:
            self.type = self.type * len(self.format)
        elif len(self.format) == 1 and len(self.type) > 1:
            self.format = self.format * len(self.type)

    # -- Convenience read-only aliases (dtypes/formats == type/format) --
    @property
    def dtypes(self) -> List[str]:
        return list(self.type)

    @property
    def formats(self) -> List[str]:
        return list(self.format)

    def validate(self, *, field_path: str = "") -> None:
        path = f"{field_path}.{self.name}" if field_path else self.name
        _check_identifier(self.name, path)
        if self.param_type not in (ParamType.REQUIRED.value, ParamType.OPTIONAL.value):
            raise OpSpecError(
                t("check.param_type_invalid", value=self.param_type),
                field_path=path,
                hint=t("check.param_type_invalid.hint"),
            )
        if len(self.format) != len(self.type):
            raise OpSpecError(
                t("check.type_format_len", fmt_len=len(self.format), type_len=len(self.type)),
                field_path=path,
                hint=t("check.type_format_len.hint"),
            )
        for i, dt in enumerate(self.type):
            if dt not in SUPPORTED_DTYPES:
                raise OpSpecError(
                    t("check.dtype_unsupported", index=i, dtype=dt),
                    field_path=path,
                    hint=t("check.supported_values.hint", values=", ".join(sorted(SUPPORTED_DTYPES))),
                )
        for i, fm in enumerate(self.format):
            if fm not in SUPPORTED_FORMATS:
                raise OpSpecError(
                    t("check.format_unsupported", index=i, fmt=fm),
                    field_path=path,
                    hint=t("check.supported_values.hint", values=", ".join(sorted(SUPPORTED_FORMATS))),
                )

    def to_dict(self) -> dict:
        """Convert to dict; key order matches the official prototype JSON entry.
        Singleton type/format are still emitted as arrays."""
        data = {
            "name": self.name,
            "param_type": self.param_type,
            "format": self.format,
            "type": self.type,
        }
        if self.shape:
            data["shape"] = self.shape
        return data

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "TensorSpec":
        try:
            spec = cls(
                name=mapping["name"],
                type=mapping.get("type", ["float"]),
                format=mapping.get("format", ["ND"]),
                param_type=mapping.get("param_type", ParamType.REQUIRED.value),
                shape=mapping.get("shape"),
            )
        except KeyError as exc:
            raise OpSpecError(
                t("check.missing_required"),
                hint=t("check.tensor_needs_name.hint", key=exc.args[0]),
            ) from exc
        spec.validate()
        return spec


@dataclass
class AttrSpec:
    """Scalar attribute descriptor, matching the official ``attr_desc`` entry (name/param_type/type/value)."""

    name: str
    type: str = "int"  # Type string (int/float/bool/string/listInt...); not restricted to a whitelist
    value: Any = None
    param_type: str = ParamType.REQUIRED.value

    def __post_init__(self) -> None:
        self.type = str(self.type).strip()
        self.param_type = str(self.param_type).strip().lower()

    def validate(self, *, field_path: str = "") -> None:
        path = f"{field_path}.{self.name}" if field_path else self.name
        _check_identifier(self.name, path)
        if not self.type:
            raise OpSpecError(t("check.attr_type_empty"), field_path=path)
        if self.param_type not in (ParamType.REQUIRED.value, ParamType.OPTIONAL.value):
            raise OpSpecError(
                t("check.param_type_invalid", value=self.param_type),
                field_path=path,
                hint=t("check.param_type_invalid.hint"),
            )

    def to_dict(self) -> dict:
        data: dict = {"name": self.name, "param_type": self.param_type, "type": self.type}
        if self.value is not None:
            data["value"] = self.value
        return data

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "AttrSpec":
        try:
            spec = cls(
                name=mapping["name"],
                type=mapping.get("type", "int"),
                value=mapping.get("value"),
                param_type=mapping.get("param_type", ParamType.REQUIRED.value),
            )
        except KeyError as exc:
            raise OpSpecError(
                t("check.missing_required"),
                hint=t("check.attr_needs_name.hint", key=exc.args[0]),
            ) from exc
        spec.validate()
        return spec


# ---------------------------------------------------------------------------
# OpSpec
# ---------------------------------------------------------------------------

@dataclass
class OpSpec:
    """Complete metadata for one operator.

    Key fields:
    - ``op_type``: PascalCase operator type name (e.g. AddCustomTemplate / Sigmoid);
    - ``soc_version``: base Ascend SoC version (e.g. ``ascend910b1``); the msopgen
      ``-c`` argument is composed as ``ai_core-<soc_version>`` at command build time
      (see msopgen.py / official-patterns SS1.2);
    - ``language``: optional official JSON field, fixed ``cpp`` (Ascend C);
    - ``inputs`` / ``outputs``: input/output TensorSpec lists;
    - ``attrs``: scalar attributes;
    - ``tiling``: reserved dict for future strategies (not used in this phase);
    - ``description``: one-line operator description.
    - ``expr``: optional element-wise computation intent, e.g.
      ``"A + 2/sigmoid(B) = C"``. It is consumed only by the expression-driven
      codegen flow (gen-op/fill-op); the spec/render paths and the msopgen
      prototype JSON ignore it, so a spec without ``expr`` keeps byte-identical
      output.
    """

    op_type: str
    soc_version: str = "ascend910b1"
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    attrs: List[AttrSpec] = field(default_factory=list)
    tiling: dict = field(default_factory=dict)
    language: str = "cpp"
    description: str = ""
    #: Optional element-wise computation intent text, e.g. ``"A + 2/sigmoid(B) = C"``.
    #: Metadata only: never enters the msopgen prototype JSON or the add-family
    #: render path, so old specs keep byte-identical behavior.
    expr: str = ""
    #: Metadata: source file path (injected by load_op_spec; not serialized to YAML)
    source: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.op_type = str(self.op_type).strip()
        self.soc_version = str(self.soc_version).strip()
        self.language = str(self.language).strip().lower() or "cpp"
        self.expr = str(self.expr or "").strip()

    # -- Convenience read-only properties --
    @property
    def op_name_snake(self) -> str:
        """snake_case form, matching the official op_type -> file/function rule
        (e.g. AddCustomTemplate -> add_custom_template)."""
        return camel_to_snake(self.op_type)

    def validate(self) -> None:
        _check_identifier(self.op_type, "op_type")
        if not self.soc_version:
            raise OpSpecError(
                t("check.soc_empty"),
                hint=t("check.soc_empty.hint"),
            )
        if not _SOC_RE.match(self.soc_version):
            raise OpSpecError(
                t("check.soc_invalid", value=self.soc_version),
                hint=t("check.soc_invalid.hint"),
            )
        if self.language not in ("cpp",):
            raise OpSpecError(
                t("check.language_invalid", value=self.language),
                hint=t("check.language_model.hint"),
            )

        seen: dict = {}
        for kind in ("inputs", "outputs"):
            for idx, tensor in enumerate(getattr(self, kind)):
                tensor.validate(field_path=f"{kind}[{idx}]")
                key = tensor.name
                if key in seen:
                    raise OpSpecError(
                        t("check.dup_name", name=key),
                        field_path=f"{kind}[{idx}]",
                        hint=t("check.dup_name_tensor.hint"),
                    )
                seen[key] = kind
        if not self.outputs:
            raise OpSpecError(
                t("check.outputs_empty"),
                hint=t("check.outputs_empty.hint"),
            )

        for idx, attr in enumerate(self.attrs):
            attr.validate(field_path=f"attrs[{idx}]")
            if attr.name in seen:
                raise OpSpecError(
                    t("check.dup_name", name=attr.name),
                    field_path=f"attrs[{idx}]",
                    hint=t("check.dup_name_attr.hint"),
                )

    # -- Serialization --
    def to_dict(self) -> dict:
        """Stable key order: op_type -> soc_version -> language -> inputs -> outputs
        -> attrs -> tiling -> description."""
        data: dict = {
            "op_type": self.op_type,
            "soc_version": self.soc_version,
        }
        if self.language != "cpp":
            data["language"] = self.language
        data["inputs"] = [x.to_dict() for x in self.inputs]
        data["outputs"] = [x.to_dict() for x in self.outputs]
        if self.attrs:
            data["attrs"] = [a.to_dict() for a in self.attrs]
        if self.tiling:
            data["tiling"] = self.tiling
        if self.description:
            data["description"] = self.description
        if self.expr:
            data["expr"] = self.expr
        return data

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "OpSpec":
        if not isinstance(mapping, dict):
            raise OpSpecError(
                t("check.top_mapping"),
                hint=t("check.top_mapping.hint"),
            )
        missing = [k for k in ("op_type",) if not mapping.get(k)]
        if missing:
            raise OpSpecError(
                t("check.missing_required"),
                hint=t("check.missing_op_type.hint", keys=", ".join(missing)),
            )

        spec = cls(
            op_type=str(mapping["op_type"]).strip(),
            soc_version=str(mapping.get("soc_version", "ascend910b1")).strip(),
            inputs=[TensorSpec.from_dict(m) for m in mapping.get("inputs", [])],
            outputs=[TensorSpec.from_dict(m) for m in mapping.get("outputs", [])],
            attrs=[AttrSpec.from_dict(m) for m in mapping.get("attrs", [])],
            tiling=dict(mapping.get("tiling", {}) or {}),
            language=str(mapping.get("language", "cpp")).strip().lower() or "cpp",
            description=str(mapping.get("description", "")).strip(),
            expr=str(mapping.get("expr", "") or "").strip(),
        )
        spec.validate()
        return spec
