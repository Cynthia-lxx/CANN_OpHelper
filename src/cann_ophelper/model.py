"""cann_ophelper.model —— 算子元信息数据模型与校验。

字段语义对齐官方 msopgen 算子原型 JSON（见 docs/official-patterns.md §1.3/§1.4）：

- msopgen 原型的 input_desc/output_desc 条目含 ``name``、``param_type``（required/optional）、
  ``format``（数组）、``type``（数组）；``format`` 与 ``type`` 为**并行数组**，
  下标相同者构成一组受支持的“format + dtype”组合（如 ["ND","ND"] + ["float16","float"]）。
- 原型的 JSON 中**不含 shape 与 soc**：shape 是运行期量，soc 是 msopgen ``-c`` 命令行参数，
  因此它们由本模块的 ``OpSpec`` 承担（``soc_version``），``shape`` 仅在 ``TensorSpec.shape`` 上
  作为可选的算子形状提示，不参与 msopgen 命令生成。

本模块只依赖标准库，供 yamlio / msopgen / 后续模板引擎共用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Mapping, Optional

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
    """算子描述非法：消息含字段上下文（原因）与修正建议（提示）。"""

    def __init__(self, message: str, *, field_path: str = "", hint: str = "") -> None:
        self.field_path = field_path
        self.hint = hint
        full = message
        if field_path:
            full = f"{field_path}: {message}"
        if hint:
            full = f"{full} 建议：{hint}"
        super().__init__(full)


# ---------------------------------------------------------------------------
# 常量与规范化
# ---------------------------------------------------------------------------

class ParamType(str, Enum):
    """参数是否必选，对应官方原型 JSON 的 ``param_type`` 取值。"""

    REQUIRED = "required"
    OPTIONAL = "optional"


#: 常见 dtype（小写）。以官方样例实测到的 float/float16 为准，并补充常用数值类型。
#: 取自 msopgen 算子原型 JSON 的 ``type`` 字段惯例（ge::DT_* 的字符串别名）。
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
        "float",  # 官方 JSON 以 "float" 表示 FP32（对应 ge::DT_FLOAT）
        "double",
        "bfloat16",
        "complex64",
        "complex128",
        "string",
    }
)

#: 常见 format（大写），对应官方 ``format`` 数组取值（ge::FORMAT_* 的字符串别名）。
SUPPORTED_FORMATS = frozenset(
    {"ND", "NCHW", "NHWC", "NC1HWC0", "NDC1HWC0", "NZ", "FRACTAL_Z", "FRACTAL_NZ", "FRACTAL_ZN_L2"}
)

_OP_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ATTR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def as_list(value: Any) -> List[Any]:
    """把标量规整为单元素列表；None 视为空列表；列表/元组原样返回。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_dtype(value: Any) -> str:
    """dtype 归一：去空白并小写。未知值保留，交校验阶段裁决。"""
    return str(value).strip().lower()


def normalize_format(value: Any) -> str:
    """format 归一：去空白并大写。未知值保留，交校验阶段裁决。"""
    return str(value).strip().upper()


def _check_identifier(value: str, what: str) -> None:
    if not value:
        raise OpSpecError(f"{what} 不能为空", hint="请提供非空名称")
    if not _OP_TYPE_RE.match(value):
        raise OpSpecError(
            f"{what} '{value}' 不合法",
            hint="须为字母/下划线开头，仅含字母、数字、下划线（会用于生成文件名/类名）",
        )


def camel_to_snake(name: str) -> str:
    """PascalCase/camelCase → snake_case（官方 op_type → 文件名/函数名规则）。

    例：AddCustomTemplate → add_custom_template。连续大写按最后一个大写字母切分。
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# TensorSpec / AttrSpec
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """单个输入/输出张量描述。字段名对齐官方原型 JSON 条目。

    ``format`` 与 ``type`` 是并行数组：例如 format=["ND","ND"]、type=["float16","float"]
    表示该张量支持 (ND, float16) 与 (ND, float) 两种组合。
    为方便使用，也允许在构造/YAML 中传单个字符串，会自动转为单元素列表。
    """

    name: str
    type: List[str] = field(default_factory=lambda: ["float"])
    format: List[str] = field(default_factory=lambda: ["ND"])
    param_type: str = ParamType.REQUIRED.value
    #: 可选：算子形状提示（如 [1024, 1024]，支持 -1 动态）。仅作元信息，不进入 msopgen 命令。
    shape: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.type = [normalize_dtype(t) for t in as_list(self.type)] or ["float"]
        self.format = [normalize_format(f) for f in as_list(self.format)] or ["ND"]
        self.param_type = str(self.param_type).strip().lower()
        if self.shape is not None:
            self.shape = list(self.shape)
        # 友好广播：一边为 1 时扩展到另一边长度。
        # 例：type=[float16, float] 且未写 format → format 自动广播为 [ND, ND]。
        if len(self.type) == 1 and len(self.format) > 1:
            self.type = self.type * len(self.format)
        elif len(self.format) == 1 and len(self.type) > 1:
            self.format = self.format * len(self.type)

    # -- 便捷只读属性（dtypes/formats 与官方字段 type/format 同义）--
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
                f"param_type '{self.param_type}' 不合法",
                field_path=path,
                hint=f"取值应为 {ParamType.REQUIRED.value} 或 {ParamType.OPTIONAL.value}",
            )
        if len(self.format) != len(self.type):
            raise OpSpecError(
                f"format 数组长度({len(self.format)})与 type 数组长度({len(self.type)})不一致",
                field_path=path,
                hint="两者须等长，下标相同者构成一组 format+dtype 组合（如 format=['ND','ND'] 与 type=['float16','float']）",
            )
        for i, dt in enumerate(self.type):
            if dt not in SUPPORTED_DTYPES:
                raise OpSpecError(
                    f"type[{i}] '{dt}' 不在支持的 dtype 集合内",
                    field_path=path,
                    hint=f"合法取值示例：{', '.join(sorted(SUPPORTED_DTYPES))}",
                )
        for i, fm in enumerate(self.format):
            if fm not in SUPPORTED_FORMATS:
                raise OpSpecError(
                    f"format[{i}] '{fm}' 不在支持的 format 集合内",
                    field_path=path,
                    hint=f"合法取值示例：{', '.join(sorted(SUPPORTED_FORMATS))}",
                )

    def to_dict(self) -> dict:
        """转 dict：键序与官方原型 JSON 条目一致；单元素 type/format 仍以数组输出。"""
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
            raise OpSpecError("缺少必填字段", hint=f"张量条目须含 'name'（缺少 {exc.args[0]}）") from exc
        spec.validate()
        return spec


@dataclass
class AttrSpec:
    """标量属性描述，对齐官方原型 JSON ``attr_desc`` 条目（name/param_type/type/value）。"""

    name: str
    type: str = "int"  # 属性类型字符串（如 int/float/bool/string/listInt...），不强制白名单
    value: Any = None
    param_type: str = ParamType.REQUIRED.value

    def __post_init__(self) -> None:
        self.type = str(self.type).strip()
        self.param_type = str(self.param_type).strip().lower()

    def validate(self, *, field_path: str = "") -> None:
        path = f"{field_path}.{self.name}" if field_path else self.name
        _check_identifier(self.name, path)
        if not self.type:
            raise OpSpecError("属性 type 不能为空", field_path=path)
        if self.param_type not in (ParamType.REQUIRED.value, ParamType.OPTIONAL.value):
            raise OpSpecError(
                f"param_type '{self.param_type}' 不合法",
                field_path=path,
                hint=f"取值应为 {ParamType.REQUIRED.value} 或 {ParamType.OPTIONAL.value}",
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
            raise OpSpecError("缺少必填字段", hint=f"属性条目须含 'name'（缺少 {exc.args[0]}）") from exc
        spec.validate()
        return spec


# ---------------------------------------------------------------------------
# OpSpec
# ---------------------------------------------------------------------------

@dataclass
class OpSpec:
    """一个算子的完整元信息。

    关键字段：
    - ``op_type``：算子类型名（PascalCase，如 AddCustomTemplate / Sigmoid）；
    - ``soc_version``：基础昇腾 SoC 版本（如 ``ascend910b1``），msopgen 的 ``-c`` 参数
      在命令生成层拼成 ``ai_core-<soc_version>``（见 msopgen.py / official-patterns §1.2）；
    - ``language``：官方 JSON 可选字段，固定 ``cpp``（Ascend C）；
    - ``inputs`` / ``outputs``：输入/输出 TensorSpec 列表；
    - ``attrs``：标量属性；
    - ``tiling``：Tiling 预留 dict（本轮不做策略，仅为后续轮扩展占位）；
    - ``description``：算子一句话描述。
    """

    op_type: str
    soc_version: str = "ascend910b1"
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    attrs: List[AttrSpec] = field(default_factory=list)
    tiling: dict = field(default_factory=dict)
    language: str = "cpp"
    description: str = ""
    #: 元信息：来源文件路径（由 load_op_spec 注入；不参与 YAML 序列化）
    source: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.op_type = str(self.op_type).strip()
        self.soc_version = str(self.soc_version).strip()
        self.language = str(self.language).strip().lower() or "cpp"

    # -- 便捷只读属性 --
    @property
    def op_name_snake(self) -> str:
        """snake_case 形式，对应官方 op_type → 文件/函数名规则（如 AddCustomTemplate → add_custom_template）。"""
        return camel_to_snake(self.op_type)

    def validate(self) -> None:
        _check_identifier(self.op_type, "op_type")
        if not self.soc_version:
            raise OpSpecError("soc_version 不能为空", hint="如 ascend910b1（msopgen -c 会拼为 ai_core-ascend910b1）")
        if not _SOC_RE.match(self.soc_version):
            raise OpSpecError(
                f"soc_version '{self.soc_version}' 不合法",
                hint="仅含字母/数字/下划线/连字符，且字母开头；不必带 ai_core- 前缀",
            )
        if self.language not in ("cpp",):
            raise OpSpecError(f"language '{self.language}' 不合法", hint="当前仅支持 cpp（Ascend C）")

        seen: dict = {}
        for kind in ("inputs", "outputs"):
            for idx, tensor in enumerate(getattr(self, kind)):
                tensor.validate(field_path=f"{kind}[{idx}]")
                key = tensor.name
                if key in seen:
                    raise OpSpecError(
                        f"名称 '{key}' 重复",
                        field_path=f"{kind}[{idx}]",
                        hint="算子所有输入/输出的 name 必须唯一",
                    )
                seen[key] = kind
        if not self.outputs:
            raise OpSpecError("outputs 不能为空", hint="算子至少需要一个输出张量")

        for idx, attr in enumerate(self.attrs):
            attr.validate(field_path=f"attrs[{idx}]")
            if attr.name in seen:
                raise OpSpecError(
                    f"名称 '{attr.name}' 重复",
                    field_path=f"attrs[{idx}]",
                    hint="属性名不得与输入/输出张量重名",
                )

    # -- 序列化 --
    def to_dict(self) -> dict:
        """输出顺序稳定：op_type → soc_version → language → inputs → outputs → attrs → tiling → description。"""
        data: dict = {
            "op_type": self.op_type,
            "soc_version": self.soc_version,
        }
        if self.language != "cpp":
            data["language"] = self.language
        data["inputs"] = [t.to_dict() for t in self.inputs]
        data["outputs"] = [t.to_dict() for t in self.outputs]
        if self.attrs:
            data["attrs"] = [a.to_dict() for a in self.attrs]
        if self.tiling:
            data["tiling"] = self.tiling
        if self.description:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "OpSpec":
        if not isinstance(mapping, dict):
            raise OpSpecError("顶层应为 YAML 映射（键值对）", hint="请检查 YAML 结构，示例见 examples/add.yaml")
        missing = [k for k in ("op_type",) if not mapping.get(k)]
        if missing:
            raise OpSpecError("缺少必填字段", hint=f"顶层须含 'op_type'（缺少 {', '.join(missing)}）")

        spec = cls(
            op_type=str(mapping["op_type"]).strip(),
            soc_version=str(mapping.get("soc_version", "ascend910b1")).strip(),
            inputs=[TensorSpec.from_dict(m) for m in mapping.get("inputs", [])],
            outputs=[TensorSpec.from_dict(m) for m in mapping.get("outputs", [])],
            attrs=[AttrSpec.from_dict(m) for m in mapping.get("attrs", [])],
            tiling=dict(mapping.get("tiling", {}) or {}),
            language=str(mapping.get("language", "cpp")).strip().lower() or "cpp",
            description=str(mapping.get("description", "")).strip(),
        )
        spec.validate()
        return spec
