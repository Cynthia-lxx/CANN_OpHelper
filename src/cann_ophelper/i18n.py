"""cann_ophelper.i18n — Lightweight bilingual message catalog (Simplified Chinese / English).

Language policy (see .codebuddy/rules/Language.mdc):
- Code comments/docstrings are written in English.
- User-facing messages (errors, hints, cloud instructions) are resolved from this
  catalog so callers can switch the display language at runtime.
- Default language is ``zh``; it can be changed by setting the ``CANN_OPHELPER_LANG``
  environment variable before importing this module, or at runtime with
  :func:`set_language`. English (``en``) is used as the fallback when a key is
  missing in the active language.

Templates use ``str.format`` placeholders, e.g. ``{path}`` or ``{index}``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "set_language",
    "get_language",
    "t",
]

#: Languages shipped with this catalog.
SUPPORTED_LANGUAGES = ("zh", "en")

#: Environment variable used to pick the initial language.
ENV_LANGUAGE = "CANN_OPHELPER_LANG"

#: Fallback language when a key is missing in the active language.
FALLBACK_LANGUAGE = "en"

#: Chinese (Simplified) message templates. ``{placeholder}`` are formatted by t().
_ZH: Dict[str, str] = {
    # -- OpSpecError message assembly --
    "msg.hint_join": " 建议：",
    # -- identifier / name checks (model) --
    "check.name_empty": "{what} 不能为空",
    "check.name_empty.hint": "请提供非空名称",
    "check.name_invalid": "{what} '{value}' 不合法",
    "check.name_invalid.hint": "须为字母/下划线开头，仅含字母、数字、下划线（会用于生成文件名/类名）",
    "check.param_type_invalid": "param_type '{value}' 不合法",
    "check.param_type_invalid.hint": "取值应为 required 或 optional",
    # -- tensor type/format checks (model) --
    "check.type_format_len": "format 数组长度({fmt_len})与 type 数组长度({type_len})不一致",
    "check.type_format_len.hint": "两者须等长，下标相同者构成一组 format+dtype 组合（如 format=['ND','ND'] 与 type=['float16','float']）",
    "check.dtype_unsupported": "type[{index}] '{dtype}' 不在支持的 dtype 集合内",
    "check.format_unsupported": "format[{index}] '{fmt}' 不在支持的 format 集合内",
    "check.supported_values.hint": "合法取值示例：{values}",
    # -- missing required fields (model) --
    "check.missing_required": "缺少必填字段",
    "check.tensor_needs_name.hint": "张量条目须含 'name'（缺少 {key}）",
    "check.attr_needs_name.hint": "属性条目须含 'name'（缺少 {key}）",
    # -- attr checks (model) --
    "check.attr_type_empty": "属性 type 不能为空",
    # -- soc / language checks (model) --
    "check.soc_empty": "soc_version 不能为空",
    "check.soc_empty.hint": "如 ascend910b1（msopgen -c 会拼为 ai_core-ascend910b1）",
    "check.soc_invalid": "soc_version '{value}' 不合法",
    "check.soc_invalid.hint": "仅含字母/数字/下划线/连字符，且字母开头；不必带 ai_core- 前缀",
    "check.language_invalid": "language '{value}' 不合法",
    "check.language_model.hint": "当前仅支持 cpp（Ascend C）",
    "check.language_msopgen.hint": "官方 msopgen 的 -lan 仅使用 cpp（Ascend C/C++），见 official-patterns §1.2",
    # -- duplicates / outputs (model) --
    "check.dup_name": "名称 '{name}' 重复",
    "check.dup_name_tensor.hint": "算子所有输入/输出的 name 必须唯一",
    "check.dup_name_attr.hint": "属性名不得与输入/输出张量重名",
    "check.outputs_empty": "outputs 不能为空",
    "check.outputs_empty.hint": "算子至少需要一个输出张量",
    # -- top-level mapping (model) --
    "check.top_mapping": "顶层应为 YAML 映射（键值对）",
    "check.top_mapping.hint": "请检查 YAML 结构，示例见 examples/add.yaml",
    "check.missing_op_type.hint": "顶层须含 'op_type'（缺少 {keys}）",
    # -- yaml I/O (yamlio) --
    "yamlio.syntax": "YAML 语法错误：{err}",
    "yamlio.syntax.hint": "请检查引号、缩进与冒号；示例见 examples/add.yaml",
    "yamlio.empty": "YAML 内容为空",
    "yamlio.empty.hint": "至少需提供 op_type；示例见 examples/add.yaml",
    "yamlio.parse": "无法解析算子描述：{err}",
    "yamlio.parse.hint": "请对照 examples/add.yaml 检查字段类型",
    "yamlio.file_missing": "文件不存在：{path}",
    "yamlio.file_missing.hint": "请检查路径，或在 YAML 所在目录执行命令",
    "yamlio.not_file": "路径不是文件：{path}",
    "yamlio.not_file.hint": "请提供一个 YAML 文件路径",
    "yamlio.read_fail": "读取文件失败：{path}（{reason}）",
    "yamlio.read_fail.hint": "请检查文件是否可读",
    "yamlio.write_fail": "写入文件失败：{path}（{reason}）",
    "yamlio.write_fail.hint": "请检查目标目录权限",
    # -- cloud instructions (msopgen) --
    "ci.title": "请按以下步骤在云端 CANN 环境完成工程生成：",
    "ci.step1": "  1. 确保算子原型 JSON 与工程输出目录规划就绪（原型 JSON 需自行准备，本工具不生成）。",
    "ci.step2": "  2. 执行以下命令（已按算子元信息拼装）：",
    "ci.step3": "  3. 确认 {out_dir} 下已生成工程（含 op_host/op_kernel 等，参见 official-patterns §1.4）。",
    "ci.step4": "  4. 将生成的整个工程目录复制回本地，供 CANN_OpHelper 后续读取与填充。",
    "ci.tip": "提示：命令中的 soc 已拼为 msopgen 规范格式 '{soc}'；如与你的云端环境不符，可手动调整。",
    # -- template maps (template/maps) --
    "tmpl.dtype_unmapped": "type '{dtype}' 未收录到 ge::DT_* 映射表",
    "tmpl.dtype_unmapped.hint": "本轮模板仅收录官方样例确认的写法（float16/float）；若为合法 dtype，请先在 template/maps.py 登记并补充 docs/official-patterns.md 出处",
    "tmpl.format_unmapped": "format '{fmt}' 未收录到 ge::FORMAT_* 映射表",
    "tmpl.format_unmapped.hint": "本轮模板仅收录官方样例确认的 ND；若为合法 format，请先在 template/maps.py 登记并注明出处",
    "tmpl.soc_unmapped": "soc_version '{soc}' 未收录到 AddConfig 映射表",
    "tmpl.soc_unmapped.hint": "本轮仅收录官方样例的 ascend910b1→ascend910b；请参照 docs/official-patterns §3.3 核对你的 soc 对应的 AddConfig 写法后再登记",
}

#: English message templates.
_EN: Dict[str, str] = {
    "msg.hint_join": " Hint: ",
    "check.name_empty": "{what} must not be empty",
    "check.name_empty.hint": "Provide a non-empty name",
    "check.name_invalid": "Invalid {what} '{value}'",
    "check.name_invalid.hint": "Must start with a letter/underscore and contain only letters, digits and underscores (used for generated file/class names)",
    "check.param_type_invalid": "Invalid param_type '{value}'",
    "check.param_type_invalid.hint": "Value must be 'required' or 'optional'",
    "check.type_format_len": "format array length ({fmt_len}) does not match type array length ({type_len})",
    "check.type_format_len.hint": "Both arrays must be equal in length; entries at the same index form a format+dtype pair (e.g. format=['ND','ND'] and type=['float16','float'])",
    "check.dtype_unsupported": "type[{index}] '{dtype}' is not in the supported dtype set",
    "check.format_unsupported": "format[{index}] '{fmt}' is not in the supported format set",
    "check.supported_values.hint": "Valid values include: {values}",
    "check.missing_required": "Missing required field",
    "check.tensor_needs_name.hint": "A tensor entry must contain 'name' (missing {key})",
    "check.attr_needs_name.hint": "An attribute entry must contain 'name' (missing {key})",
    "check.attr_type_empty": "Attribute 'type' must not be empty",
    "check.soc_empty": "soc_version must not be empty",
    "check.soc_empty.hint": "e.g. ascend910b1 (msopgen -c will prefix it as ai_core-ascend910b1)",
    "check.soc_invalid": "Invalid soc_version '{value}'",
    "check.soc_invalid.hint": "Must start with a letter and contain only letters/digits/underscores/hyphens; no ai_core- prefix needed",
    "check.language_invalid": "Invalid language '{value}'",
    "check.language_model.hint": "Only cpp (Ascend C) is supported",
    "check.language_msopgen.hint": "Official msopgen '-lan' only accepts cpp (Ascend C/C++); see official-patterns §1.2",
    "check.dup_name": "Duplicate name '{name}'",
    "check.dup_name_tensor.hint": "All input/output tensor names must be unique",
    "check.dup_name_attr.hint": "Attribute names must not collide with input/output tensor names",
    "check.outputs_empty": "outputs must not be empty",
    "check.outputs_empty.hint": "An operator needs at least one output tensor",
    "check.top_mapping": "Top level must be a YAML mapping (key-value pairs)",
    "check.top_mapping.hint": "Check the YAML structure; see examples/add.yaml",
    "check.missing_op_type.hint": "Top level must contain 'op_type' (missing {keys})",
    "yamlio.syntax": "YAML syntax error: {err}",
    "yamlio.syntax.hint": "Check quotes, indentation and colons; see examples/add.yaml",
    "yamlio.empty": "YAML content is empty",
    "yamlio.empty.hint": "At least op_type is required; see examples/add.yaml",
    "yamlio.parse": "Cannot parse the operator spec: {err}",
    "yamlio.parse.hint": "Check field types against examples/add.yaml",
    "yamlio.file_missing": "File not found: {path}",
    "yamlio.file_missing.hint": "Check the path, or run the command from the YAML directory",
    "yamlio.not_file": "Path is not a file: {path}",
    "yamlio.not_file.hint": "Provide a YAML file path",
    "yamlio.read_fail": "Failed to read file: {path} ({reason})",
    "yamlio.read_fail.hint": "Check that the file is readable",
    "yamlio.write_fail": "Failed to write file: {path} ({reason})",
    "yamlio.write_fail.hint": "Check write permission of the target directory",
    "ci.title": "Follow these steps to generate the project in a cloud CANN environment:",
    "ci.step1": "  1. Prepare the operator prototype JSON and plan the output directory (the JSON is supplied by you; this tool does not generate it).",
    "ci.step2": "  2. Run the following command (assembled from the operator metadata):",
    "ci.step3": "  3. Confirm the project (op_host/op_kernel etc.; see official-patterns §1.4) was created under {out_dir}.",
    "ci.step4": "  4. Copy the generated project directory back to local for CANN_OpHelper to read and fill in.",
    "ci.tip": "Tip: the soc has been formatted as '{soc}' for msopgen; adjust it manually if it does not match your cloud environment.",
    # -- template maps (template/maps) --
    "tmpl.dtype_unmapped": "type '{dtype}' has no ge::DT_* mapping",
    "tmpl.dtype_unmapped.hint": "Only dtypes confirmed by the official sample are registered (float16/float) in this phase; to add one, register it in template/maps.py and record its source in docs/official-patterns.md",
    "tmpl.format_unmapped": "format '{fmt}' has no ge::FORMAT_* mapping",
    "tmpl.format_unmapped.hint": "Only the official sample format ND is registered in this phase; to add one, register it in template/maps.py and note its source",
    "tmpl.soc_unmapped": "soc_version '{soc}' has no AddConfig mapping",
    "tmpl.soc_unmapped.hint": "Only the official sample mapping ascend910b1->ascend910b is registered; check the AddConfig form for your soc in docs/official-patterns SS3.3 before registering",
}

_CATALOG: Dict[str, Dict[str, str]] = {"zh": _ZH, "en": _EN}

#: Language in effect. Read once at import time; switch later via set_language().
DEFAULT_LANGUAGE = os.environ.get(ENV_LANGUAGE, "zh").strip().lower()
if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    DEFAULT_LANGUAGE = "zh"

_language: str = DEFAULT_LANGUAGE


def set_language(language: str) -> None:
    """Switch the display language used by :func:`t`.

    :param language: One of ``SUPPORTED_LANGUAGES`` (``zh`` / ``en``).
    :raises ValueError: If the language is not supported.
    """
    global _language
    lang = str(language).strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}; choose from {SUPPORTED_LANGUAGES}")
    _language = lang


def get_language() -> str:
    """Return the language currently in effect."""
    return _language


def t(key: str, **params: Any) -> str:
    """Resolve a message template for the active language.

    Lookup order: active language → fallback (English) → the raw ``key``.
    Placeholders in the template are filled from ``params`` when provided;
    if formatting fails the raw template is returned unchanged.
    """
    table = _CATALOG.get(get_language()) or {}
    template = table.get(key) or _CATALOG.get(FALLBACK_LANGUAGE, {}).get(key) or key
    if not params:
        return template
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template
