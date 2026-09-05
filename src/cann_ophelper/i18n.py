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
    # -- prototype JSON export (proto) --
    "proto.attr_unsupported": "该 OpSpec 声明了 {count} 个 attrs，暂不支持导出原型 JSON",
    "proto.attr_unsupported.hint": "官方文档样例中尚无 attrs 在原型 JSON 里的写法，本工具不臆造格式；请先去除 attrs 或手工编写原型文件",
    "proto.write_fail": "写入原型 JSON 失败：{path}（{reason}）",
    "proto.write_fail.hint": "请检查目标目录权限",
    # -- cloud instructions (msopgen) --
    "ci.title": "请按以下步骤在云端 CANN 环境完成工程生成：",
    "ci.step1": "  1. 准备算子原型 JSON 与工程输出目录：原型 JSON 可手工编写，或用 new-op 向导生成 spec YAML 后经 gen-msopgen --proto-out 导出。",
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
    # -- CLI surface (cli) --
    "cli.error.title": "错误：",
    "cli.overview.title": "算子元信息概览",
    "cli.overview.op_type": "算子类型",
    "cli.overview.op_snake": "snake 名称",
    "cli.overview.soc": "SoC 版本",
    "cli.overview.desc": "描述",
    "cli.overview.none": "(无)",
    "cli.tensor.inputs": "输入张量",
    "cli.tensor.outputs": "输出张量",
    "cli.tensor.no_shape": "(未给)",
    "cli.cmd.title": "msopgen 命令（复制到云端执行）",
    "cli.render.title_preview": "渲染产物预览（未写盘）",
    "cli.render.title_wrote": "已写盘 {count} 个文件 → {root}",
    "cli.render.col_file": "文件",
    "cli.render.col_bytes": "大小",
    "cli.render.col_status": "状态",
    "cli.render.written": "写入",
    "cli.render.overwritten": "覆盖",
    "cli.render.no_out": "未指定 --out：本次未写盘。",
    "cli.render.dry_note": "--dry-run：仅预览，未写盘。",
    "cli.render.suggest": "提示：指定 --out {suggest} 可写盘（覆盖该目录下的 op_host/op_kernel）。",
    # -- new-op command (cli) --
    "cli.new_op.preset_applied": "已应用预设 '{preset}'：直接回车即采用预填值，或输入新值。",
    "cli.new_op.confirm": "确认将算子描述写入 {path}？",
    "cli.new_op.cancelled": "已取消，未写盘。",
    "cli.new_op.written": "已写入算子描述：{path}",
    "cli.new_op.suggest": "下一步：cann-ophelper gen-msopgen {path} 预览 msopgen 命令；加 --proto-out 可直接导出原型 JSON。",
    # -- quickstart (cli) --
    "cli.quickstart.title": "从零到云端 CANN 工程（快速上手）",
    "cli.quickstart.body": "以下命令覆盖「需求 → YAML → 原型 JSON → 云端 msopgen 工程 → 本地渲染」全流程。\n将 {py} 替换为你的 Python 入口（如 penv 激活后直接 `cann-ophelper`）。\n\n**① 收集算子需求，生成 spec YAML**\n\n```text\n{py} new-op                        # 交互逐问 → 得到 myop.yaml\n{py} new-op --from add --yes --out add.yaml   # 或用内置 Add 预设\n```\n\n**② 导出原型 JSON，并得到 msopgen 命令**\n\n```text\n{py} gen-msopgen myop.yaml --proto-out myop.json\n```\n\n命令中的 `-i` 已自动指向 myop.json。\n\n**③ 在云端执行 msopgen**\n\n把 myop.json 与生成的工程上传到云端 CANN 环境，复制 ② 打印的命令执行：\n\n```text\nmsopgen gen -i myop.json -c ai_core-ascend910b1 -lan cpp -out out/MyOp\n```\n\n**④ 本地渲染填充，回传编译验证**\n\n```text\n{py} render myop.yaml --out <云端拷回的工程目录>\n```\n\n更多细节见 `gen-msopgen --help` 中的示例与 README。",
    # -- proto-out export (cli) --
    "cli.proto_out.written": "原型 JSON 已写入：",
    "cli.proto_out.suggest": "将该文件上传到云端后，可作为 msopgen 命令中 -i 指向的原型 JSON。",
    "cli.proto_demo.hint": "未指定 --proto/--proto-out：命令中的 -i 指向的是内置 Add 演示样例 {demo}，并非当前算子的原型 JSON。",
    "cli.proto_demo.action": "请用 --proto-out 从当前 spec 导出原型 JSON，或用 --proto 指向你已有的原型文件；仅作演示时可忽略本提示。",
    "cli.proto_missing.hint": "已自动让命令中的 -i 指向 --proto-out 导出的文件。",
    "cli.proto_mismatch.hint": "--proto 与 --proto-out 不一致：导出文件名为 {exported}，但命令 -i 仍指向 --proto 指定的文件。",
    "cli.proto_mismatch.action": "若要在云端使用刚导出的原型，请改传 --proto {proto} 或去掉 --proto。",
    # -- new-op wizard (wizard) --
    "wizard.prompt.op_type": "算子类型名（PascalCase，如 AddCustomTemplate）",
    "wizard.prompt.soc": "SoC 基础版本（如 ascend910b1；msopgen -c 会拼为 ai_core-ascend910b1）",
    "wizard.prompt.desc": "一句话描述（可留空）",
    "wizard.prompt.n_inputs": "输入张量数量（0-9）",
    "wizard.prompt.n_outputs": "输出张量数量（1-9）",
    "wizard.prompt.tensor_name": "{kind}张量 #{index} 的名称",
    "wizard.prompt.param_type": "{name} 的 param_type（required/optional）",
    "wizard.prompt.dtype_csv": "{name} 支持的 dtype，多个用英文逗号分隔（如 float16,float；留空=float）",
    "wizard.prompt.format_csv": "{name} 的对应 format，逗号分隔、与 dtype 一一对应（留空=ND）",
    "wizard.prompt.shape": "{name} 的 shape，逗号分隔、动态维用 -1（可留空跳过）",
    "wizard.kind.inputs": "输入",
    "wizard.kind.outputs": "输出",
    "wizard.combo.note": "官方原型中 format 与 type 是并行数组：下标相同的二者组成一种受支持的 dtype+format（如 type=[float16,float] 配 format=[ND,ND]）。format 只给一个时会自动广播到全部 dtype。",
    "wizard.err.op_type": "算子类型 '{value}' 不合法：须字母/下划线开头，仅含字母、数字、下划线",
    "wizard.err.soc": "soc_version '{value}' 不合法：须字母开头，仅含字母/数字/下划线/连字符",
    "wizard.err.count": "请输入 {lo}-{hi} 之间的整数",
    "wizard.err.name_empty": "张量名称不能为空",
    "wizard.err.name_dup": "名称 '{value}' 已被占用，请换一个",
    "wizard.err.dtype": "dtype '{value}' 不受支持。可用：{values}",
    "wizard.err.format": "format '{value}' 不受支持。可用：{values}",
    "wizard.err.shape": "shape 各项须为整数或 -1，逗号分隔",
    "wizard.err.unknown_preset": "未知预设 '{value}'。可用预设：{values}",
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
    # -- prototype JSON export (proto) --
    "proto.attr_unsupported": "This OpSpec declares {count} attrs; prototype JSON export is not supported yet",
    "proto.attr_unsupported.hint": "No official sample shows an attr entry layout in the prototype JSON, so this tool will not invent a format; drop the attrs or hand-write the prototype instead",
    "proto.write_fail": "Failed to write prototype JSON: {path} ({reason})",
    "proto.write_fail.hint": "Check write permission of the target directory",
    "ci.title": "Follow these steps to generate the project in a cloud CANN environment:",
    "ci.step1": "  1. Prepare the operator prototype JSON and the output directory: write it by hand, or run the new-op wizard to create a spec YAML and export it via `gen-msopgen --proto-out`.",
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
    # -- CLI surface (cli) --
    "cli.error.title": "Error: ",
    "cli.overview.title": "Operator metadata overview",
    "cli.overview.op_type": "Operator type",
    "cli.overview.op_snake": "Snake name",
    "cli.overview.soc": "SoC version",
    "cli.overview.desc": "Description",
    "cli.overview.none": "(none)",
    "cli.tensor.inputs": "Input tensors",
    "cli.tensor.outputs": "Output tensors",
    "cli.tensor.no_shape": "(unspecified)",
    "cli.cmd.title": "msopgen command (copy & run on the cloud)",
    "cli.render.title_preview": "Rendered artifacts (preview; nothing written)",
    "cli.render.title_wrote": "Wrote {count} files -> {root}",
    "cli.render.col_file": "File",
    "cli.render.col_bytes": "Size",
    "cli.render.col_status": "Status",
    "cli.render.written": "written",
    "cli.render.overwritten": "overwritten",
    "cli.render.no_out": "--out not given: nothing written.",
    "cli.render.dry_note": "--dry-run: preview only; nothing written.",
    "cli.render.suggest": "Tip: pass --out {suggest} to write (overwriting its op_host/op_kernel).",
    # -- new-op command (cli) --
    "cli.new_op.preset_applied": "Preset '{preset}' applied: press Enter to accept a prefilled value or type a new one.",
    "cli.new_op.confirm": "Write the operator spec to {path}?",
    "cli.new_op.cancelled": "Cancelled; nothing written.",
    "cli.new_op.written": "Operator spec written to: {path}",
    "cli.new_op.suggest": "Next: run `cann-ophelper gen-msopgen {path}` to preview the msopgen command; add --proto-out to also export the prototype JSON.",
    # -- quickstart (cli) --
    "cli.quickstart.title": "From zero to a cloud-ready CANN project",
    "cli.quickstart.body": "These commands cover the whole flow: requirement -> YAML -> prototype JSON -> cloud msopgen project -> local render.\nReplace {py} with your Python entry (e.g. `cann-ophelper` after activating penv).\n\n**1. Collect the operator spec as YAML**\n\n```text\n{py} new-op                        # interactive -> writes myop.yaml\n{py} new-op --from add --yes --out add.yaml   # or use the built-in add preset\n```\n\n**2. Export the prototype JSON and get the msopgen command**\n\n```text\n{py} gen-msopgen myop.yaml --proto-out myop.json\n```\n\nThe `-i` in the printed command already points at myop.json.\n\n**3. Run msopgen on the cloud**\n\nUpload myop.json (and later the project) to a cloud CANN environment, then run the command printed in step 2, e.g.:\n\n```text\nmsopgen gen -i myop.json -c ai_core-ascend910b1 -lan cpp -out out/MyOp\n```\n\n**4. Render locally, then compile on the cloud**\n\n```text\n{py} render myop.yaml --out <cloud project directory copied back>\n```\n\nSee the examples in `gen-msopgen --help` and the README for more.",
    # -- proto-out export (cli) --
    "cli.proto_out.written": "Prototype JSON written to:",
    "cli.proto_out.suggest": "Upload this file to the cloud, then point msopgen '-i' at it.",
    "cli.proto_demo.hint": "No --proto/--proto-out given: the '-i' in the command references the built-in add demo sample {demo}, which is NOT the prototype of the current operator.",
    "cli.proto_demo.action": "Export the prototype for this spec with --proto-out, or point --proto at a prototype file you already have; ignore this note only for a demo run.",
    "cli.proto_missing.hint": "Pointed the command '-i' at the file exported via --proto-out automatically.",
    "cli.proto_mismatch.hint": "--proto differs from --proto-out: the exported file is named {exported}, but the command '-i' still uses the file given by --proto.",
    "cli.proto_mismatch.action": "To use the freshly exported prototype on the cloud, pass --proto {proto} or drop --proto.",
    # -- new-op wizard (wizard) --
    "wizard.prompt.op_type": "Operator type (PascalCase, e.g. AddCustomTemplate)",
    "wizard.prompt.soc": "Base SoC version (e.g. ascend910b1; msopgen -c becomes ai_core-ascend910b1)",
    "wizard.prompt.desc": "One-line description (optional)",
    "wizard.prompt.n_inputs": "Number of input tensors (0-9)",
    "wizard.prompt.n_outputs": "Number of output tensors (1-9)",
    "wizard.prompt.tensor_name": "Name of {kind} tensor #{index}",
    "wizard.prompt.param_type": "param_type of {name} (required/optional)",
    "wizard.prompt.dtype_csv": "dtypes of {name}, comma-separated (e.g. float16,float; empty=float)",
    "wizard.prompt.format_csv": "formats of {name}, comma-separated, one per dtype (empty=ND)",
    "wizard.prompt.shape": "shape of {name}, comma-separated, dynamic dims as -1 (empty to skip)",
    "wizard.kind.inputs": "input",
    "wizard.kind.outputs": "output",
    "wizard.combo.note": "In the official prototype, format and type are parallel arrays: entries with the same index form one supported dtype+format pair (e.g. type=[float16,float] with format=[ND,ND]). A single format is broadcast to all dtypes.",
    "wizard.err.op_type": "Invalid operator type '{value}': must start with a letter/underscore and contain only letters, digits and underscores",
    "wizard.err.soc": "Invalid soc_version '{value}': must start with a letter and contain only letters/digits/underscores/hyphens",
    "wizard.err.count": "Please enter an integer between {lo} and {hi}",
    "wizard.err.name_empty": "Tensor name must not be empty",
    "wizard.err.name_dup": "Name '{value}' is already taken; choose another",
    "wizard.err.dtype": "dtype '{value}' is not supported. Available: {values}",
    "wizard.err.format": "format '{value}' is not supported. Available: {values}",
    "wizard.err.shape": "shape entries must be integers or -1, comma-separated",
    "wizard.err.unknown_preset": "Unknown preset '{value}'. Available presets: {values}",
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
