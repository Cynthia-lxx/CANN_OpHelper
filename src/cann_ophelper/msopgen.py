"""cann_ophelper.msopgen —— 生成 msopgen 命令行与云端执行说明。

命令格式依据官方文档（见 docs/official-patterns.md §1.2，出处 03.02 章）：

    msopgen gen -i <原型JSON> -c ai_core-<soc> -lan cpp -out <输出目录>

本模块只做**纯文本拼装**，不做文件系统副作用；原型 JSON 由用户提供（工具不生成/不解析
JSON 内容，仅引用其路径——规约 6.4）。
"""

from __future__ import annotations

import os
import shlex
from typing import Union

from .model import OpSpec, OpSpecError

__all__ = [
    "MSOPGEN_SOC_PREFIX",
    "format_soc_for_msopgen",
    "shell_quote",
    "build_msopgen_command",
    "show_cloud_instructions",
]

#: msopgen ``-c`` 参数的固定前缀（官方写法 ai_core-ascend910b1）。
MSOPGEN_SOC_PREFIX = "ai_core-"


def format_soc_for_msopgen(soc_version: str) -> str:
    """把基础 soc（如 ascend910b1）规整为 msopgen -c 取值（ai_core-ascend910b1）。

    若调用方已给出带 ``ai_core-`` 前缀的值则原样返回，避免重复前缀。
    """
    soc = soc_version.strip()
    if soc.lower().startswith(MSOPGEN_SOC_PREFIX):
        return soc
    return f"{MSOPGEN_SOC_PREFIX}{soc}"


def shell_quote(value: str) -> str:
    """对进入命令行的路径做 shell 安全处理：含特殊字符时加单引号（POSIX 规则）。

    官方样例（云端 Linux/CANN 环境执行）多为相对路径，如 Sources/03.02/add_custom.json；
    仅当路径含空格等字符时加引号，保证「一条命令即可复制执行」。
    """
    if not value:
        return "''"
    if any(ch.isspace() or ch in "\"'\\$`;&|<>()*?[]{}~#" for ch in value):
        return shlex.quote(value)
    return value


def build_msopgen_command(
    spec: OpSpec,
    proto_json: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
    language: str = "cpp",
) -> str:
    """依据算子元信息拼装一条完整 msopgen 命令。

    :param spec: 已校验的 OpSpec（op_type/soc_version 仅用于 soc 拼装与提示，命令本身
        不依赖 op_type——算子名来自用户原型 JSON）；
    :param proto_json: 算子原型 JSON 路径（msopgen ``-i``），本地路径或云端相对路径均可；
    :param out_dir: 工程输出目录（``-out``）；
    :param language: 开发语言，当前仅支持官方写法 cpp；
    :returns: 一条可复制到云端执行的命令字符串。
    """
    spec.validate()
    lang = language.strip().lower()
    if lang != "cpp":
        raise OpSpecError(
            f"language '{language}' 不合法",
            hint="官方 msopgen 的 -lan 仅使用 cpp（Ascend C/C++），见 official-patterns §1.2",
        )
    json_arg = shell_quote(str(proto_json))
    soc_arg = shell_quote(format_soc_for_msopgen(spec.soc_version))
    out_arg = shell_quote(str(out_dir))
    return f"msopgen gen -i {json_arg} -c {soc_arg} -lan {lang} -out {out_arg}"


def show_cloud_instructions(
    spec: OpSpec,
    proto_json: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
) -> str:
    """生成配套的云端执行说明（供 CLI/README 引用，纯文本）。

    前提事实（official-patterns §5）：msopgen 须在已安装 CANN Toolkit 的云端环境执行；
    生成工程内含 op_host/op_kernel 等，拷回本地后可被本工具后续轮读取与填充。
    """
    cmd = build_msopgen_command(spec, proto_json, out_dir)
    return "\n".join(
        [
            "请按以下步骤在云端 CANN 环境完成工程生成：",
            "",
            f"  1. 确保算子原型 JSON 与工程输出目录规划就绪（原型 JSON 需自行准备，本工具不生成）。",
            f"  2. 执行以下命令（已按算子元信息拼装）：",
            f"     {cmd}",
            f"  3. 确认 {out_dir} 下已生成工程（含 op_host/op_kernel 等，参见 official-patterns §1.4）。",
            "  4. 将生成的整个工程目录复制回本地，供 CANN_OpHelper 后续读取与填充。",
            "",
            "提示：命令中的 soc 已拼为 msopgen 规范格式 "
            f"'{format_soc_for_msopgen(spec.soc_version)}'；如与你的云端环境不符，可手动调整。",
        ]
    )
