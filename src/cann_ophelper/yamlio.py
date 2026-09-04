"""cann_ophelper.yamlio —— OpSpec 与 YAML 之间的持久化。

职责：
- ``load_op_spec(path)``：读取 YAML → 校验 → 返回 ``OpSpec``；
- ``dump_op_spec(spec, path)``：``OpSpec`` → 校验 → 块式 YAML 落盘（UTF-8、字段序稳定）。

错误处理：缺失文件、YAML 语法错、结构/字段非法，均抛带上下文与修正建议的
``OpSpecError``（``ValueError`` 子类），为后续 typer/rich 渲染预留结构化字段。

依赖：PyYAML（声明于 pyproject；仅在真实环境/测试中执行，不阻塞纯语法检查）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .model import OpSpec, OpSpecError

__all__ = ["load_op_spec", "dump_op_spec", "op_spec_to_yaml_text", "yaml_text_to_op_spec"]

#: YAML 允许非顶层非字符串键；为友好报错将文档根部限制为映射。
_FILE_ENCODING = "utf-8"


def yaml_text_to_op_spec(text: str) -> OpSpec:
    """把 YAML 文本解析为 OpSpec。语法或语义错误均转成 OpSpecError。"""
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OpSpecError(f"YAML 语法错误：{exc}", hint="请检查引号、缩进与冒号；示例见 examples/add.yaml") from exc
    if raw is None:
        raise OpSpecError("YAML 内容为空", hint="至少需提供 op_type；示例见 examples/add.yaml")
    try:
        return OpSpec.from_dict(raw)
    except OpSpecError:
        raise
    except Exception as exc:  # noqa: BLE001 —— 防御性兜底，统一错误面
        raise OpSpecError(f"无法解析算子描述：{exc}", hint="请对照 examples/add.yaml 检查字段类型") from exc


def op_spec_to_yaml_text(spec: OpSpec, *, sort_keys: bool = False) -> str:
    """把 OpSpec 序列化为块式 YAML 文本。

    - 校验优先：非法模型不落盘；
    - ``sort_keys=False``：保持模型 to_dict 的稳定字段序，便于 diff 与人工阅读。
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
    """从文件加载并校验算子描述 YAML。"""
    p = Path(path)
    if not p.exists():
        raise OpSpecError(f"文件不存在：{p}", hint="请检查路径，或在 YAML 所在目录执行命令")
    if not p.is_file():
        raise OpSpecError(f"路径不是文件：{p}", hint="请提供一个 YAML 文件路径")
    try:
        text = p.read_text(encoding=_FILE_ENCODING)
    except OSError as exc:
        raise OpSpecError(f"读取文件失败：{p}（{exc.strerror or exc}）", hint="请检查文件是否可读") from exc
    spec = yaml_text_to_op_spec(text)
    spec.source = str(p)  # type: ignore[attr-defined]
    return spec

def dump_op_spec(spec: OpSpec, path: Union[str, Path]) -> Path:
    """将 OpSpec 校验后以块式 YAML 写入文件；父目录不存在时自动创建。"""
    p = Path(path)
    text = op_spec_to_yaml_text(spec)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding=_FILE_ENCODING)
    except OSError as exc:
        raise OpSpecError(f"写入文件失败：{p}（{exc.strerror or exc}）", hint="请检查目标目录权限") from exc
    return p
