"""cann_ophelper —— Windows 本地运行的 CANN Ascend C 算子工程模板生成助手。

本项目不编译、不运行任何 C++ 代码；编译验证均在云端 CANN Lab 完成。
"""

__version__ = "0.1.0"

from .model import AttrSpec, OpSpec, OpSpecError, ParamType, TensorSpec

__all__ = ["__version__", "OpSpec", "TensorSpec", "AttrSpec", "ParamType", "OpSpecError"]
