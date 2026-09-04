"""cann_ophelper.template.naming -- Pure naming helpers for generated C++ code.

These helpers derive the identifier tokens used by the op_kernel / op_host
templates: snake_case file/function names, the Kernel class name, the shared
tiling struct name, the dtype template/macro aliases and the tiling include
guard.

The rules are provisional and mirror the official AddCustomTemplate msopgen
sample of chapter 03 (see docs/official-patterns.md SS2/SS6): the Kernel class
is named ``Kernel`` + the op core name (the op type with a trailing
``CustomTemplate``/``Custom`` suffix removed, e.g. AddCustomTemplate ->
KernelAdd) and the shared tiling struct is the constant ``TilingDataTemplate``.

All functions are pure; callers pass identifiers that model validation has
already checked.
"""

from __future__ import annotations

from ..model import camel_to_snake

__all__ = [
    "KERNEL_SUFFIXES",
    "TILING_STRUCT_NAME",
    "dtype_alias",
    "kernel_class",
    "macro_alias",
    "op_snake",
    "tiling_guard",
    "tiling_struct",
]

#: Suffixes msopgen keeps off the generated Kernel class name in the
#: chapter-03 sample route (AddCustomTemplate -> KernelAdd).
KERNEL_SUFFIXES: tuple = ("CustomTemplate", "Custom")

#: Tiling struct name of the msopgen sample route; shared by op_host
#: (writer) and op_kernel (reader). A provisional constant, not derived from
#: the op type (see docs/official-patterns.md SS6).
TILING_STRUCT_NAME = "TilingDataTemplate"


def op_snake(op_type: str) -> str:
    """Return the snake_case file/function form of ``op_type``.

    E.g. ``AddCustomTemplate`` -> ``add_custom_template`` (the official
    op_type -> file/function rule of msopgen).
    """
    return camel_to_snake(op_type)


def kernel_class(op_type: str) -> str:
    """Return the generated Kernel class name for ``op_type``.

    The chapter-03 sample names the class ``KernelAdd`` for op type
    ``AddCustomTemplate``: a trailing ``CustomTemplate`` (or ``Custom``)
    suffix is stripped and ``Kernel`` is prepended to the remaining core
    name. When no known suffix is present the whole ``op_type`` is used.
    """
    core = op_type
    for suffix in KERNEL_SUFFIXES:
        if op_type.endswith(suffix) and len(op_type) > len(suffix):
            core = op_type[: -len(suffix)]
            break
    return f"Kernel{core}"


def tiling_struct() -> str:
    """Return the tiling struct name shared by host and kernel templates."""
    return TILING_STRUCT_NAME


def dtype_alias(name: str) -> str:
    """Return the lower-case template dtype parameter for tensor ``name``.

    E.g. tensor ``x`` -> ``dtypeX`` (the ``<class dtypeX>`` template
    parameter used in the official kernel skeleton).
    """
    return f"dtype{name[:1].upper()}{name[1:]}"


def macro_alias(name: str) -> str:
    """Return the build-system dtype macro for tensor ``name``.

    E.g. tensor ``x`` -> ``DTYPE_X``. Official builds inject DTYPE_* macros
    per data type; templates emit the macro name, never an expanded type.
    """
    return f"DTYPE_{name.upper()}"


def tiling_guard(op_snake_value: str) -> str:
    """Return the include-guard macro for the tiling header of ``op_snake_value``.

    E.g. ``add_custom_template`` -> ``ADD_CUSTOM_TEMPLATE_TILING_H``.
    """
    return f"{op_snake_value.upper()}_TILING_H"
