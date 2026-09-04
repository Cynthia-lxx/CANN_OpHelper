"""cann_ophelper.template.engine -- Jinja2 rendering entry point.

:class:`TemplateEngine` loads the package templates (whole-file .j2 files
aligned with the official AddCustomTemplate sample) through Jinja2's
``PackageLoader`` and renders the three op files for an :class:`OpSpec`:

- ``op_kernel/<op_snake>.cpp``      (kernel)
- ``op_kernel/<op_snake>_tiling.h`` (shared tiling header)
- ``op_host/<op_snake>.cpp``        (host)

Whitespace handling keeps templates verbatim: no trimming is applied and a
trailing newline is preserved (``keep_trailing_newline=True``), so rendered
output stays diff-able against the official sample files.

The engine depends only on the render context; templates contain no business
logic. Later rounds that split whole-file templates into apply snippets change
the templates and this mapping table, not ``context``/``maps``.
"""

from __future__ import annotations

from typing import Dict, Tuple

from jinja2 import Environment, PackageLoader

from ..model import OpSpec
from .context import build_render_context

__all__ = ["TEMPLATE_OUTPUTS", "TemplateEngine", "render"]

#: (package template path, output relpath pattern). Logical template names are
#: stable inside the package; the output file names follow the official
#: op_type -> snake_case rule.
TEMPLATE_OUTPUTS: Tuple[Tuple[str, str], ...] = (
    ("op_kernel/kernel.cpp.j2", "op_kernel/{op_snake}.cpp"),
    ("op_kernel/tiling.h.j2", "op_kernel/{op_snake}_tiling.h"),
    ("op_host/host.cpp.j2", "op_host/{op_snake}.cpp"),
)

#: Jinja2 package used by PackageLoader (cann_ophelper.template.templates).
_LOADER_PACKAGE = "cann_ophelper"
_LOADER_PATH = "template/templates"


class TemplateEngine:
    """Renders the whole-file templates for an operator spec."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader(_LOADER_PACKAGE, _LOADER_PATH),
            keep_trailing_newline=True,
            autoescape=False,
        )
        # trim_blocks removes only the newline that directly follows a {% %}
        # block tag (e.g. a helper {% set %} line), so templates stay aligned
        # with the official files; lstrip_blocks stays off so indentation is
        # never touched.
        self._env.trim_blocks = True
        self._env.lstrip_blocks = False

    def render(self, spec: OpSpec) -> Dict[str, str]:
        """Render all templates for ``spec``.

        :return: Mapping ``relative output path -> file text``. Keys keep the
            official project layout (``op_kernel/...``, ``op_host/...``).
        """
        context = build_render_context(spec)
        op_snake_value = context["op_snake"]
        files: Dict[str, str] = {}
        for template_name, relpath_pattern in TEMPLATE_OUTPUTS:
            template = self._env.get_template(template_name)
            relpath = relpath_pattern.format(op_snake=op_snake_value)
            files[relpath] = template.render(**context)
        return files


def render(spec: OpSpec) -> Dict[str, str]:
    """Convenience wrapper: render ``spec`` with a fresh engine."""
    return TemplateEngine().render(spec)
