"""cann_ophelper.template.engine -- Assembly-driven Jinja2 rendering entry point.

:class:`TemplateEngine` renders the op files declared by
:mod:`.assembly`: for each produced-file assembly it renders every referenced
snippet through Jinja2's ``PackageLoader`` and concatenates the texts in
assembly order, so the output files for an :class:`OpSpec` are:

- ``op_kernel/<op_snake>.cpp``      (kernel)
- ``op_kernel/<op_snake>_tiling.h`` (shared tiling header)
- ``op_host/<op_snake>.cpp``        (host)

Whitespace handling keeps snippets verbatim: a trailing newline is preserved
(``keep_trailing_newline=True``), so concatenated output stays diff-able
against the official sample files. The engine holds no per-file template
choices -- the assembly table in :mod:`.assembly` is the single orchestration
fact; templates and context contain no business logic.
"""

from __future__ import annotations

from typing import Dict

from jinja2 import Environment, PackageLoader

from ..model import OpSpec
from .assembly import (
    FILE_ASSEMBLIES,
    SNIPPETS,
    TEMPLATE_OUTPUTS,
    assembly_ids_for,
    validate_assemblies,
)
from .context import build_render_context

__all__ = [
    "FILE_ASSEMBLIES",
    "SNIPPETS",
    "TEMPLATE_OUTPUTS",
    "TemplateEngine",
    "render",
    "render_file",
    "render_snippet",
]

#: Jinja2 package used by PackageLoader (cann_ophelper.template.templates).
_LOADER_PACKAGE = "cann_ophelper"
_LOADER_PATH = "template/templates"


class TemplateEngine:
    """Renders the assembly-declared op files for an operator spec."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=PackageLoader(_LOADER_PACKAGE, _LOADER_PATH),
            keep_trailing_newline=True,
            autoescape=False,
        )
        # trim_blocks removes only the newline that directly follows a {% %}
        # block tag (e.g. a helper {% set %} line), so snippets stay aligned
        # with the official files; lstrip_blocks stays off so indentation is
        # never touched.
        self._env.trim_blocks = True
        self._env.lstrip_blocks = False
        validate_assemblies()

    def render_snippet(self, snippet_id: str, spec: OpSpec) -> str:
        """Render one snippet for ``spec``.

        :raises KeyError: if ``snippet_id`` is not registered in :data:`SNIPPETS`.
        """
        try:
            template_name = SNIPPETS[snippet_id]
        except KeyError:
            raise KeyError(f"unknown snippet id: {snippet_id!r}") from None
        context = build_render_context(spec)
        return self._env.get_template(template_name).render(**context)

    def render_file(self, spec: OpSpec, relpath_pattern: str) -> str:
        """Render one produced file for ``spec`` by its assembly pattern.

        :raises KeyError: if ``relpath_pattern`` is not a produced-file pattern.
        """
        snippet_ids = assembly_ids_for(relpath_pattern)
        context = build_render_context(spec)
        return "".join(
            self._env.get_template(SNIPPETS[snippet_id]).render(**context)
            for snippet_id in snippet_ids
        )

    def render(self, spec: OpSpec) -> Dict[str, str]:
        """Render all assemblies for ``spec``.

        :return: Mapping ``relative output path -> file text``. Keys keep the
            official project layout (``op_kernel/...``, ``op_host/...``).
        """
        context = build_render_context(spec)
        op_snake_value = context["op_snake"]
        files: Dict[str, str] = {}
        for _, relpath_pattern, snippet_ids in FILE_ASSEMBLIES:
            relpath = relpath_pattern.format(op_snake=op_snake_value)
            files[relpath] = "".join(
                self._env.get_template(SNIPPETS[snippet_id]).render(**context)
                for snippet_id in snippet_ids
            )
        return files


def render(spec: OpSpec) -> Dict[str, str]:
    """Convenience wrapper: render ``spec`` with a fresh engine."""
    return TemplateEngine().render(spec)


def render_snippet(snippet_id: str, spec: OpSpec) -> str:
    """Convenience wrapper: render one snippet with a fresh engine."""
    return TemplateEngine().render_snippet(snippet_id, spec)


def render_file(spec: OpSpec, relpath_pattern: str) -> str:
    """Convenience wrapper: render one produced file with a fresh engine."""
    return TemplateEngine().render_file(spec, relpath_pattern)
