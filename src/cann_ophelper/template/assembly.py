"""cann_ophelper.template.assembly -- Snippet registry and file-assembly table.

This module is the single source of truth that maps each produced op file to
the ordered list of Jinja2 *snippets* it is assembled from:

- :data:`SNIPPETS`     -- snippet id -> template relpath (relative to the
  package ``template/templates`` directory).
- :data:`FILE_ASSEMBLIES` -- one entry per produced file: ``(logical name,
  output relpath pattern, ordered snippet ids)``.
- :data:`TEMPLATE_OUTPUTS` -- compatibility export derived from
  ``FILE_ASSEMBLIES`` (logical name, relpath pattern); kept because earlier
  rounds/tests import it from ``cann_ophelper.template.engine``.
- :func:`assembly_ids_for` / :func:`validate_assemblies` -- lookup / integrity
  helpers.

Assembly entries contain no rendering logic; :mod:`.engine` only renders each
referenced snippet and concatenates the texts in table order. Because every
snippet template is a line-exact slice of the original whole-file template,
``concat(render(snippet) for id in ids)`` reproduces the whole-file render
byte for byte (locked by the golden / official-alignment tests).

Round 2 transition state: the table still points at the three whole-file
templates (``file_kernel`` / ``file_tiling`` / ``file_host``) so rendering is
unchanged while the engine moves to assembly-driven dispatch. Round 2 then
replaces each whole-file entry with fine-grained snippets under
``templates/snippets/`` (shared ``license`` + section snippets).
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "FILE_ASSEMBLIES",
    "SNIPPETS",
    "TEMPLATE_OUTPUTS",
    "assembly_ids_for",
    "validate_assemblies",
]

#: Snippet id -> Jinja2 template path, relative to ``template/templates``.
#: Whole-file placeholders (Round 2 step 1); replaced by fine-grained snippets
#: under ``templates/snippets/`` in the same round.
SNIPPETS: Dict[str, str] = {
    "file_kernel": "op_kernel/kernel.cpp.j2",
    "file_tiling": "op_kernel/tiling.h.j2",
    "file_host": "op_host/host.cpp.j2",
}

#: Produced file assemblies: (logical name, output relpath pattern, ordered
#: snippet ids). Relpath patterns may contain ``{op_snake}``; the renderer
#: substitutes the context value.
FILE_ASSEMBLIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "op_kernel_cpp",
        "op_kernel/{op_snake}.cpp",
        ("file_kernel",),
    ),
    (
        "op_kernel_tiling_h",
        "op_kernel/{op_snake}_tiling.h",
        ("file_tiling",),
    ),
    (
        "op_host_cpp",
        "op_host/{op_snake}.cpp",
        ("file_host",),
    ),
)

#: Compatibility export kept for earlier rounds: (logical name, relpath
#: pattern). Unique by construction (logical names differ).
TEMPLATE_OUTPUTS: Tuple[Tuple[str, str], ...] = tuple(
    (logical, relpath) for logical, relpath, _ in FILE_ASSEMBLIES
)


def assembly_ids_for(relpath_pattern: str) -> Tuple[str, ...]:
    """Return the ordered snippet ids that assemble ``relpath_pattern``.

    :raises KeyError: if ``relpath_pattern`` is not a produced-file pattern.
    """
    for _, pattern, snippet_ids in FILE_ASSEMBLIES:
        if pattern == relpath_pattern:
            return snippet_ids
    raise KeyError(f"unknown produced-file pattern: {relpath_pattern!r}")


def validate_assemblies() -> None:
    """Check registry/assembly integrity; raise :class:`ValueError` on problems.

    Every referenced snippet must be registered, no assembly may be empty or
    duplicate a snippet, produced-file patterns must be unique, and every
    registered snippet must be referenced (no dead entries).
    """
    problems: list[str] = []

    patterns = [pattern for _, pattern, _ in FILE_ASSEMBLIES]
    if len(patterns) != len(set(patterns)):
        problems.append("FILE_ASSEMBLIES contains duplicate relpath patterns")

    referenced: set[str] = set()
    for logical, pattern, snippet_ids in FILE_ASSEMBLIES:
        if not logical or not pattern or not snippet_ids:
            problems.append(f"assembly {pattern!r} must have a non-empty logical "
                            "name, pattern and snippet sequence")
        if len(snippet_ids) != len(set(snippet_ids)):
            problems.append(f"assembly {pattern!r} repeats a snippet id")
        for snippet_id in snippet_ids:
            referenced.add(snippet_id)
            if snippet_id not in SNIPPETS:
                problems.append(f"assembly {pattern!r} references unknown "
                                f"snippet {snippet_id!r}")

    unused = sorted(set(SNIPPETS) - referenced)
    if unused:
        problems.append(f"SNIPPETS entries never referenced: {unused}")

    if problems:
        raise ValueError("invalid template assembly: " + "; ".join(problems))
