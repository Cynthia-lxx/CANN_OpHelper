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

Round 2 final state: every produced file is assembled from line-exact snippets
under ``templates/snippets/``. ``license`` is shared by all three files; the
old whole-file templates (``op_kernel/kernel.cpp.j2``,
``op_kernel/tiling.h.j2``, ``op_host/host.cpp.j2``) have been removed.
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
#: ``license`` is shared by every produced file. Snippets are the line-exact
#: split of the old whole-file templates (rows after each file's Jinja prelude
#: comment/set lines); blank lines between sections are owned by the preceding
#: snippet as trailing blanks.
SNIPPETS: Dict[str, str] = {
    # shared license block + two trailing blank lines (identical text and blank
    # count in every official head)
    "license": "snippets/license.j2",
    # --- kernel (old op_kernel/kernel.cpp.j2) ---
    "kernel_includes": "snippets/kernel_includes.j2",
    "kernel_class_head": "snippets/kernel_class_head.j2",
    "kernel_init": "snippets/kernel_init.j2",
    "kernel_process": "snippets/kernel_process.j2",
    "kernel_copyin": "snippets/kernel_copyin.j2",
    "kernel_compute": "snippets/kernel_compute.j2",
    "kernel_copyout": "snippets/kernel_copyout.j2",
    "kernel_members": "snippets/kernel_members.j2",
    "kernel_entry": "snippets/kernel_entry.j2",
    # --- tiling header (old op_kernel/tiling.h.j2) ---
    "tiling_body": "snippets/tiling_body.j2",
    # --- host (old op_host/host.cpp.j2) ---
    "host_includes": "snippets/host_includes.j2",
    "host_tiling": "snippets/host_tiling.j2",
    "host_infer": "snippets/host_infer.j2",
    "host_opdef": "snippets/host_opdef.j2",
}

#: Produced file assemblies: (logical name, output relpath pattern, ordered
#: snippet ids). Relpath patterns may contain ``{op_snake}``; the renderer
#: substitutes the context value.
FILE_ASSEMBLIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "op_kernel_cpp",
        "op_kernel/{op_snake}.cpp",
        (
            "license",
            "kernel_includes",
            "kernel_class_head",
            "kernel_init",
            "kernel_process",
            "kernel_copyin",
            "kernel_compute",
            "kernel_copyout",
            "kernel_members",
            "kernel_entry",
        ),
    ),
    (
        "op_kernel_tiling_h",
        "op_kernel/{op_snake}_tiling.h",
        (
            "license",
            "tiling_body",
        ),
    ),
    (
        "op_host_cpp",
        "op_host/{op_snake}.cpp",
        (
            "license",
            "host_includes",
            "host_tiling",
            "host_infer",
            "host_opdef",
        ),
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
