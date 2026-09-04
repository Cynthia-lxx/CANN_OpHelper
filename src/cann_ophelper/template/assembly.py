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

Round 2 transition state: the kernel produced file is already assembled from
line-exact snippets (shared ``license`` + ``kernel_*``) under
``templates/snippets/``; tiling and host still point at their whole-file
templates (``file_tiling`` / ``file_host``) until they are split in the same
round.
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
#: ``license`` is shared by every produced file; kernel snippets are the
#: line-exact split of the old whole-file ``op_kernel/kernel.cpp.j2`` (rows
#: 11-104; the first ten template rows were Jinja prelude). tiling/host still
#: use whole-file placeholders until the round-2 host/tiling migration.
SNIPPETS: Dict[str, str] = {
    # shared license block + two trailing blank lines (rows 11-21 of the old
    # kernel template; identical text and blank count in the tiling/host heads)
    "license": "snippets/license.j2",
    # --- kernel (old kernel.cpp.j2 rows) ---
    "kernel_includes": "snippets/kernel_includes.j2",
    "kernel_class_head": "snippets/kernel_class_head.j2",
    "kernel_init": "snippets/kernel_init.j2",
    "kernel_process": "snippets/kernel_process.j2",
    "kernel_copyin": "snippets/kernel_copyin.j2",
    "kernel_compute": "snippets/kernel_compute.j2",
    "kernel_copyout": "snippets/kernel_copyout.j2",
    "kernel_members": "snippets/kernel_members.j2",
    "kernel_entry": "snippets/kernel_entry.j2",
    # --- whole-file placeholders (removed when tiling/host are split) ---
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
