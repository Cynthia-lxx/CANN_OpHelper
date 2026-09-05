"""cann_ophelper.apply -- Inspect and fill a msopgen empty-shell project.

This module is the last local stage of the expression-driven flow (``fill-op``).
Given

- an operator spec YAML whose ``expr`` intent is non-empty, and
- a project directory produced by the cloud ``msopgen`` for that operator,

``fill-op`` builds the three texts via :func:`fillgen.build_three_files`, then
this module:

1. **inspects** the empty shell and extracts its naming profile -- kernel entry
   function, host ``Input/Output`` tensor names with their declared dtypes and
   the ``AICore().AddConfig(...)`` soc -- using plain-text patterns (no C++
   semantic analysis, nothing is compiled locally);
2. **cross-checks** that profile against the spec-derived :class:`FileProfile`
   so pointing at a wrong/other shell fails loudly with bilingual errors;
3. **overwrites exactly three files** (``op_kernel/<entry>.cpp``,
   ``op_kernel/<entry>_tiling.h``, ``op_host/<entry>.cpp``); every other file
   in the project directory stays byte-for-byte untouched.

The tiling struct name inside the shell is read but deliberately *not* compared:
the tiling header is rewritten wholesale by fillgen, so only the entry name,
tensor names/dtypes and the soc family have to match (see docs/expr-rules.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .fillgen import FileProfile, _clean_soc
from .i18n import t
from .model import OpSpecError

__all__ = [
    "ShellTensor",
    "ShellImage",
    "expected_relpaths",
    "inspect_project",
    "check_shell",
    "apply",
]

#: ge::DT_* suffix -> canonical dtype label (fillgen's supported set).
_GE_DTYPE_LABELS = {"FLOAT": "float", "FLOAT16": "float16"}

#: Kernel entry signature: ``... __global__ __aicore__ void <entry>(<params>)``.
#: The official msopgen skeleton may put a single leading space before
#: ``__global__`` and fillgen adds ``extern "C"``; neither is required by the
#: pattern because it is not anchored.
_ENTRY_RE = re.compile(
    r"(?:extern\s*\"C\"\s*)?__global__\s+__aicore__\s+void\s+([A-Za-z_]\w*)\s*\(([^)]*)\)"
)
_GM_PARAM_RE = re.compile(r"GM_ADDR\s+([A-Za-z_]\w*)")
_REG_TILING_RE = re.compile(r"REGISTER_TILING_DEFAULT\s*\(\s*([A-Za-z_]\w*)\s*\)")

# Host-side patterns (op_host/<entry>.cpp).
_CLASS_RE = re.compile(r"class\s+([A-Za-z_]\w*)\s*:\s*public\s+OpDef\b")
_DECL_RE = re.compile(r"this->(Input|Output)\s*\(\s*\"([A-Za-z_]\w*)\"\s*\)")
_GE_DT_RE = re.compile(r"ge::DT_(\w+)")
_ADD_CONFIG_RE = re.compile(r"\.AddConfig\s*\(\s*\"([^\"]*)\"\s*\)")

#: GM_ADDR parameters appended by the official template; not tensors.
_NON_TENSOR_GMS = frozenset({"workspace", "tiling"})


@dataclass(frozen=True)
class ShellTensor:
    """One tensor declared by the shell's host OpDef (name + dtype set)."""

    name: str
    dtypes: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ShellImage:
    """Naming profile read from a msopgen empty shell."""

    entry: str                    # kernel entry + file stem, e.g. "asc_try"
    op_pascal: str                # OpDef class name, e.g. "AscTry"
    tiling_struct: str            # read only (informational; header is rewritten)
    inputs: tuple[ShellTensor, ...]
    outputs: tuple[ShellTensor, ...]
    soc: str                      # AICore().AddConfig(...) value


def _raise_shell(path: Path, reason: str) -> None:
    raise OpSpecError(
        t("fill_op.err.not_a_shell"),
        hint=f"{t('fill_op.err.not_a_shell.hint')} ({path}: {reason})",
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - filesystem race / permissions
        raise OpSpecError(
            t("fill_op.err.read_fail", path=path, reason=exc.strerror or str(exc)),
            hint=t("fill_op.err.read_fail.hint"),
        ) from exc


def _ge_labels(match: re.Match[str]) -> str:
    suffix = match.group(1)
    return _GE_DTYPE_LABELS.get(suffix, suffix.lower())


def expected_relpaths(entry: str) -> tuple[str, str, str]:
    """The three relative paths fill-op writes for a given kernel entry."""
    return (
        f"op_kernel/{entry}.cpp",
        f"op_kernel/{entry}_tiling.h",
        f"op_host/{entry}.cpp",
    )


def _find_kernel(
    kernel_dir: Path, expected_entry: Optional[str]
) -> tuple[Path, str, str, str]:
    """Locate + parse the kernel file.

    :returns: (path, entry, tiling_struct, gm_csv) with the GM tensor names in
        declaration order (``workspace``/``tiling`` parameters filtered out).
    :raises OpSpecError: fill_op.err.* when nothing matches.
    """
    if expected_entry:
        path = kernel_dir / f"{expected_entry}.cpp"
        if not path.is_file():
            raise OpSpecError(
                t("fill_op.err.missing_shell", missing=f"op_kernel/{expected_entry}.cpp"),
                hint=t("fill_op.err.missing_shell.hint"),
            )
        candidates: Sequence[Path] = [path]
    else:  # pragma: no cover - defensive branch; apply always knows the entry
        candidates = sorted(kernel_dir.glob("*.cpp"))

    for path in candidates:
        text = _read_text(path)
        match = _ENTRY_RE.search(text)
        if not match:
            continue
        entry = match.group(1)
        gm_names = [
            name
            for name in _GM_PARAM_RE.findall(match.group(2))
            if name not in _NON_TENSOR_GMS
        ]
        tiling_m = _REG_TILING_RE.search(text)
        tiling_struct = tiling_m.group(1) if tiling_m else ""
        return path, entry, tiling_struct, ", ".join(gm_names)

    expected = expected_entry or "<entry>"
    raise OpSpecError(
        t("fill_op.err.no_entry", entry=expected),
        hint=t("fill_op.err.no_entry.hint"),
    )


def _parse_host(host_text: str, entry: str) -> tuple[str, list[ShellTensor], list[ShellTensor], str]:
    """Parse the shell's host OpDef registration.

    :raises OpSpecError: fill_op.err.host_parse when the expected layout is not
        present (the directory is not an msopgen empty shell of <op>).
    """
    ops_at = host_text.find("namespace ops")
    segment = host_text[ops_at:] if ops_at >= 0 else host_text

    class_m = _CLASS_RE.search(segment)
    if not class_m:
        raise OpSpecError(
            t("fill_op.err.host_parse", entry=entry),
            hint=t("fill_op.err.host_parse.hint"),
        )
    pascal = class_m.group(1)

    brace = segment.find("{", class_m.end())
    end = segment.find("SetInferShape", brace) if brace != -1 else -1
    if end == -1:
        end = len(segment)
    body = segment[brace + 1 : end] if brace != -1 else ""

    inputs: list[ShellTensor] = []
    outputs: list[ShellTensor] = []
    for statement in body.split(";"):
        decl = _DECL_RE.search(statement)
        if not decl:
            continue
        kind, name = decl.group(1), decl.group(2)
        dtypes = frozenset(_ge_labels(m) for m in _GE_DT_RE.finditer(statement))
        if not dtypes:
            continue
        (inputs if kind == "Input" else outputs).append(ShellTensor(name, dtypes))

    if not inputs or not outputs:
        raise OpSpecError(
            t("fill_op.err.host_parse", entry=entry),
            hint=t("fill_op.err.host_parse.hint"),
        )

    soc_m = _ADD_CONFIG_RE.search(segment)
    if not soc_m:
        raise OpSpecError(
            t("fill_op.err.host_parse", entry=entry),
            hint=t("fill_op.err.host_parse.hint"),
        )
    return pascal, inputs, outputs, soc_m.group(1)


def inspect_project(
    project_dir: Path, *, expected_entry: Optional[str] = None
) -> ShellImage:
    """Read a msopgen empty shell and return its naming profile (no writes).

    :param project_dir: root of the shell project (contains ``op_host/`` and
        ``op_kernel/`` directories);
    :param expected_entry: kernel entry the operator spec expects (snake-case
        op_type). It locates the exact kernel/host files, so a wrong shell is
        reported as ``fill_op.err.missing_shell`` right away;
    :raises OpSpecError: fill_op.err.* when the directory is not a matching
        msopgen empty shell.
    """
    root = Path(project_dir)
    if not root.is_dir():
        _raise_shell(root, "not a directory")

    kernel_dir = root / "op_kernel"
    host_dir = root / "op_host"
    if not kernel_dir.is_dir() or not host_dir.is_dir():
        missing = [part for part in ("op_kernel", "op_host") if not (root / part).is_dir()]
        _raise_shell(root, "missing " + ", ".join(missing))

    kernel_path, entry, tiling_struct, _gm_csv = _find_kernel(kernel_dir, expected_entry)
    host_path = host_dir / f"{entry}.cpp"
    if not host_path.is_file():
        raise OpSpecError(
            t("fill_op.err.missing_shell", missing=f"op_host/{entry}.cpp"),
            hint=t("fill_op.err.missing_shell.hint"),
        )
    del kernel_path  # filename not needed once located

    pascal, inputs, outputs, soc = _parse_host(_read_text(host_path), entry)
    return ShellImage(
        entry=entry,
        op_pascal=pascal,
        tiling_struct=tiling_struct,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        soc=soc,
    )


def _fmt_names(tensors: Sequence[ShellTensor] | Sequence) -> str:
    return ", ".join(item.name for item in tensors) or "(none)"


def _verify_shell(shell: ShellImage, profile: FileProfile) -> None:
    """Raise bilingual errors when the shell does not match the spec profile."""
    if shell.entry != profile.entry:
        raise OpSpecError(
            t(
                "fill_op.err.entry_mismatch",
                shell_entry=shell.entry,
                spec_entry=profile.entry,
            ),
            hint=t("fill_op.err.entry_mismatch.hint"),
        )

    shell_in = {item.name for item in shell.inputs}
    shell_out = {item.name for item in shell.outputs}
    spec_in = {ref.name for ref in profile.inputs}
    spec_out = {ref.name for ref in profile.outputs}
    if shell_in != spec_in or shell_out != spec_out:
        raise OpSpecError(
            t(
                "fill_op.err.tensor_mismatch",
                shell_in=_fmt_names(shell.inputs),
                shell_out=_fmt_names(shell.outputs),
                spec_in=_fmt_names(profile.inputs),
                spec_out=_fmt_names(profile.outputs),
            ),
            hint=t("fill_op.err.tensor_mismatch.hint"),
        )

    by_name = {item.name: item for item in (*shell.inputs, *shell.outputs)}
    for ref in (*profile.inputs, *profile.outputs):
        if ref.dtype not in by_name[ref.name].dtypes:
            raise OpSpecError(
                t("fill_op.err.dtype_mismatch", tensor=ref.name, dtype=ref.dtype),
                hint=t("fill_op.err.dtype_mismatch.hint"),
            )

    if _clean_soc(shell.soc) != _clean_soc(profile.soc):
        raise OpSpecError(
            t(
                "fill_op.err.soc_mismatch",
                shell_soc=shell.soc or "(none)",
                spec_soc=profile.soc,
            ),
            hint=t("fill_op.err.soc_mismatch.hint"),
        )


def check_shell(project_dir: Path, profile: FileProfile) -> ShellImage:
    """Inspect + verify a shell project against a spec-derived profile.

    Also requires the three fill-op targets to exist in the shell (they are
    overwritten, never created).
    """
    shell = inspect_project(project_dir, expected_entry=profile.entry)
    _verify_shell(shell, profile)

    missing = [
        relpath
        for relpath in expected_relpaths(shell.entry)
        if not (Path(project_dir) / relpath).is_file()
    ]
    if missing:
        raise OpSpecError(
            t("fill_op.err.missing_shell", missing=", ".join(missing)),
            hint=t("fill_op.err.missing_shell.hint"),
        )
    return shell


def apply(
    project_dir: Path,
    profile: FileProfile,
    files: Mapping[str, str],
    *,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Overwrite exactly the three msopgen files of the matching shell.

    :param project_dir: root of the msopgen empty-shell project;
    :param profile: spec-derived FileProfile (identity + soc checks);
    :param files: the three texts produced by ``fillgen.build_three_files``;
    :param dry_run: only verify; do not touch the disk;
    :returns: relative paths of the (would-be) written files;
    :raises OpSpecError: fill_op.err.* when the shell is not the matching one
        or ``files`` is not exactly the three expected relpaths.
    """
    shell = check_shell(project_dir, profile)
    expected = expected_relpaths(shell.entry)
    if set(files) != set(expected):
        raise OpSpecError(
            t("fill_op.err.wrong_files", files=", ".join(sorted(files))),
            hint=t("fill_op.err.wrong_files.hint"),
        )

    written: list[str] = []
    for relpath in expected:
        target = Path(project_dir) / relpath
        if dry_run:
            written.append(relpath)
            continue
        try:
            target.write_text(files[relpath], encoding="utf-8", newline="")
        except OSError as exc:
            raise OpSpecError(
                t("yamlio.write_fail", path=target, reason=exc.strerror or str(exc)),
                hint=t("yamlio.write_fail.hint"),
            ) from exc
        written.append(relpath)
    return tuple(written)
