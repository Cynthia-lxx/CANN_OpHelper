"""cann_ophelper.cli -- Typer command-line interface for CANN_OpHelper.

Three commands expose the official msopgen workflow (docs/official-patterns.md):

- ``new-op``: collect an operator spec interactively (or from a built-in
  preset such as ``add``) and save it as YAML -- no more hand-writing a
  prototype file from scratch.
- ``gen-msopgen``: validate an operator spec YAML, preview its metadata and
  print the ready-to-run ``msopgen`` command together with the cloud execution
  steps. With ``--proto-out`` it additionally exports the official prototype
  JSON (derived mechanically from the spec) to a local file, and the msopgen
  command then references that file via ``-i``. Without ``--proto``/``--proto-out``
  it prints a reminder that the ``-i`` default is only the add demo sample, not
  the current operator's prototype. It performs no other filesystem writes.
- ``render``: render the ``op_kernel``/``op_host`` artifacts from a spec YAML.
  With ``--out`` the three files are written into that directory -- the local
  copy of an msopgen project -- overwriting what is already there; without
  ``--out`` (or with ``--dry-run``) it only previews file names and sizes.
- ``fill-op``: fill a cloud ``msopgen`` empty shell with an expression kernel.
  It cross-checks the shell profile (entry/tensors/dtypes/soc) against the
  spec, overwrites exactly the kernel/tiling/host trio, then drops the
  deterministic ``verify/`` bundle (inputs + golden.bin + aclnn runner +
  ``run_verify.sh``) that turns the operator into a one-key cloud test.
- ``quickstart``: print a copy-paste path from zero to a cloud-ready CANN
  project (wizard -> YAML -> prototype JSON -> cloud msopgen -> render); it
  performs no writes.

Language policy: static ``--help`` text stays in English, while every runtime
message is resolved from ``cann_ophelper.i18n`` at print time, so ``--lang``
(and the ``CANN_OPHELPER_LANG`` env var) apply immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, NoReturn, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as esc
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .apply import apply as apply_fill
from .fillgen import files_from_spec
from .i18n import SUPPORTED_LANGUAGES, set_language, t
from .model import OpSpec, OpSpecError, TensorSpec
from .msopgen import build_msopgen_command, show_cloud_instructions
from .proto import dump_prototype_json
from .template import render as render_artifacts
from .verifygen import verify_files, write_verify_assets
from .wizard import collect_op_spec, resolve_preset
from .yamlio import dump_op_spec, load_op_spec

__all__ = ["app", "main"]

#: Default prototype JSON path when --proto is omitted (official ch.03 sample
#: path used by the earlier demo; override it for your own operator).
DEFAULT_PROTO = "Sources/03.02/add_custom.json"

#: Default msopgen output directory when --out is omitted (cloud-side path).
DEFAULT_OUT = "out/AddCustomTemplate"

app = typer.Typer(
    name="cann-ophelper",
    help="Windows-local helper that turns an operator spec YAML into an msopgen "
    "command and Ascend C project template files (no C++ is built locally).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

#: Fresh Console per call: a cached console would pin the sys.stdout stream of
#: the first invocation, hiding later output from typer.testing.CliRunner runs.
def _c() -> Console:
    return Console()


def _print_version(value: bool) -> None:
    if value:
        _c().print(f"cann-ophelper {__version__}")
        raise typer.Exit()


def _apply_lang(value: Optional[str]) -> None:
    if not value:
        return
    lang = value.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise typer.BadParameter(f"choose from {', '.join(SUPPORTED_LANGUAGES)}")
    set_language(lang)


def _fail(message: str) -> NoReturn:
    """Print a red error line (message already localized) and exit with code 1."""
    _c().print(f"[bold red]{t('cli.error.title')}{esc(message)}[/]")
    raise typer.Exit(code=1)


def _print_note(console: Console, first: str, second: str = "") -> None:
    """Print a yellow guidance panel (labels escaped, lines kept short)."""
    lines = [f"[bold yellow]{esc(first)}[/]"]
    if second:
        lines.append(esc(second))
    console.print()
    console.print(Panel("\n".join(lines), border_style="yellow", expand=False))


@app.callback()
def _root_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_print_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
    lang: Optional[str] = typer.Option(
        None,
        "--lang",
        callback=_apply_lang,
        help="Display language ('zh' or 'en'); default comes from "
        "CANN_OPHELPER_LANG, falling back to 'zh'.",
    ),
) -> None:
    """CANN Ascend C operator template helper (local codegen only)."""


def main() -> None:
    """Real entry point for the console script and ``python -m cann_ophelper``.

    Calling the Typer instance parses ``sys.argv`` and dispatches to the
    subcommand; direct calls to this function must therefore receive no
    arguments (the package's ``__main__`` module and the console script both
    rely on that).
    """
    app()


def _load_spec(path: Path) -> OpSpec:
    """Load+validate the spec YAML; print a localized error and exit 1 on failure."""
    try:
        return load_op_spec(path)
    except OpSpecError as exc:
        _fail(str(exc))
    raise AssertionError("unreachable")


def _print_overview(console: Console, spec: OpSpec) -> None:
    """Print the operator metadata summary panel and the tensor tables."""
    meta = [
        (t("cli.overview.op_type"), spec.op_type),
        (t("cli.overview.op_snake"), spec.op_name_snake),
        (t("cli.overview.soc"), spec.soc_version),
        (t("cli.overview.desc"), spec.description or t("cli.overview.none")),
    ]
    lines = [f"[bold]{esc(label)}[/] : {esc(value)}" for label, value in meta]
    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title=t("cli.overview.title"),
            border_style="cyan",
            expand=False,
        )
    )
    for title, tensors in (
        (t("cli.tensor.inputs"), spec.inputs),
        (t("cli.tensor.outputs"), spec.outputs),
    ):
        _print_tensor_table(console, title, tensors)


def _print_tensor_table(console: Console, title: str, tensors: list[TensorSpec]) -> None:
    table = Table(title=title, title_style="bold")
    for header in ("name", "param_type", "dtype", "format", "shape"):
        table.add_column(header)
    for tensor in tensors:
        shape = ", ".join(str(dim) for dim in tensor.shape) if tensor.shape else t("cli.tensor.no_shape")
        table.add_row(
            esc(tensor.name),
            esc(tensor.param_type),
            esc(", ".join(tensor.type)),
            esc(", ".join(tensor.format)),
            esc(shape),
        )
    console.print(table)


@app.command()
def gen_msopgen(
    yaml_path: Path = typer.Argument(
        ...,
        help="Operator spec YAML file (see examples/add.yaml).",
    ),
    proto: Optional[str] = typer.Option(
        None,
        "--proto",
        help="Operator prototype JSON path (cloud-side, msopgen '-i'). When "
        "--proto-out is given, its file name is used by default; otherwise the "
        "add demo sample is referenced (see warning).",
    ),
    out: str = typer.Option(
        DEFAULT_OUT,
        "--out",
        help="msopgen project output directory (cloud-side, msopgen '-out').",
    ),
    proto_out: Optional[Path] = typer.Option(
        None,
        "--proto-out",
        help="Also write the official msopgen prototype JSON to this local path "
        "(derived mechanically from the spec) and use that file as the msopgen "
        "'-i' input; upload it to the cloud next to where msopgen runs.",
    ),
) -> None:
    """Show operator metadata and the ready-to-run msopgen command.

    Examples (typical end-to-end flows):

    # 1) interactive wizard -> spec YAML -> prototype JSON -> msopgen command
    python -m cann_ophelper new-op                       # answers become myop.yaml
    python -m cann_ophelper gen-msopgen myop.yaml --proto-out myop.json
    #    upload myop.json to the cloud, then run the printed msopgen command there

    # 2) start from the built-in 'add' preset, non-interactively
    python -m cann_ophelper new-op --from add --yes --out add.yaml
    python -m cann_ophelper gen-msopgen add.yaml --proto add.json --out out/Add

    # 3) point at a prototype you already prepared on the cloud
    python -m cann_ophelper gen-msopgen add.yaml --proto cloud/path/add.json
    """
    console = _c()
    spec = _load_spec(yaml_path)
    _print_overview(console, spec)

    # Resolve the prototype JSON shown in the msopgen command (-i). Two main
    # flows, mirroring the decision table tested in tests/test_cli.py:
    # - --proto-out given (the common "export my own prototype" flow): -i
    #   references that exported file automatically.
    # - only --proto given: it is used as-is.
    # - neither given: fall back to the built-in add demo sample and warn that
    #   it is NOT the prototype of the current operator.
    if proto_out is not None:
        if proto is None:
            proto = Path(proto_out).name
            _print_note(
                console,
                t("cli.proto_missing.hint"),
                t("cli.proto_out.suggest"),
            )
        elif Path(proto_out).name != proto:
            _print_note(
                console,
                t("cli.proto_mismatch.hint", exported=Path(proto_out).name),
                t("cli.proto_mismatch.action", proto=proto),
            )
    elif proto is None:
        proto = DEFAULT_PROTO
        _print_note(
            console,
            t("cli.proto_demo.hint", demo=DEFAULT_PROTO),
            t("cli.proto_demo.action"),
        )

    try:
        command = build_msopgen_command(spec, proto, out)
    except OpSpecError as exc:
        _fail(str(exc))
    console.print()
    console.print(
        Panel.fit(esc(command), title=t("cli.cmd.title"), border_style="green")
    )
    console.print()
    console.print(show_cloud_instructions(spec, proto, out))

    if proto_out is not None:
        try:
            dump_prototype_json(spec, proto_out)
        except OpSpecError as exc:
            _fail(str(exc))
        console.print()
        console.print(t("cli.proto_out.written"))
        console.print(esc(str(proto_out)))


def _preview_table(console: Console, files: Dict[str, str]) -> None:
    """Table of artifact relative paths + UTF-8 byte sizes (no writes)."""
    table = Table(title=t("cli.render.title_preview"), title_style="bold")
    table.add_column(t("cli.render.col_file"))
    table.add_column(t("cli.render.col_bytes"), justify="right")
    for relpath in sorted(files):
        table.add_row(esc(relpath), f"{len(files[relpath].encode('utf-8'))} B")
    console.print()
    console.print(table)


def _write_artifacts(console: Console, files: Dict[str, str], out_dir: Path) -> None:
    """Write every rendered artifact under ``out_dir`` (overwriting existing)."""
    table = Table(
        title=t("cli.render.title_wrote", count=len(files), root=esc(str(out_dir))),
        title_style="bold",
    )
    table.add_column(t("cli.render.col_file"))
    table.add_column(t("cli.render.col_bytes"), justify="right")
    table.add_column(t("cli.render.col_status"))
    for relpath in sorted(files):
        target = out_dir / relpath
        existed = target.exists()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(files[relpath], encoding="utf-8", newline="")
        except OSError as exc:
            reason = exc.strerror or str(exc)
            _fail(f"{target} ({esc(reason)})")
        status = t("cli.render.overwritten" if existed else "cli.render.written")
        table.add_row(
            esc(relpath),
            f"{len(files[relpath].encode('utf-8'))} B",
            esc(status),
        )
    console.print()
    console.print(table)


@app.command()
def render(
    yaml_path: Path = typer.Argument(
        ...,
        help="Operator spec YAML file (see examples/add.yaml).",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory to write the artifacts into (the local copy of the "
        "msopgen project root, which already contains op_host/op_kernel).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview only; never write files.",
    ),
) -> None:
    """Render op_kernel/op_host artifacts from a spec YAML."""
    console = _c()
    spec = _load_spec(yaml_path)
    try:
        files = render_artifacts(spec)
    except OpSpecError as exc:
        _fail(str(exc))

    if out_dir is None or dry_run:
        _preview_table(console, files)
        if dry_run and out_dir is not None:
            console.print(t("cli.render.dry_note"))
            return
        suggest = Path("out") / spec.op_type
        console.print(t("cli.render.no_out"))
        console.print(t("cli.render.suggest", suggest=esc(suggest.as_posix())))
        return

    _write_artifacts(console, files, out_dir)


@app.command()
def fill_op(
    yaml_path: Path = typer.Argument(
        ...,
        help="Operator spec YAML that carries an expression ('expr' field).",
    ),
    project_dir: Path = typer.Argument(
        ...,
        help="Root directory of the msopgen empty-shell project to fill in "
        "(the one copied back from the cloud, containing op_host/op_kernel).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Verify the shell against the spec and preview the writes; never "
        "modify files.",
    ),
) -> None:
    """Fill an msopgen empty-shell project with the expression kernel.

    Reads the expression-driven spec (op metadata + 'expr' intent), parses and
    lowers the expression, regenerates the kernel/tiling/host trio from it and
    overwrites exactly those three files inside the shell project -- every
    other file of the msopgen project stays untouched. The shell is first
    cross-checked (entry function, tensor names/dtypes and AddConfig soc must
    match the spec), otherwise the command fails with a bilingual error.

    After the write it also drops a deterministic ``verify/`` bundle into the
    project: one float32 input file per declared tensor, a ``golden.bin``
    computed by interpreting the same lowered program, an aclnn single-op host
    runner and a one-key ``bash verify/run_verify.sh`` that builds, deploys,
    runs and numerically compares the result on the cloud.

    Examples (typical flow):

    # 1) fill the cloud msopgen shell + generate verify assets
    python -m cann_ophelper fill-op asctry.yaml cloud_shell/asc_try/

    # 2) preview only: validate + list what would be written
    python -m cann_ophelper fill-op asctry.yaml cloud_shell/asc_try/ --dry-run

    # 3) after uploading the whole project back to the cloud
    bash cloud_shell/asc_try/verify/run_verify.sh   # -> TEST PASSED
    """
    console = _c()
    spec = _load_spec(yaml_path)
    console.print(
        Panel(
            esc(t("fill_op.info.expr", expr=spec.expr) if spec.expr else t("fill_op.info.empty")),
            title=t("fill_op.info.title"),
            border_style="cyan",
            expand=False,
        )
    )

    try:
        _profile, _program, files = files_from_spec(spec)
    except OpSpecError as exc:
        _fail(str(exc))

    try:
        written = apply_fill(project_dir, profile=_profile, files=files, dry_run=dry_run)
    except OpSpecError as exc:
        _fail(str(exc))

    # Stage 1: the kernel/tiling/host trio (overwrite, nothing else touched).
    table = Table(
        title=t("fill_op.ok.dry" if dry_run else "fill_op.ok.written"),
        title_style="bold",
    )
    table.add_column(t("cli.render.col_file"))
    table.add_column(t("cli.render.col_bytes"), justify="right")
    table.add_column(t("cli.render.col_status"))
    for relpath in sorted(written):
        table.add_row(
            esc(relpath),
            f"{len(files[relpath].encode('utf-8'))} B",
            esc(t("cli.render.overwritten")),
        )
    console.print()
    console.print(table)

    # Stage 2: cloud verification assets (deterministic inputs + golden +
    # aclnn runner + one-key run script), only written for a real fill.
    verify = verify_files(_program, _profile)
    if dry_run:
        created = sorted(verify)
    else:
        try:
            created = write_verify_assets(project_dir, _program, _profile)
        except OpSpecError as exc:
            _fail(str(exc))

    assets = Table(
        title=t(
            "verifygen.ok.dry" if dry_run else "verifygen.ok.created",
            count=len(created),
        ),
        title_style="bold",
    )
    assets.add_column(t("cli.render.col_file"))
    assets.add_column(t("cli.render.col_bytes"), justify="right")
    for relpath in created:
        assets.add_row(esc(relpath), f"{len(verify[relpath])} B")
    console.print()
    console.print(assets)

    if not dry_run:
        console.print()
        console.print(
            Panel(
                f"[bold]{esc(t('verifygen.next'))}[/]\n\n{t('verifygen.next.hint')}",
                title=t("verifygen.title"),
                border_style="green",
                expand=False,
            )
        )


@app.command()
def new_op(
    from_preset: Optional[str] = typer.Option(
        None,
        "--from",
        help="Start from a built-in preset (currently 'add') to prefill every "
        "answer; empty replies keep the prefilled value.",
    ),
    out_path: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Target path for the generated spec YAML (default: "
        "<snake_case_name>.yaml in the current directory).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Collect an operator spec interactively (or from a preset) and save it as YAML."""
    console = _c()
    seed = None
    if from_preset:
        try:
            seed = resolve_preset(from_preset)
        except OpSpecError as exc:
            _fail(str(exc))
        console.print(t("cli.new_op.preset_applied", preset=esc(from_preset)))

    try:
        spec = collect_op_spec(seed=seed, console=console)
    except OpSpecError as exc:
        _fail(str(exc))
    except (EOFError, KeyboardInterrupt):
        console.print(t("cli.new_op.cancelled"))
        raise typer.Exit(code=1)

    _print_overview(console, spec)

    target = out_path or Path(f"{spec.op_name_snake}.yaml")
    if not yes:
        try:
            confirmed = typer.confirm(
                t("cli.new_op.confirm", path=esc(str(target))), default=True
            )
        except typer.Abort as exc:
            raise typer.Exit(code=130) from exc
        if not confirmed:
            console.print(t("cli.new_op.cancelled"))
            raise typer.Exit()

    try:
        dump_op_spec(spec, target)
    except OpSpecError as exc:
        _fail(str(exc))

    console.print()
    console.print(f"[bold green]{esc(t('cli.new_op.written', path=str(target)))}[/]")
    console.print(t("cli.new_op.suggest", path=esc(str(target))))


@app.command()
def quickstart() -> None:
    """Show a copy-paste path from zero to a cloud-ready CANN project."""
    console = _c()
    py = "python -m cann_ophelper"
    console.print()
    console.print(
        Panel.fit(
            Markdown(t("cli.quickstart.body").replace("{py}", py)),
            title=t("cli.quickstart.title"),
            border_style="cyan",
        )
    )
