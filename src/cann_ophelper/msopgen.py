"""cann_ophelper.msopgen -- Compose msopgen command lines and cloud instructions.

Command format follows the official docs (see docs/official-patterns.md SS1.2):

    msopgen gen -i <prototype JSON> -c ai_core-<soc> -lan cpp -out <output dir>

This module only does **pure text assembly**, with no filesystem side effects.
The prototype JSON is supplied by the user (this tool neither generates nor
parses its content -- rule 6.4).

All user-facing messages are resolved through ``cann_ophelper.i18n``.
"""

from __future__ import annotations

import os
import shlex
from typing import Union

from .i18n import t
from .model import OpSpec, OpSpecError

__all__ = [
    "MSOPGEN_SOC_PREFIX",
    "format_soc_for_msopgen",
    "shell_quote",
    "build_msopgen_command",
    "show_cloud_instructions",
]

#: Fixed prefix for the msopgen ``-c`` argument (official form: ai_core-ascend910b1).
MSOPGEN_SOC_PREFIX = "ai_core-"


def format_soc_for_msopgen(soc_version: str) -> str:
    """Normalize a bare soc (e.g. ascend910b1) to an msopgen ``-c`` value
    (ai_core-ascend910b1). Values already prefixed with ``ai_core-`` are
    returned unchanged to avoid a duplicated prefix."""
    soc = soc_version.strip()
    if soc.lower().startswith(MSOPGEN_SOC_PREFIX):
        return soc
    return f"{MSOPGEN_SOC_PREFIX}{soc}"


def shell_quote(value: str) -> str:
    """Shell-safe quoting of command line paths (POSIX rules, matching the Linux
    cloud/CANN environment). Only values containing special characters are
    single-quoted so the command stays copy-paste friendly."""
    if not value:
        return "''"
    if any(ch.isspace() or ch in "\"'\\$`;&|<>()*?[]{}~#" for ch in value):
        return shlex.quote(value)
    return value


def build_msopgen_command(
    spec: OpSpec,
    proto_json: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
    language: str = "cpp",
) -> str:
    """Compose one complete msopgen command from operator metadata.

    :param spec: A validated OpSpec (op_type/soc_version drive soc assembly only;
        the command itself does not depend on op_type -- the operator name comes
        from the user's prototype JSON);
    :param proto_json: Path to the operator prototype JSON (msopgen ``-i``);
        local or cloud-relative paths are both accepted;
    :param out_dir: Project output directory (``-out``);
    :param language: Development language; only ``cpp`` (the official value)
        is supported.
    :returns: A command string ready to be copied into the cloud environment.
    """
    spec.validate()
    lang = language.strip().lower()
    if lang != "cpp":
        raise OpSpecError(
            t("check.language_invalid", value=language),
            hint=t("check.language_msopgen.hint"),
        )
    json_arg = shell_quote(str(proto_json))
    soc_arg = shell_quote(format_soc_for_msopgen(spec.soc_version))
    out_arg = shell_quote(str(out_dir))
    return f"msopgen gen -i {json_arg} -c {soc_arg} -lan {lang} -out {out_arg}"


def show_cloud_instructions(
    spec: OpSpec,
    proto_json: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
) -> str:
    """Build companion cloud-execution instructions (plain text; reusable by CLI/README).

    Fact base (official-patterns SS5): msopgen must run in a cloud environment
    with the CANN Toolkit installed; the generated project contains op_host /
    op_kernel etc. and can be copied back locally for later reading/filling.
    """
    cmd = build_msopgen_command(spec, proto_json, out_dir)
    lines = [
        t("ci.title"),
        "",
        t("ci.step1"),
        t("ci.step2"),
        f"     {cmd}",
        t("ci.step3", out_dir=out_dir),
        t("ci.step4"),
        "",
        t("ci.tip", soc=format_soc_for_msopgen(spec.soc_version)),
    ]
    return "\n".join(lines)
