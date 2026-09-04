"""Tests for the snippet/assembly layer (round 2 apply-pipeline foundation).

The produced files are assembled from line-exact snippets (see
docs/official-patterns.md §8). These tests lock the *assembly contract*:

- the assembly table covers exactly the three produced files, references only
  registered snippets, never repeats or drops a snippet;
- for every produced file, concatenating its rendered snippets reproduces
  ``render(spec)[relpath]`` byte for byte (== golden == official sample);
- every snippet is independently renderable (non-empty, trailing newline);
- unknown snippet ids / relpath patterns raise :class:`KeyError`.
"""

from __future__ import annotations

import pytest

from cann_ophelper.model import OpSpec, TensorSpec
from cann_ophelper.template.assembly import (
    FILE_ASSEMBLIES,
    SNIPPETS,
    TEMPLATE_OUTPUTS,
    assembly_ids_for,
    validate_assemblies,
)
from cann_ophelper.template.engine import TemplateEngine, render, render_snippet

#: Produced-file relpath patterns declared by the assembly table.
PRODUCED_PATTERNS = [relpath for _, relpath, _ in FILE_ASSEMBLIES]


def _spec() -> OpSpec:
    """The reference spec: official AddCustomTemplate, float16/float, ND."""
    tensor = dict(type=["float16", "float"], format=["ND", "ND"])
    return OpSpec(
        op_type="AddCustomTemplate",
        soc_version="ascend910b1",
        inputs=[TensorSpec(name="x", **tensor), TensorSpec(name="y", **tensor)],
        outputs=[TensorSpec(name="z", **tensor)],
    )


class TestRegistryCompleteness:
    def test_assembly_table_covers_exactly_three_files(self):
        assert len(PRODUCED_PATTERNS) == len(set(PRODUCED_PATTERNS)) == 3
        for pattern in PRODUCED_PATTERNS:
            assert pattern.startswith("op_kernel/") or pattern.startswith("op_host/")
            assert "{op_snake}" in pattern

    def test_every_referenced_snippet_is_registered(self):
        validate_assemblies()  # raises ValueError on any integrity problem

    def test_no_snippet_is_shared_as_dead_or_repeated_in_one_assembly(self):
        for _, pattern, snippet_ids in FILE_ASSEMBLIES:
            assert snippet_ids, f"{pattern} has an empty snippet sequence"
            assert len(snippet_ids) == len(set(snippet_ids)), (
                f"{pattern} repeats a snippet id"
            )

    def test_license_is_shared_by_every_assembly(self):
        for _, _, snippet_ids in FILE_ASSEMBLIES:
            assert snippet_ids[0] == "license"

    def test_all_registered_snippets_are_referenced(self):
        referenced = {
            sid for _, _, snippet_ids in FILE_ASSEMBLIES for sid in snippet_ids
        }
        assert set(SNIPPETS) == referenced

    def test_assembly_ids_for_returns_stable_order(self):
        for _, pattern, snippet_ids in FILE_ASSEMBLIES:
            assert assembly_ids_for(pattern) == snippet_ids

    def test_template_outputs_derived_from_assemblies(self):
        assert TEMPLATE_OUTPUTS == tuple(
            (logical, relpath) for logical, relpath, _ in FILE_ASSEMBLIES
        )


class TestConcatenationInvariant:
    @pytest.mark.parametrize("pattern", PRODUCED_PATTERNS)
    def test_concat_of_snippets_equals_render(self, pattern):
        """concat(render_snippet(s) for s in assembly) == render(spec)[relpath]."""
        spec = _spec()
        relpath = pattern.format(op_snake=spec.op_name_snake)
        assembled = "".join(render_snippet(sid, spec) for sid in assembly_ids_for(pattern))
        assert assembled == render(spec)[relpath]

    @pytest.mark.parametrize("pattern", PRODUCED_PATTERNS)
    def test_render_file_matches_full_render(self, pattern):
        spec = _spec()
        assert TemplateEngine().render_file(spec, pattern) == render(spec)[
            pattern.format(op_snake=spec.op_name_snake)
        ]


class TestSnippetIndependence:
    def test_every_snippet_renders_nonempty_and_ends_with_newline(self):
        spec = _spec()
        engine = TemplateEngine()
        for snippet_id in SNIPPETS:
            text = engine.render_snippet(snippet_id, spec)
            assert text.strip(), f"snippet {snippet_id!r} rendered empty"
            assert text.endswith("\n"), f"snippet {snippet_id!r} missing trailing newline"

    def test_host_snippets_do_not_leak_template_prelude(self):
        """The {% set %} helper rows must not leak into rendered snippets."""
        text = render_snippet("host_opdef", _spec())
        assert text.startswith("namespace ops {")
        assert "{% set" not in text


class TestErrorPaths:
    def test_unknown_snippet_id_raises_keyerror(self):
        with pytest.raises(KeyError):
            TemplateEngine().render_snippet("no_such_snippet", _spec())

    def test_unknown_relpath_pattern_raises_keyerror(self):
        with pytest.raises(KeyError):
            assembly_ids_for("op_kernel/no_such_pattern.cpp")
        with pytest.raises(KeyError):
            TemplateEngine().render_file(_spec(), "op_kernel/no_such_pattern.cpp")
