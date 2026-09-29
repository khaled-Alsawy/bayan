"""Identify the fixed contextual-topic fallback without copying its text.

The fallback remains authored in :mod:`synthesis`. Consumers parse that source so a
template edit either updates every consumer or fails loudly if its shape changes.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path


@cache
def deterministic_contextual_topic_template_markers() -> frozenset[str]:
    """Return the two fixed fallback openings from the production draft function."""

    source_path = Path(__file__).with_name("synthesis.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "deterministic_contextual_topic_draft"
        ),
        None,
    )
    if function is None:
        raise RuntimeError("could not locate deterministic_contextual_topic_draft")

    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "text" for target in node.targets)
            and isinstance(node.value, ast.IfExp)
        ):
            continue
        try:
            templates = frozenset(
                {ast.literal_eval(node.value.body), ast.literal_eval(node.value.orelse)}
            )
        except ValueError:
            continue
        if len(templates) == 2 and all(isinstance(template, str) for template in templates):
            return frozenset(_opening_marker(template) for template in templates)

    raise RuntimeError("could not extract the fixed contextual-topic fallback templates")


def _opening_marker(template: str) -> str:
    """Derive the stable opening clause used to identify a rendered fallback."""

    separator_positions = [
        position
        for separator in (",", "،")
        if (position := template.find(separator)) >= 0
    ]
    return template[: min(separator_positions)] if separator_positions else template


def is_deterministic_contextual_topic_template_prose(prose: object) -> bool:
    """Whether a rendered prose string carries the fixed contextual-topic fallback."""

    return isinstance(prose, str) and any(
        marker in prose for marker in deterministic_contextual_topic_template_markers()
    )
