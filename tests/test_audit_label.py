"""How an element is named in logs and results: controls by their label, data by nothing at all.

A results row is somebody's record ("12345 TESTERSON, ADA ACTIVE"). It is fine on the screen and
in the model's view of the page, but the audit trail must not copy it."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.test_gateway import element as base_element

from cua.surface.base import ElementInfo, audit_label


def element(
    ref: str,
    tag: str,
    text: str = "",
    role: str | None = None,
    label_hint: str | None = None,
    **attrs: str,
) -> ElementInfo:
    return replace(base_element(ref, tag, text, role, **attrs), label_hint=label_hint)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tag": "button", "text": "Search"}, "e1 <button> 'Search'"),
        ({"tag": "a", "text": "Member Inquiry", "role": "link"}, "e1 <a> 'Member Inquiry'"),
        ({"tag": "input", "label_hint": "Member No:"}, "e1 <input> 'Member No:'"),
        ({"tag": "input", "text": ""}, "e1 <input> ''"),
        ({"tag": "div", "role": "button", "text": "Save draft"}, "e1 <div> 'Save draft'"),
    ],
)
def test_a_control_is_named_by_its_own_label(kwargs: dict[str, object], expected: str) -> None:
    assert audit_label(element("e1", **kwargs)) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("tag", ["tr", "td", "th", "div", "span", "li"])
def test_data_is_never_copied_into_the_audit_trail(tag: str) -> None:
    info = element("e6", tag, "12345 TESTERSON, ADA ACTIVE")
    label = audit_label(info)
    assert label == f"e6 <{tag}> (contents not logged)"
    assert "TESTERSON" not in label
    assert "12345" not in label


def test_a_clickable_cell_is_identified_by_its_handler_not_its_text() -> None:
    info = element("e4", "td", "Open Sub-Account 12345", onclick="openSub('NS')")
    assert audit_label(info) == "e4 <td> onclick=openSub('NS')"


def test_a_long_handler_is_cut() -> None:
    label = audit_label(element("e4", "td", "x", onclick="a" * 200))
    assert len(label) < 70
