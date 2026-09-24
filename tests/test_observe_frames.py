"""What the surface lets the agent *see* on a hostile legacy UI.

Real Chromium against the MemberServ mock. The interesting cases are exactly the ones a naive
observer misses: a true <frameset>, a nested iframe, controls that are only a clickable row or
cell, inputs whose label is a neighbouring table cell, and a decoy that is not visible.
"""

from __future__ import annotations

import re
import time

import pytest
from tests.browser_support import by_text, find, open_member, sign_in
from tests.mockbank_support import MockHandle

from cua.surface import Action, PlaywrightSurface

pytestmark = pytest.mark.browser


def test_the_login_page_is_one_frame_with_unlabeled_inputs_and_no_hidden_fields(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    assert obs.title == "MemberServ 3.1"
    assert len(obs.frames) == 1
    user, password = find(obs, name="u"), find(obs, name="p")
    assert user.role == "textbox"
    assert user.label_hint == "User ID"  # the label is a neighbouring table cell
    assert password.label_hint == "Password"
    assert find(obs, type="image").role == "button"
    assert not any(el.attrs.get("name") == "tok" for el in obs.elements())  # hidden: not offered


def test_the_screenshot_is_a_png_of_the_viewport(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    assert obs.screenshot.startswith(b"\x89PNG")
    assert obs.viewport == (1280, 800)


def test_typed_secrets_never_appear_in_an_observation(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="p").ref, secret="MOCK_PASS"))
    obs = surface.observe()
    password = find(obs, name="p")
    assert password.attrs["value"] == "<masked>"
    dumped = repr(obs.frames)
    assert "demo-only" not in dumped


def test_textbox_lines_in_the_outline_carry_no_value(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    surface.act(Action.fill(find(obs, name="F1").ref, text="12345"))
    main = surface.observe().frames[3]
    lines = [ln.strip() for ln in main.aria.splitlines() if ln.strip().startswith("- textbox")]
    assert lines
    assert not any(re.match(r'- textbox(?: "[^"]*")?:', ln) for ln in lines)
    assert find(surface.observe(), name="F1").attrs["value"] == "12345"  # it is on the element


def test_a_registered_secret_is_masked_wherever_it_shows_up(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    box = find(obs, name="F2")  # an ordinary field: only the registered-secret layer can catch it
    surface.act(Action.fill(box.ref, secret="MOCK_PASS"))
    obs = surface.observe()
    assert find(obs, name="F2").attrs["value"] == "<masked>"
    assert "demo-only" not in repr(obs.frames)  # neither the element, the outline nor the text


def test_a_frameset_is_walked_frame_by_frame_in_document_order(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    assert [f.name for f in obs.frames] == [None, "hdr", "nav", "main"]
    top, hdr, nav, main = obs.frames
    assert top.path == ()
    assert [hop.name for hop in main.path] == ["main"]
    assert main.url.endswith("/msv/search.cgi")
    assert "TELLER01" in hdr.text
    assert "Member Inquiry" in nav.text
    assert obs.url.endswith("/msv/frameset.cgi")  # the top URL never says where you are


def test_the_frameset_document_itself_is_observed_without_hanging(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    sign_in(surface, mock)
    started = time.monotonic()
    obs = surface.observe()
    assert time.monotonic() - started < 10  # aria on a frameset <body> would time out
    assert "iframe" in obs.frames[0].aria  # falls back to the document outline
    assert obs.frames[0].elements == []


def test_refs_are_unique_across_frames_and_resolve_back_to_elements(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    refs = [el.ref for el in obs.elements()]
    assert len(refs) == len(set(refs)) > 5
    for ref in refs:
        assert obs.element(ref).ref == ref
    assert {el.frame for el in obs.elements()} == {1, 2, 3}  # hdr, nav, main have controls


def test_javascript_links_and_off_allowlist_bait_are_visible_as_links(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    nav = obs.frames[2].elements
    inquiry = next(el for el in nav if el.text == "Member Inquiry")
    assert inquiry.role == "link"
    assert inquiry.attrs["href"] == "javascript:go('MI')"
    assert any(el.text == "Admin" for el in nav)


def test_an_unlabeled_input_carries_its_neighbouring_cell_text_as_a_label_hint(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    number = find(obs, name="F1")
    assert number.role == "textbox"
    assert number.label_hint == "Member No:"
    assert number.attrs["id"].startswith("f1_")  # volatile, so it must not be a locator
    radios = [el for el in obs.elements() if el.attrs.get("name") == "SB"]
    assert [r.label_after for r in radios] == ["Member #", "Name", "Phone"]
    assert radios[0].checked is True


def test_an_image_with_an_onclick_is_offered_with_a_box_in_top_level_coordinates(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    go = find(obs, alt="Go")
    assert go.tag == "img"
    assert go.role is None  # no semantics at all: only alt, onclick and position
    assert go.box is not None
    assert go.box.x > 170  # right of the nav frame, so frame offsets were applied
    assert go.box.y > 48  # below the header frame


def test_a_hidden_decoy_control_is_not_offered(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    surface.act(Action.click(by_text(obs, "Open Sub-Account").ref))
    obs = surface.observe()
    submits = [el for el in obs.elements() if el.attrs.get("value") == "Submit"]
    assert len(submits) == 1  # the print form's Submit is display:none


def test_rows_and_cells_that_are_the_control_are_listed_with_no_role(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    surface.act(Action.fill(find(obs, name="F1").ref, text="12345"))
    surface.act(Action.click(find(obs, alt="Go").ref))
    obs = surface.observe()
    row = by_text(obs, "TESTERSON, ADA", tag="tr")
    assert row.role is None
    assert "12345" in row.text
    surface.act(Action.click(row.ref))
    obs = surface.observe()
    button = by_text(obs, "Open Sub-Account", tag="td")
    assert button.role is None


def test_the_nested_iframe_is_its_own_frame_two_levels_down(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    acct = next(f for f in obs.frames if f.name == "acct")
    assert [hop.name for hop in acct.path] == ["main", "acct"]
    assert "SHARE SAVINGS" in acct.text
    assert "$2,480.15" in acct.text
    assert "(12.00)" in acct.text
    closes = [el for el in acct.elements if el.text == "Close"]
    assert len(closes) == 3
    assert all(el.attrs["href"].startswith("javascript:closeAcct") for el in closes)


def test_a_parents_aria_snapshot_does_not_contain_the_iframes_content(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    """The reason observe walks frames instead of trusting one snapshot (see docs/spikes.md)."""
    obs = open_member(surface, mock)
    main = next(f for f in obs.frames if f.name == "main")
    acct = next(f for f in obs.frames if f.name == "acct")
    assert "iframe" in main.aria
    assert "$2,480.15" not in main.aria
    assert "$2,480.15" in acct.aria


def test_dialogs_seen_since_the_last_observation_are_reported_once(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    surface.act(Action.click(by_text(obs, "Open Sub-Account").ref))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="F8").ref, text="25.00"))
    surface.act(Action.click(find(obs, value="Submit").ref))
    first = surface.observe()
    assert [(d.kind, d.accepted) for d in first.dialogs] == [("confirm", True)]
    assert "member 12345" in first.dialogs[0].message
    assert surface.observe().dialogs == []
