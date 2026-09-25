"""Locator harvesting and resolution on the hostile mock.

The guarantee under test: every strategy kept in a bundle was verified against the live page to
find exactly the element it was harvested from, and the bundle keeps working after the page is
re-rendered (the mock regenerates its ids on every render).
"""

from __future__ import annotations

import pytest
from tests.browser_support import by_text, find, open_member, sign_in
from tests.mockbank_support import MockHandle

from cua.artifact import (
    AncestorAnchorLocator,
    AttributeFingerprintLocator,
    CoordinatesLocator,
    FrameSelector,
    LocatorBundle,
    RoleNameLocator,
    TextLocator,
)
from cua.surface import (
    Action,
    HarvestError,
    LocatorNotFound,
    PlaywrightSurface,
    StaleRefError,
    SurfaceError,
    UnknownRefError,
)

pytestmark = pytest.mark.browser


def kinds(bundle: LocatorBundle) -> list[str]:
    return [s.kind for s in bundle.strategies]


# --- a named, semantic control -----------------------------------------------------------------


def test_a_link_is_found_by_role_and_name_first_with_its_href_as_a_fallback(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    link = by_text(obs, "Member Inquiry", tag="a")
    bundle = surface.harvest(link.ref)
    assert kinds(bundle)[0] == "role_name"
    first = bundle.strategies[0]
    assert isinstance(first, RoleNameLocator)
    assert (first.role, first.name) == ("link", "Member Inquiry")
    assert "attribute_fingerprint" in kinds(bundle)
    assert bundle.frame_path == [FrameSelector(name="nav")]
    assert all(s.rationale for s in bundle.strategies)


def test_every_kept_strategy_finds_the_element_it_was_harvested_from(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    bundle = surface.harvest(by_text(obs, "Member Inquiry", tag="a").ref)
    for index in range(len(bundle.strategies)):
        single = bundle.model_copy(update={"strategies": [bundle.strategies[index]]})
        resolved = surface.resolve(single)
        assert resolved.info is not None
        assert resolved.info.text == "Member Inquiry"


# --- unlabeled, volatile controls --------------------------------------------------------------


def test_an_unlabeled_input_is_anchored_on_its_neighbouring_cell_and_never_its_volatile_id(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    bundle = surface.harvest(find(obs, name="F1").ref)
    assert kinds(bundle) == ["ancestor_anchor", "attribute_fingerprint"]  # no role name, no label
    anchor, fingerprint = bundle.strategies
    assert isinstance(anchor, AncestorAnchorLocator)
    assert (anchor.anchor_text, anchor.container, anchor.target_tag) == (
        "Member No:",
        "row",
        "input",
    )
    assert isinstance(fingerprint, AttributeFingerprintLocator)
    assert fingerprint.attributes == {"name": "F1"}  # not the id, not a typed value
    assert "id" not in fingerprint.attributes


def test_the_bundle_still_finds_the_element_after_the_page_is_rendered_again(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    original = find(obs, name="F1")
    bundle = surface.harvest(original.ref)
    surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))  # the mock regenerates ids
    fresh = find(surface.observe(), name="F1")
    assert fresh.attrs["id"] != original.attrs["id"]
    resolved = surface.resolve(bundle)
    assert resolved.info is not None
    assert resolved.info.attrs["name"] == "F1"
    assert resolved.strategy_index == 0
    assert surface.act(Action.fill(resolved.ref or "", text="12345")).ok


def test_a_bundle_resolves_in_the_live_frame_after_the_frameset_is_reloaded_repeatedly(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    bundle = surface.harvest(by_text(obs, "Member Inquiry", tag="a").ref)
    for _ in range(3):  # each reload leaves detached frames behind in the client's bookkeeping
        surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))
        resolved = surface.resolve(bundle)
        assert resolved.info is not None
        assert resolved.info.text == "Member Inquiry"
        assert surface.act(Action.click(resolved.ref or "")).ok


def test_a_typed_value_is_never_part_of_a_fingerprint(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    box = find(obs, name="F1")
    surface.act(Action.fill(box.ref, text="12345"))
    obs = surface.observe()
    bundle = surface.harvest(find(obs, name="F1").ref)
    fingerprint = next(s for s in bundle.strategies if isinstance(s, AttributeFingerprintLocator))
    assert "value" not in fingerprint.attributes


def test_an_image_with_no_role_is_found_by_its_alt_and_source(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    bundle = surface.harvest(find(obs, alt="Go").ref, for_click=True)
    assert "role_name" not in kinds(bundle)
    fingerprint = next(s for s in bundle.strategies if isinstance(s, AttributeFingerprintLocator))
    assert fingerprint.attributes["alt"] == "Go"
    assert fingerprint.attributes["src"] == "go.gif"


def test_radios_are_told_apart_by_their_value(surface: PlaywrightSurface, mock: MockHandle) -> None:
    obs = sign_in(surface, mock)
    by_name = find(obs, name="SB", value="2")
    bundle = surface.harvest(by_name.ref)
    fingerprint = next(s for s in bundle.strategies if isinstance(s, AttributeFingerprintLocator))
    assert fingerprint.attributes["value"] == "2"
    assert surface.resolve(bundle).info is not None


def test_a_candidate_that_is_not_unique_is_dropped(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    closes = [e for f in obs.frames if f.name == "acct" for e in f.elements if e.text == "Close"]
    assert len(closes) == 3
    bundle = surface.harvest(closes[0].ref, for_click=True)
    assert "role_name" not in kinds(bundle)  # three links are called "Close"
    assert "text" not in kinds(bundle)
    fingerprint = next(s for s in bundle.strategies if isinstance(s, AttributeFingerprintLocator))
    assert "SYN-12345-S01" in fingerprint.attributes["href"]  # the one thing that differs
    assert [hop.name for hop in bundle.frame_path] == ["main", "acct"]


# --- parameterized locators --------------------------------------------------------------------


def search_for(surface: PlaywrightSurface, mock: MockHandle, number: str) -> None:
    """From a signed-in session: reload the frameset (fresh ids) and search for a member."""
    surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="F1").ref, text=number))
    surface.act(Action.click(find(obs, alt="Go").ref))


def test_a_result_row_is_found_by_the_input_it_was_searched_with_so_it_works_for_any_member(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    sign_in(surface, mock)
    search_for(surface, mock, "12345")
    row = by_text(surface.observe(), "12345", tag="tr")
    bundle = surface.harvest(row.ref, params={"member_id": "12345"}, for_click=True)
    text = next(s for s in bundle.strategies if isinstance(s, TextLocator))
    assert text.text == "{member_id}"
    assert text.exact is False
    # the rest of the row is the member's own data: it must not survive into the artifact
    assert bundle.description == "tr containing {member_id}"

    search_for(surface, mock, "12347")  # a different member, same capability
    resolved = surface.resolve(bundle, {"member_id": "12347"})
    assert resolved.info is not None
    assert "12347" in surface.page.frames[3].evaluate("document.body.innerText")
    surface.act(Action.click(resolved.ref or ""))
    assert surface.wait_for_text("SAMPLE, JORDAN")


# --- values to extract -------------------------------------------------------------------------


def test_a_value_shown_two_frames_deep_is_located_by_the_row_that_labels_it(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    open_member(surface, mock)
    bundle = surface.harvest_value("$2,480.15", anchor_text="SHARE SAVINGS")
    assert [hop.name for hop in bundle.frame_path] == ["main", "acct"]
    anchor = bundle.strategies[0]
    assert isinstance(anchor, AncestorAnchorLocator)
    assert (anchor.anchor_text, anchor.container, anchor.target_tag, anchor.nth) == (
        "SHARE SAVINGS",
        "row",
        "td",
        2,
    )
    resolved = surface.resolve(bundle)
    assert resolved.ref is not None
    assert surface.read_text(resolved.ref) == "$2,480.15"


def test_a_whole_row_quoted_as_the_anchor_is_cut_back_to_its_label(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    # a real model quoted the entire row, value and account number included: the artifact must
    # keep the label ("SHARE SAVINGS"), not one member's account number or balance
    open_member(surface, mock)
    bundle = surface.harvest_value(
        "$2,480.15",
        anchor_text="SHARE SAVINGS SYN-12345-S01 $2,480.15 Close",
        params={"member_id": "12345"},
    )
    anchor = bundle.strategies[0]
    assert isinstance(anchor, AncestorAnchorLocator)
    assert anchor.anchor_text == "SHARE SAVINGS"
    assert bundle.description == "value near 'SHARE SAVINGS'"
    assert "2,480" not in bundle.model_dump_json()
    assert "SYN-12345" not in bundle.model_dump_json()


def test_a_value_that_is_not_on_the_page_is_an_error_the_model_can_read(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    open_member(surface, mock)
    with pytest.raises(HarvestError, match="not found"):
        surface.harvest_value("$9,999.99", anchor_text="SHARE SAVINGS")


def test_an_ambiguous_value_needs_an_anchor(surface: PlaywrightSurface, mock: MockHandle) -> None:
    open_member(surface, mock)
    with pytest.raises(HarvestError, match="more than one"):
        surface.harvest_value("Close")
    bundle = surface.harvest_value("Close", anchor_text="SYN-12345-L01")
    resolved = surface.resolve(bundle)
    assert resolved.ref is not None
    assert surface.read_text(resolved.ref) == "Close"


# --- resolution --------------------------------------------------------------------------------


def test_strategies_are_tried_in_order_and_the_misses_are_reported(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    good = surface.harvest(find(obs, name="F1").ref)
    stale = TextLocator(text="Text that is not there", exact=True, rationale="drifted wording")
    bundle = good.model_copy(update={"strategies": [stale, *good.strategies]})
    resolved = surface.resolve(bundle)
    assert resolved.strategy_index == 1
    assert [(a.kind, a.outcome) for a in resolved.attempts] == [
        ("text", "no_match"),
        ("ancestor_anchor", "matched"),
    ]


def test_an_ambiguous_strategy_is_skipped_not_guessed(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    close = next(e for f in obs.frames if f.name == "acct" for e in f.elements if e.text == "Close")
    good = surface.harvest(close.ref, for_click=True)
    vague = TextLocator(text="Close", exact=True, rationale="three links say this")
    resolved = surface.resolve(good.model_copy(update={"strategies": [vague, *good.strategies]}))
    assert resolved.attempts[0].outcome == "ambiguous"
    assert resolved.strategy_index == 1


def test_a_bundle_that_finds_nothing_says_what_each_strategy_saw(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    sign_in(surface, mock)
    bundle = LocatorBundle(
        description="ghost button",
        frame_path=[FrameSelector(name="main")],
        strategies=[TextLocator(text="Ghost", exact=True, rationale="does not exist")],
    )
    with pytest.raises(LocatorNotFound) as info:
        surface.resolve(bundle)
    assert info.value.attempts[0].outcome == "no_match"
    assert "ghost button" in str(info.value)


def test_a_missing_frame_is_reported_as_such(surface: PlaywrightSurface, mock: MockHandle) -> None:
    sign_in(surface, mock)
    bundle = LocatorBundle(
        description="something",
        frame_path=[FrameSelector(name="no_such_frame")],
        strategies=[TextLocator(text="x", exact=True, rationale="whatever")],
    )
    with pytest.raises(LocatorNotFound, match="frame"):
        surface.resolve(bundle)


def test_a_coordinate_strategy_resolves_to_a_point_not_an_element(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    box = find(obs, alt="Go").box
    assert box is not None
    x, y = box.center
    bundle = LocatorBundle(
        description="Go button by position",
        strategies=[
            CoordinatesLocator(
                x=x, y=y, viewport_width=1280, viewport_height=800, rationale="no anchor verified"
            )
        ],
    )
    resolved = surface.resolve(bundle)
    assert resolved.ref is None
    assert resolved.coordinates == (x, y)
    assert resolved.strategy_kind == "coordinates"


def test_a_placeholder_without_a_value_is_an_error_not_a_blank_match(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    sign_in(surface, mock)
    bundle = LocatorBundle(
        description="row",
        frame_path=[FrameSelector(name="main")],
        strategies=[TextLocator(text="{member_id}", exact=False, rationale="the searched number")],
    )
    with pytest.raises(SurfaceError, match="member_id"):
        surface.resolve(bundle)


def test_a_resolved_element_can_be_acted_on_like_an_observed_one(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    bundle = surface.harvest(by_text(obs, "Loans", tag="a").ref, for_click=True)
    surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))
    resolved = surface.resolve(bundle)
    assert resolved.ref is not None
    assert surface.element_info(resolved.ref).text == "Loans"  # the gateway can classify it
    assert surface.act(Action.click(resolved.ref)).ok
    assert surface.wait_for_text("NOT AVAILABLE IN THIS DEMO")


# --- misuse ------------------------------------------------------------------------------------


def test_harvesting_an_unknown_or_stale_ref_fails_clearly(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    with pytest.raises(UnknownRefError):
        surface.harvest("e404")
    number = find(obs, name="F1")
    surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))
    with pytest.raises(StaleRefError):
        surface.harvest(number.ref)


def test_an_element_with_no_anchor_falls_back_to_a_positional_path(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.page.set_content(
        '<div onclick="void 0" style="width:60px;height:20px">&nbsp;</div>'
        '<div onclick="void 0" style="width:60px;height:20px">&nbsp;</div>'
    )
    obs = surface.observe()
    second = obs.frames[0].elements[1]
    bundle = surface.harvest(second.ref, for_click=True)
    assert kinds(bundle) == ["css"]
    resolved = surface.resolve(bundle)
    assert resolved.info is not None
    assert resolved.info.box is not None
    assert second.box is not None
    assert resolved.info.box.y == second.box.y  # the second div, not the first
