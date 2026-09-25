"""Recording what a person does in the live session, without recording what they type.

The person clicks and types in the same browser the run is using; the surface reports each action
as a short description. Text is never captured: only *that* something was typed and how much, so a
regulated value a person enters cannot end up in a log."""

from __future__ import annotations

import pytest
from tests.browser_support import find, sign_in
from tests.mockbank_support import MockHandle

from cua.surface import Action, HumanAction, PlaywrightSurface

pytestmark = pytest.mark.browser


def capturing(surface: PlaywrightSurface) -> list[HumanAction]:
    seen: list[HumanAction] = []
    surface.start_capture(seen.append)
    return seen


def described(seen: list[HumanAction]) -> list[str]:
    return [a.describe() for a in seen]


def test_a_click_on_a_control_is_reported_with_the_controls_own_name(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    seen = capturing(surface)
    obs = surface.observe()
    surface.act(Action.click(find(obs, type="image").ref))
    surface.pause(100)  # events reach us while the surface is pumped, as a handoff does
    clicks = [(a.frame, a.target) for a in seen if a.kind == "click"]
    assert clicks == [("", "image button")]  # the sign-in page is top level: no frame name


def test_typing_is_reported_by_length_never_by_content(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    seen = capturing(surface)
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="u").ref, text="jordan.sample"))
    surface.act(Action.fill(find(obs, name="p").ref, text="hunter22-secret"))
    surface.act(Action.click(find(obs, type="image").ref))
    typed = [a for a in seen if a.kind == "typed"]
    assert [a.chars for a in typed] == [13, None]  # the password's length is not recorded either
    everything = " ".join(described(seen)) + repr(seen)
    for value in ("jordan.sample", "hunter22-secret"):
        assert value not in everything


def test_a_password_field_is_described_as_one(surface: PlaywrightSurface, mock: MockHandle) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    seen = capturing(surface)
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="p").ref, text="whatever-it-is"))
    surface.act(Action.click(find(obs, name="u").ref))  # leaving the field commits the typing
    assert described(seen)[0].startswith("typed into a password field")
    assert seen[0].chars is None


def test_a_click_on_a_data_cell_records_where_but_never_what_it_said(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    sign_in(surface, mock)
    surface.act(Action.navigate(f"{mock.base}/msv/frameset.cgi"))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="F1").ref, text="12345"))
    surface.act(Action.click(find(obs, alt="Go").ref))
    seen = capturing(surface)
    row = next(e for e in surface.observe().elements() if e.tag == "tr" and "12345" in e.text)
    surface.act(Action.click(row.ref))
    clicks = [a for a in seen if a.kind == "click"]
    assert clicks, described(seen)
    assert "TESTERSON" not in repr(seen)  # a member's name is data, not a control label
    assert "12345" not in repr(seen)


def test_navigations_are_reported_as_paths(surface: PlaywrightSurface, mock: MockHandle) -> None:
    seen = capturing(surface)
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi?tok=abc123"))
    navigations = [a.describe() for a in seen if a.kind == "navigated"]
    assert navigations
    assert all("tok=" not in n and "abc123" not in n for n in navigations)  # never the query
    assert any("/msv/login.cgi" in n for n in navigations)


def test_nothing_is_recorded_after_capture_stops(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    seen = capturing(surface)
    surface.stop_capture()
    obs = surface.observe()
    surface.act(Action.click(find(obs, type="image").ref))
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert seen == []


def test_starting_twice_does_not_double_report(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    first = capturing(surface)
    second = capturing(surface)  # replaces the sink; the page must not report twice
    obs = surface.observe()
    surface.act(Action.click(find(obs, type="image").ref))
    assert first == []
    assert len([a for a in second if a.kind == "click"]) == 1


def test_stopping_without_starting_is_harmless(surface: PlaywrightSurface) -> None:
    surface.stop_capture()


def test_pages_loaded_after_capture_started_are_covered_too(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    seen = capturing(surface)
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))  # loaded after start
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="u").ref, text="abcdef"))
    surface.act(Action.click(find(obs, name="p").ref))
    assert any(a.kind == "typed" and a.chars == 6 for a in seen), described(seen)
