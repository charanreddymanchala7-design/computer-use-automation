"""Helpers for tests that drive the real Chromium surface against the MemberServ mock."""

from __future__ import annotations

from tests.mockbank_support import MockHandle

from cua.surface import Action, ElementInfo, Observation, PlaywrightSurface


def find(obs: Observation, **attrs: str) -> ElementInfo:
    """The one element whose attributes match; fails loudly if there are zero or several."""
    matches = [
        el
        for el in obs.elements()
        if all(el.attrs.get(key) == value for key, value in attrs.items())
    ]
    assert len(matches) == 1, f"expected one element with {attrs}, found {len(matches)}"
    return matches[0]


def by_text(obs: Observation, text: str, *, tag: str | None = None) -> ElementInfo:
    matches = [el for el in obs.elements() if text in el.text and (tag is None or el.tag == tag)]
    assert matches, f"no element with text {text!r}"
    return matches[0]


def sign_in(surface: PlaywrightSurface, mock: MockHandle) -> Observation:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="u").ref, secret="MOCK_USER"))
    surface.act(Action.fill(find(obs, name="p").ref, secret="MOCK_PASS"))
    surface.act(Action.click(find(obs, type="image").ref))
    return surface.observe()


def open_member(surface: PlaywrightSurface, mock: MockHandle, number: str = "12345") -> Observation:
    """Sign in, search for a member number and open the detail page."""
    obs = sign_in(surface, mock)
    surface.act(Action.fill(find(obs, name="F1").ref, text=number))
    surface.act(Action.click(find(obs, alt="Go").ref))
    obs = surface.observe()
    surface.act(Action.click(by_text(obs, number, tag="tr").ref))
    return surface.observe()
