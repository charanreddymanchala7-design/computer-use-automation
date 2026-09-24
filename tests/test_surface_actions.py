"""What the surface lets the agent *do*: refs, coordinates, secrets, dialogs, waiting, lifecycle."""

from __future__ import annotations

import pytest
from tests.browser_support import by_text, find, open_member, sign_in
from tests.mockbank_support import MockHandle

from cua.surface import (
    Action,
    PlaywrightSurface,
    StaleRefError,
    UnknownRefError,
    UnknownSecretError,
)

pytestmark = pytest.mark.browser


# --- acting by ref ----------------------------------------------------------------------------


def test_a_javascript_link_in_one_frame_drives_another_frame(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    loans = by_text(obs, "Loans", tag="a")
    result = surface.act(Action.click(loans.ref))
    assert result.ok
    main = surface.observe().frames[3]
    assert "LOANS" in main.text
    assert "NOT AVAILABLE IN THIS DEMO" in main.text


def test_fill_then_observe_shows_the_typed_value(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    assert surface.act(Action.fill(find(obs, name="F1").ref, text="12345")).ok
    assert find(surface.observe(), name="F1").attrs["value"] == "12345"


def test_clicking_an_image_by_ref_runs_the_search(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    surface.act(Action.fill(find(obs, name="F1").ref, text="12345"))
    surface.act(Action.click(find(obs, alt="Go").ref))
    assert "TESTERSON, ADA" in surface.observe().frames[3].text


def test_clicking_by_coordinates_works_where_there_is_no_usable_locator(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    surface.act(Action.fill(find(obs, name="F1").ref, text="12345"))
    box = find(obs, alt="Go").box
    assert box is not None
    x, y = box.center
    assert surface.act(Action.click_at(x, y)).ok
    assert "TESTERSON, ADA" in surface.observe().frames[3].text


def test_a_native_select_can_be_set_by_label_or_value(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    surface.act(Action.click(by_text(obs, "Open Sub-Account").ref))
    obs = surface.observe()
    select = find(obs, name="F7")
    assert select.role == "combobox"
    assert select.options == ["Share Savings", "Holiday Club"]
    assert surface.act(Action.select(select.ref, "Holiday Club")).ok
    assert surface.observe().frames[3].text  # still rendered
    assert find(surface.observe(), name="F7").text == "Holiday Club"
    assert surface.act(Action.select(select.ref, "01")).ok
    assert find(surface.observe(), name="F7").text == "Share Savings"


def test_the_confirm_dialog_is_accepted_and_the_flow_completes_after_the_meta_refresh(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = open_member(surface, mock)
    surface.act(Action.click(by_text(obs, "Open Sub-Account").ref))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="F8").ref, text="25.00"))
    result = surface.act(Action.click(find(obs, value="Submit").ref))
    assert [(d.kind, d.message, d.accepted) for d in result.dialogs] == [
        ("confirm", "Open new sub-account for member 12345?", True)
    ]
    assert surface.wait_for_text("SUB-ACCOUNT OPENED", timeout_ms=8000)
    assert mock.server.state.confirmations == 1
    assert "CNF-000001" in surface.observe().frames[3].text


def test_press_submits_a_form_with_an_image_button(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    surface.act(Action.fill(find(obs, name="u").ref, secret="MOCK_USER"))
    password = find(obs, name="p")
    surface.act(Action.fill(password.ref, secret="MOCK_PASS"))
    assert surface.act(Action.press("Enter")).ok
    assert surface.wait_for_text("Member No", timeout_ms=5000)


def test_navigate_goes_to_an_absolute_url_and_reports_where_it_landed(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    result = surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert result.ok
    assert result.url == f"{mock.base}/msv/login.cgi"


def test_wait_for_text_returns_false_on_timeout_instead_of_raising(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    assert surface.wait_for_text("no such text anywhere", timeout_ms=300) is False


def test_wait_is_a_bounded_pause(
    surface: PlaywrightSurface, mock: MockHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    waited: list[float] = []
    monkeypatch.setattr(surface.page, "wait_for_timeout", waited.append)
    assert surface.act(Action.wait(50)).detail == "waited 50 ms"
    assert surface.act(Action.wait(10_000_000)).detail == "waited 30000 ms"  # clamped, not slept
    assert surface.act(Action.wait(-5)).detail == "waited 0 ms"
    assert max(waited) == 30000  # the huge request was clamped to the cap, never slept
    assert 0 in waited


# --- refs are only valid for the observation that issued them ----------------------------------


def test_an_unknown_ref_is_a_protocol_error(surface: PlaywrightSurface, mock: MockHandle) -> None:
    sign_in(surface, mock)
    with pytest.raises(UnknownRefError, match="e9999"):
        surface.act(Action.click("e9999"))


def test_a_ref_from_before_a_navigation_is_stale_not_silently_wrong(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    number = find(obs, name="F1")
    surface.act(Action.fill(number.ref, text="12345"))
    surface.act(Action.click(find(obs, alt="Go").ref))  # the search page is replaced
    with pytest.raises(StaleRefError, match="observe again"):
        surface.act(Action.fill(number.ref, text="1"))


# --- secrets ----------------------------------------------------------------------------------


def test_a_secret_is_typed_from_its_name_and_never_echoed(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    result = surface.act(Action.fill(find(obs, name="p").ref, secret="MOCK_PASS"))
    assert result.ok
    assert "demo-only" not in result.detail
    assert "MOCK_PASS" in result.detail
    typed = surface.page.frames[0].evaluate("document.querySelector('input[name=p]').value")
    assert typed == "demo-only"


def test_an_unknown_secret_name_is_an_error(surface: PlaywrightSurface, mock: MockHandle) -> None:
    surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
    obs = surface.observe()
    with pytest.raises(UnknownSecretError, match="NOPE"):
        surface.act(Action.fill(find(obs, name="p").ref, secret="NOPE"))


def test_fill_needs_exactly_one_of_text_or_secret() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Action.fill("e1")
    with pytest.raises(ValueError, match="exactly one"):
        Action.fill("e1", text="a", secret="B")


# --- reset ---


def test_reset_clears_the_session_and_forgets_refs(
    surface: PlaywrightSurface, mock: MockHandle
) -> None:
    obs = sign_in(surface, mock)
    ref = find(obs, name="F1").ref
    surface.reset()
    with pytest.raises(UnknownRefError):
        surface.act(Action.fill(ref, text="1"))
    surface.act(Action.navigate(f"{mock.base}/msv/search.cgi"))
    assert "User ID" in surface.observe().frames[0].text  # the session cookie is gone
