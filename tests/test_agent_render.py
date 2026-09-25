"""What the model is shown: a compact, honest description of every frame and its controls."""

from __future__ import annotations

from tests.agent_support import BASE, element, frame, page

from cua.agent import render_observation
from cua.surface import DialogEvent, FrameHop, FrameObservation


def test_each_frame_is_introduced_with_its_name_path_and_url() -> None:
    top = frame(0, None, "", [])
    acct = FrameObservation(
        index=2,
        path=(FrameHop("main", 0, "u"), FrameHop("acct", 0, "u")),
        name="acct",
        url=f"{BASE}/msv/accounts.cgi?mid=12345",
        text="SHARE SAVINGS $2,480.15",
        aria="",
        elements=[],
    )
    text = render_observation(page(top, acct, url=f"{BASE}/msv/frameset.cgi"))
    assert "URL: /msv/frameset.cgi" in text
    assert '[frame 2 "acct" path main > acct /msv/accounts.cgi?mid=12345]' in text
    assert "SHARE SAVINGS $2,480.15" in text
    assert "[frame 0] (a frameset: its content is in the frames below)" in text


def test_elements_show_ref_role_text_label_and_the_attributes_that_matter() -> None:
    field = element("e5", "input", role="textbox", label_hint="Member No:", name="F1", value="123")
    link = element("e6", "a", "Member Inquiry", role="link", href="javascript:go('MI')")
    image = element("e7", "img", "Go", alt="Go", src="go.gif", onclick="doSearch()")
    text = render_observation(page(frame(1, "main", "Search", [field, link, image])))
    assert 'e5 textbox label="Member No:" name=F1 value="123"' in text
    assert "e6 link \"Member Inquiry\" href=javascript:go('MI')" in text
    assert 'e7 img (no role, clickable) "Go"' in text
    assert "alt=Go" in text
    assert "src=go.gif" in text


def test_selects_show_their_options_and_radios_their_state() -> None:
    select = element("e1", "select", "Share Savings", role="combobox", name="F7")
    select.options = ["Share Savings", "Holiday Club"]
    radio = element("e2", "input", role="radio", name="SB", type="radio", value="1")
    radio.checked = True
    radio.label_after = "Member #"
    text = render_observation(page(frame(0, "main", "", [select, radio])))
    assert 'options=["Share Savings", "Holiday Club"]' in text
    assert "checked" in text
    assert 'after="Member #"' in text


def test_volatile_ids_are_left_out_because_they_would_mislead() -> None:
    field = element("e1", "input", role="textbox", name="F1", id="f1_a83k")
    assert "f1_a83k" not in render_observation(page(frame(0, "main", "", [field])))


def test_dialogs_handled_since_the_last_observation_are_reported() -> None:
    obs = page(frame(0, "main", "x"))
    obs.dialogs = [DialogEvent("confirm", "Open new sub-account for member 12345?", True)]
    text = render_observation(obs)
    assert 'Dialogs handled: confirm "Open new sub-account for member 12345?" -> accepted' in text


def test_long_frame_text_is_cut_so_one_page_cannot_swamp_the_prompt() -> None:
    text = render_observation(page(frame(0, "main", "x" * 5000)), max_text=100)
    assert "x" * 100 in text
    assert "x" * 101 not in text
    assert "truncated" in text


def test_an_unreadable_frame_says_so_instead_of_pretending_to_be_empty() -> None:
    broken = frame(1, "main", "")
    broken.unreadable = True
    assert "(this frame changed while it was being read: observe again)" in render_observation(
        page(broken)
    )


def test_a_frame_with_nothing_in_it_is_reported_as_empty() -> None:
    assert '[frame 1 "main" /msv/x.cgi] (empty)' in render_observation(page(frame(1, "main")))
