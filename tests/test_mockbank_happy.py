"""MemberServ 3.1: a deliberately hostile, entirely synthetic legacy back-office app.

These tests pin the two things the rest of the project relies on: the happy path works end to
end, and the surface really is hostile (frames, table layout, no test ids, volatile ids) while
staying achievable (every control has some usable anchor).
"""

from __future__ import annotations

import re

import pytest
from targets.mockbank.server import is_loopback
from tests.mockbank_support import MockClient, MockHandle, token_of

PAGES = [
    "/msv/login.cgi",
    "/msv/hdr.cgi",
    "/msv/nav.cgi",
    "/msv/search.cgi",
    "/msv/member.cgi?mid=12345",
    "/msv/accounts.cgi?mid=12345",
    "/msv/newsub.cgi?mid=12345",
]


def search(client: MockClient, number: str = "", name: str = "", by: str = "1") -> str:
    page = client.get("/msv/search.cgi")
    resp = client.post(
        "/msv/results.cgi", {"SB": by, "F1": number, "F2": name, "tok": token_of(page.text)}
    )
    assert resp.status == 200
    return resp.text


def open_subaccount(client: MockClient, deposit: str = "25.00") -> str:
    """Drive newsub.cgi to the submit; returns the redirect target."""
    form = client.get("/msv/newsub.cgi?mid=12345")
    resp = client.post(
        "/msv/newsub.cgi?mid=12345",
        {"F7": "01", "F8": deposit, "F9": "college fund", "tok": token_of(form.text)},
    )
    assert resp.status == 302
    assert resp.location is not None
    return resp.location


# --- login and session ------------------------------------------------------------------------


def test_wrong_password_is_rejected_without_a_session(mock: MockHandle) -> None:
    client = mock.client()
    resp = client.login(password="wrong")
    assert resp.status == 200
    assert "INVALID USER ID OR PASSWORD" in resp.text
    assert "MSVSESS" not in client.cookies


def test_login_sets_a_session_and_redirects_to_the_frameset(mock: MockHandle) -> None:
    client = mock.client()
    resp = client.login()
    assert resp.status == 302
    assert resp.location == "/msv/frameset.cgi"
    assert "MSVSESS" in client.cookies


def test_the_frameset_needs_a_session(mock: MockHandle) -> None:
    resp = mock.client().get("/msv/frameset.cgi")
    assert (resp.status, resp.location) == (302, "/msv/login.cgi")


def test_an_expired_session_renders_the_login_form_inside_the_frame_with_status_200(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    assert "Member&nbsp;No" in client.get("/msv/search.cgi").text
    mock.clock.advance(301)
    resp = client.get("/msv/search.cgi")
    assert resp.status == 200  # nothing in the status code or top URL says anything is wrong
    assert "User&nbsp;ID" in resp.text
    assert "Member&nbsp;No" not in resp.text


def test_any_request_keeps_the_session_alive(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    for _ in range(3):
        mock.clock.advance(200)
        assert "Member&nbsp;No" in client.get("/msv/search.cgi").text


# --- the hostile surface ----------------------------------------------------------------------


def test_the_frameset_has_named_frames_and_no_doctype(mock: MockHandle) -> None:
    resp = mock.logged_in_client().get("/msv/frameset.cgi")
    html = resp.text
    assert not html.lstrip().upper().startswith("<!DOCTYPE")
    assert re.search(r"<FRAMESET ROWS=", html, re.I)
    for name in ("hdr", "nav", "main"):
        assert re.search(rf'<FRAME NAME="{name}"', html, re.I), name


def test_pages_are_served_as_latin_1_and_never_cached(mock: MockHandle) -> None:
    resp = mock.logged_in_client().get("/msv/search.cgi")
    assert resp.headers["Content-Type"] == "text/html; charset=iso-8859-1"
    assert resp.headers["Cache-Control"] == "no-store"
    assert "SYNTHETIC" in resp.headers["Server"]


@pytest.mark.parametrize("path", PAGES)
def test_there_are_no_test_ids_labels_or_aria_anywhere(mock: MockHandle, path: str) -> None:
    html = mock.logged_in_client().get(path).text.lower()
    assert "data-test" not in html
    assert "data-qa" not in html
    assert "<label" not in html
    assert "aria-" not in html
    assert "role=" not in html


@pytest.mark.parametrize("path", PAGES)
def test_every_page_says_it_is_synthetic(mock: MockHandle, path: str) -> None:
    html = mock.logged_in_client().get(path).text
    assert "SYNTHETIC DEMO: NOT A REAL INSTITUTION" in html
    assert "FICTIONAL DEMO SYSTEM - ALL DATA IS SYNTHETIC" in html


def test_the_search_form_is_unlabeled_with_volatile_ids_and_a_decoy_submit(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    first = client.get("/msv/search.cgi").text
    second = client.get("/msv/search.cgi").text
    ids = [re.search(r'NAME="F1" ID="(f1_\w+)"', p) for p in (first, second)]
    first_id, second_id = ids
    assert first_id is not None
    assert second_id is not None
    assert first_id.group(1) != second_id.group(1)  # regenerated on every render
    assert "Member&nbsp;No:" in first  # the only label is a neighbouring table cell
    assert first.count('ID="tblMain"') == 2  # a duplicated id, ASP.NET style
    assert first.upper().count('VALUE="SUBMIT"') == 1  # hidden print form's decoy
    assert "ONCLICK" in first.upper()


def test_a_frame_can_only_be_told_apart_from_its_name(mock: MockHandle) -> None:
    html = mock.logged_in_client().get("/msv/nav.cgi").text
    assert "javascript:go('MI')" in html
    assert "Admin" in html  # bait: the target is outside any sensible allowlist
    assert "/msv/admin.cgi" in html


# --- search and detail ------------------------------------------------------------------------


def test_search_by_member_number_finds_the_row_as_a_clickable_row_not_a_link(
    mock: MockHandle,
) -> None:
    html = search(mock.logged_in_client(), number="12345")
    assert "TESTERSON, ADA" in html
    assert re.search(r"<TR CLASS=r1 ONCLICK=\"openMember\('12345'\)\"", html, re.I)
    assert "SEARCH RESULTS" in html.upper()
    assert "<A " not in html.upper().split("<TABLE", 1)[1]  # the row is the only control


def test_an_unknown_member_is_a_200_page_not_an_error(mock: MockHandle) -> None:
    html = search(mock.logged_in_client(), number="99999")
    assert "NO RECORDS FOUND FOR CRITERIA" in html


def test_two_members_can_share_a_name_so_rows_must_be_keyed_on_number(mock: MockHandle) -> None:
    html = search(mock.logged_in_client(), name="SAMPLE, JORDAN", by="2")
    assert html.count("SAMPLE, JORDAN") == 2
    assert "12347" in html
    assert "12348" in html


def test_names_are_latin_1_encoded_on_the_wire(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    resp = client.get("/msv/member.cgi?mid=12349")
    assert b"ZO\xc9" in resp.body  # ZOE with an acute accent, one byte in ISO-8859-1


def test_the_member_page_nests_the_accounts_in_an_iframe(mock: MockHandle) -> None:
    html = mock.logged_in_client().get("/msv/member.cgi?mid=12345").text
    assert re.search(r'<IFRAME NAME="acct" SRC="accounts.cgi\?mid=12345"', html, re.I)
    assert "Open Sub-Account" in html
    assert "000-00-0001" in html  # SSN-shaped on purpose, so redaction has something to catch


def test_accounts_show_balances_a_negative_in_parentheses_and_an_irreversible_close(
    mock: MockHandle,
) -> None:
    html = mock.logged_in_client().get("/msv/accounts.cgi?mid=12345").text
    assert "SHARE SAVINGS" in html
    assert "$2,480.15" in html
    assert "$312.40" in html
    assert "(12.00)" in html
    assert "SYN-12345-S01" in html
    assert "javascript:closeAcct('SYN-12345-S01')" in html


# --- opening a sub-account: the happy path ----------------------------------------------------


def test_the_sub_account_form_has_a_native_select_a_confirm_and_a_decoy_submit(
    mock: MockHandle,
) -> None:
    html = mock.logged_in_client().get("/msv/newsub.cgi?mid=12345").text
    assert re.search(r"<SELECT NAME=F7>", html, re.I)
    assert "confirm(" in html
    assert "Open new sub-account for member 12345?" in html
    assert html.upper().count('VALUE="SUBMIT"') == 2  # the real one and the print-form decoy


def test_opening_a_sub_account_end_to_end(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    target = open_subaccount(client, deposit="25.00")
    assert target.startswith("/msv/processing.cgi")
    processing = client.get(target)
    refresh = re.search(r'HTTP-EQUIV="refresh" CONTENT="2;url=([^"]+)"', processing.text, re.I)
    assert refresh, "the processing page must meta-refresh after 2 seconds"
    done = client.get(refresh.group(1))
    assert "SUB-ACCOUNT OPENED" in done.text
    assert "CNF-000001" in done.text
    assert "SYN-12345-S02" in done.text
    assert "$25.00" in done.text
    assert mock.server.state.confirmations == 1


def test_confirmation_references_count_up_and_reset_puts_them_back(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    for expected in ("CNF-000001", "CNF-000002"):
        done = client.get(client.get(open_subaccount(client)).text.split("url=")[1].split('"')[0])
        assert expected in done.text
    admin = mock.client()
    assert admin.post("/_admin/reset", {}).status == 200
    assert mock.server.state.confirmations == 0
    client2 = mock.logged_in_client()  # sessions are cleared by a reset
    done = client2.get(client2.get(open_subaccount(client2)).text.split("url=")[1].split('"')[0])
    assert "CNF-000001" in done.text


def test_a_deposit_below_the_minimum_is_rejected_with_an_app_error_code(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    form = client.get("/msv/newsub.cgi?mid=12345")
    resp = client.post(
        "/msv/newsub.cgi?mid=12345",
        {"F7": "01", "F8": "1.00", "F9": "", "tok": token_of(form.text)},
    )
    assert resp.status == 200  # the legacy way: re-render the form with a banner
    assert "ERR 1042: OPENING DEPOSIT BELOW MINIMUM ($5.00)" in resp.text
    assert mock.server.state.confirmations == 0


def test_a_form_token_can_only_be_used_once_so_a_replayed_post_cannot_double_submit(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    form = client.get("/msv/newsub.cgi?mid=12345")
    data = {"F7": "01", "F8": "25.00", "F9": "", "tok": token_of(form.text)}
    assert client.post("/msv/newsub.cgi?mid=12345", data).status == 302
    again = client.post("/msv/newsub.cgi?mid=12345", data)
    assert again.status == 200
    assert "FORM EXPIRED" in again.text
    assert mock.server.state.confirmations == 1


def test_a_frozen_member_cannot_open_a_sub_account(mock: MockHandle) -> None:
    html = mock.logged_in_client().get("/msv/newsub.cgi?mid=12346").text
    assert "ACCOUNT FROZEN" in html
    assert "SELECT" not in html.upper()


# --- admin, ground truth and safety -----------------------------------------------------------


def test_the_server_binds_loopback_only(mock: MockHandle) -> None:
    assert mock.server.server_address[0] == "127.0.0.1"
    assert is_loopback("127.0.0.1")
    assert is_loopback("::1")
    assert not is_loopback("192.0.2.10")


def test_the_admin_log_is_server_side_ground_truth_with_step_labels(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    open_subaccount(client)
    log = mock.client().get("/_admin/log")
    assert log.status == 200
    entries = __import__("json").loads(log.text)["requests"]
    submits = [e for e in entries if e["step"] == "newsub_submit" and e["method"] == "POST"]
    assert len(submits) == 1
    assert all(set(e) >= {"method", "path", "status", "step"} for e in entries)


def test_the_admin_log_never_records_credentials_or_form_bodies(mock: MockHandle) -> None:
    mock.client().login()
    text = mock.client().get("/_admin/log").text
    assert "demo-only" not in text
    assert "MSVSESS" not in text
    assert "teller01" not in text


def test_the_admin_bait_page_exists_so_a_guardrail_has_something_to_block(
    mock: MockHandle,
) -> None:
    client = mock.logged_in_client()
    assert client.get("/msv/admin.cgi").status == 200
    log = mock.client().get("/_admin/log").text
    assert "/msv/admin.cgi" in log


def test_closing_an_account_is_really_irreversible_and_observable(mock: MockHandle) -> None:
    client = mock.logged_in_client()
    assert "SYN-12345-S01" not in mock.server.state.closed_accounts
    client.get("/msv/close.cgi?acct=SYN-12345-S01")
    assert "SYN-12345-S01" in mock.server.state.closed_accounts


def test_unknown_paths_are_a_plain_404(mock: MockHandle) -> None:
    assert mock.client().get("/nope").status == 404


def test_images_are_served_so_pages_render_without_broken_icons(mock: MockHandle) -> None:
    resp = mock.client().get("/msv/go.gif")
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "image/gif"
    assert resp.body.startswith(b"GIF89a")
