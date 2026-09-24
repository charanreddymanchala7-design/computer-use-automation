"""HTML for MemberServ 3.1, written the way 1999 wrote it: uppercase tags, no doctype, table
layout, neighbouring-cell labels, javascript: links, onclick on cells and rows, volatile ids.

Every control keeps at least one usable anchor (visible text, name, or alt) so the flow stays
achievable for a careful agent. The hostility is the absence of test ids, roles, labels and
stable ids, not impossibility.
"""

from __future__ import annotations

import secrets
from html import escape

from targets.mockbank.data import BUSINESS_DATE, PRODUCTS, Account, Member, money

_STYLE = (
    "<STYLE>BODY,TD{font-family:Verdana,sans-serif;font-size:11px;color:#333333}"
    ".btn{background:#4A6FA5;color:#FFFFFF;cursor:pointer;padding:3px 8px}"
    ".r1{background:#EEEEEE;cursor:pointer}.r2{background:#DDDDDD;cursor:pointer}"
    ".red{color:#CC0000;font-weight:bold}</STYLE>"
)
_GIF = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,"
    b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)
GIF_BYTES = _GIF


def _rand4() -> str:
    return secrets.token_hex(2)


def page(title: str, body: str, head: str = "") -> str:
    return (
        f"<HTML><HEAD><TITLE>{title}</TITLE>\n"
        "<!-- SYNTHETIC DEMO: NOT A REAL INSTITUTION -->\n"
        f"{_STYLE}{head}</HEAD>\n"
        f'<BODY BGCOLOR="#D8D8D0" TEXT="#333333">\n{body}\n'
        '<P><FONT SIZE=1 COLOR="#999999" STYLE="font-size:9px">'
        "FICTIONAL DEMO SYSTEM - ALL DATA IS SYNTHETIC</FONT></BODY></HTML>"
    )


def banner(text: str) -> str:
    return f'<P><FONT COLOR="#CC0000" CLASS=red>{escape(text, quote=False)}</FONT></P>'


def hidden_token(tok: str) -> str:
    return f"<INPUT TYPE=HIDDEN NAME=tok VALUE={tok}>"


# --- chrome -----------------------------------------------------------------------------------


def login_page(tok: str, error: str | None = None) -> str:
    err = banner(error) if error else ""
    body = (
        "<TABLE WIDTH=360 CELLPADDING=4 CELLSPACING=0 BORDER=0 ALIGN=CENTER><TR><TD COLSPAN=2>"
        "<FONT SIZE=3>MEMBERSERV 3.1 SIGN-ON</FONT><BR>EXAMPLE FCU (DEMO)</TD></TR></TABLE>"
        f"{err}"
        '<FORM METHOD=POST ACTION="/msv/login.cgi" TARGET="_top">'
        f"{hidden_token(tok)}"
        "<TABLE WIDTH=360 CELLPADDING=4 CELLSPACING=0 ALIGN=CENTER>"
        '<TR><TD ALIGN=RIGHT WIDTH=120><FONT SIZE=1 COLOR="#666666">User&nbsp;ID</FONT></TD>'
        "<TD><INPUT NAME=u SIZE=12></TD></TR>"
        '<TR><TD ALIGN=RIGHT><FONT SIZE=1 COLOR="#666666">Password</FONT></TD>'
        "<TD><INPUT TYPE=PASSWORD NAME=p SIZE=12></TD></TR>"
        '<TR><TD></TD><TD><INPUT TYPE=IMAGE SRC="/msv/login.gif" ALT="" WIDTH=60 HEIGHT=18 '
        'STYLE="background:#4A6FA5"></TD></TR></TABLE></FORM>'
    )
    return page("MemberServ 3.1", body)


def frameset_page() -> str:
    # No doctype, on purpose (quirks mode). The top URL never changes after login.
    return (
        "<HTML><HEAD><TITLE>MemberServ 3.1</TITLE>\n"
        "<!-- SYNTHETIC DEMO: NOT A REAL INSTITUTION -->\n</HEAD>\n"
        '<FRAMESET ROWS="48,*" BORDER=1>\n'
        '  <FRAME NAME="hdr" SRC="hdr.cgi" SCROLLING=NO>\n'
        '  <FRAMESET COLS="170,*">\n'
        '    <FRAME NAME="nav" SRC="nav.cgi">\n'
        '    <FRAME NAME="main" SRC="search.cgi">\n'
        "  </FRAMESET>\n</FRAMESET></HTML>"
    )


def hdr_page(user: str) -> str:
    body = (
        '<TABLE WIDTH="100%" CELLPADDING=2 CELLSPACING=0 BORDER=0><TR>'
        f"<TD>{BUSINESS_DATE}</TD><TD>User: {escape(user.upper())}</TD>"
        '<TD ALIGN=RIGHT><A HREF="javascript:logoff()">Log Off</A></TD></TR></TABLE>'
        "<SCRIPT>function logoff(){top.location='/msv/logoff.cgi';}</SCRIPT>"
    )
    return page("hdr", body)


def nav_page() -> str:
    body = (
        "<SCRIPT>function go(k){var m={MI:'search.cgi',LN:'loans.cgi',RP:'reports.cgi',"
        "AD:'/msv/admin.cgi'};parent.main.location=m[k];}</SCRIPT>"
        "<TABLE CELLPADDING=3 CELLSPACING=0 BORDER=0><TR><TD><TABLE CELLPADDING=2 BORDER=0>"
        "<TR><TD><A HREF=\"javascript:go('MI')\">Member&nbsp;Inquiry</A></TD></TR>"
        "<TR><TD><A HREF=\"javascript:go('LN')\">Loans</A></TD></TR>"
        '<TR><TD><IMG SRC="/msv/go.gif" ONCLICK="go(\'RP\')" WIDTH=60 HEIGHT=14 '
        'STYLE="background:#8899AA"></TD></TR>'
        "<TR><TD><A HREF=\"javascript:go('AD')\">Admin</A></TD></TR>"
        "</TABLE></TD></TR></TABLE>"
    )
    return page("nav", body)


def stub_page(title: str) -> str:
    return page(title, f"<P>{escape(title)}</P><P>NOT AVAILABLE IN THIS DEMO</P>")


def admin_page() -> str:
    return page(
        "ADMINISTRATION",
        "<P>ADMINISTRATION - AUTHORIZED PERSONNEL ONLY</P>"
        "<TABLE BORDER=1 CELLPADDING=3><TR><TD>USER</TD><TD>ROLE</TD></TR>"
        "<TR><TD>TELLER01</TD><TD>TELLER</TD></TR><TR><TD>SUPER01</TD><TD>SUPERVISOR</TD></TR>"
        "</TABLE>",
    )


# --- member inquiry ---------------------------------------------------------------------------


def search_page(tok: str) -> str:
    body = (
        "<SCRIPT>function doSearch(){document.forms[0].submit();}</SCRIPT>"
        '<FORM METHOD=POST ACTION="results.cgi">'
        f"{hidden_token(tok)}"
        '<TABLE ID="tblMain" WIDTH=600 CELLPADDING=2 CELLSPACING=0><TR><TD>'
        '<TABLE ID="tblMain" WIDTH=560 CELLPADDING=2 CELLSPACING=0>'
        '<TR><TD ALIGN=RIGHT WIDTH=120><FONT SIZE=1 COLOR="#666666">Search&nbsp;by:</FONT></TD>'
        "<TD><INPUT TYPE=RADIO NAME=SB VALUE=1 CHECKED>Member&nbsp;#&nbsp;"
        "<INPUT TYPE=RADIO NAME=SB VALUE=2>Name&nbsp;<INPUT TYPE=RADIO NAME=SB VALUE=3>Phone"
        "</TD><TD></TD></TR>"
        '<TR><TD ALIGN=RIGHT WIDTH=120><FONT SIZE=1 COLOR="#666666">Member&nbsp;No:</FONT></TD>'
        f'<TD><INPUT NAME="F1" ID="f1_{_rand4()}" SIZE=10 MAXLENGTH=8></TD>'
        '<TD><IMG SRC="go.gif" ONCLICK="doSearch()" ALT="Go" WIDTH=28 HEIGHT=16 '
        'STYLE="background:#4A6FA5"></TD></TR>'
        '<TR><TD ALIGN=RIGHT><FONT SIZE=1 COLOR="#666666">Name:</FONT></TD>'
        f'<TD COLSPAN=2><INPUT NAME="F2" ID="f2_{_rand4()}" SIZE=24></TD></TR>'
        "</TABLE></TD></TR></TABLE></FORM>"
        '<FORM STYLE="display:none" METHOD=POST ACTION="print.cgi">'
        '<INPUT TYPE=SUBMIT VALUE="Submit"></FORM>'
    )
    return page("Member Inquiry", body)


def results_page(rows: list[Member]) -> str:
    if rows:
        table_rows = "".join(
            f"<TR CLASS=r{1 + i % 2} ONCLICK=\"openMember('{m.number}')\">"
            f"<TD>{m.number}</TD><TD>{escape(m.name, quote=False)}</TD><TD>{m.status}</TD></TR>"
            for i, m in enumerate(rows)
        )
        table = (
            "<TABLE WIDTH=520 CELLPADDING=3 CELLSPACING=1 BORDER=0>"
            '<TR><TD COLSPAN=3 STYLE="text-transform:uppercase">Search results</TD></TR>'
            "<TR><TD>Member</TD><TD>Name</TD><TD>Status</TD></TR>"
            f"{table_rows}</TABLE>"
        )
    else:
        table = (
            '<TABLE WIDTH=520 CELLPADDING=3><TR><TD STYLE="text-transform:uppercase">'
            "Search results</TD></TR><TR><TD>"
            '<FONT COLOR="#CC0000">NO RECORDS FOUND FOR CRITERIA</FONT></TD></TR></TABLE>'
        )
    script = "<SCRIPT>function openMember(n){location='member.cgi?mid='+n;}</SCRIPT>"
    return page("Results", script + table)


def member_page(member: Member) -> str:
    n = member.number
    body = (
        "<TABLE WIDTH=560 CELLPADDING=2 CELLSPACING=0 BORDER=0><TR><TD>"
        "<TABLE WIDTH=540 CELLPADDING=2 CELLSPACING=0 BORDER=0>"
        f"<TR><TD COLSPAN=2><FONT SIZE=3>{escape(member.name, quote=False)}</FONT></TD></TR>"
        f"<TR><TD>Member:</TD><TD>{n}</TD></TR>"
        f"<TR><TD>Status:</TD><TD>{member.status}</TD></TR>"
        f"<TR><TD>Branch:</TD><TD>{member.branch}</TD></TR>"
        f"<TR><TD>Member&nbsp;since:</TD><TD>{member.since}</TD></TR>"
        f"<TR><TD>TIN:</TD><TD>{member.tin}</TD></TR>"
        f"<TR><TD>Phone:</TD><TD>{member.phone}</TD></TR>"
        "</TABLE></TD></TR><TR><TD>"
        f'<IFRAME NAME="acct" SRC="accounts.cgi?mid={n}" WIDTH=540 HEIGHT=150 FRAMEBORDER=0>'
        "</IFRAME></TD></TR>"
        f"<TR><TD><TABLE><TR><TD CLASS=btn ONCLICK=\"location='newsub.cgi?mid={n}'\">"
        "Open Sub-Account...</TD></TR></TABLE></TD></TR></TABLE>"
    )
    return page("Member", body)


def accounts_page(member: Member, closed: set[str]) -> str:
    def row(a: Account) -> str:
        state = (
            "CLOSED" if a.id in closed else f"<A HREF=\"javascript:closeAcct('{a.id}')\">Close</A>"
        )
        return (
            f"<TR><TD>{a.kind}</TD><TD>{a.id}</TD>"
            f"<TD ALIGN=RIGHT>{money(a.balance_cents)}</TD><TD>{state}</TD></TR>"
        )

    script = (
        "<SCRIPT>function closeAcct(id){if(confirm('Close account '+id+'?'))"
        "location='close.cgi?acct='+id;}</SCRIPT>"
    )
    body = (
        f"{script}<TABLE WIDTH=520 CELLPADDING=3 CELLSPACING=1 BORDER=0>"
        "<TR><TD>Type</TD><TD>Account</TD><TD ALIGN=RIGHT>Balance</TD><TD></TD></TR>"
        f"{''.join(row(a) for a in member.accounts)}</TABLE>"
    )
    return page("Accounts", body)


def closed_page(account_id: str) -> str:
    return page("Closed", f"<P>ACCOUNT CLOSED: {escape(account_id)}</P>")


# --- opening a sub-account --------------------------------------------------------------------


def newsub_page(member: Member, tok: str, error: str | None = None, product: str = "01") -> str:
    n = member.number
    if member.status != "ACTIVE":
        return page("New sub-account", banner("ACCOUNT FROZEN - FUNCTION NOT PERMITTED"))
    err = banner(error) if error else ""
    options = "".join(
        f"<OPTION VALUE={code}{' SELECTED' if code == product else ''}>{name.title()}"
        for code, (name, _) in PRODUCTS.items()
    )
    body = (
        "<SCRIPT>function doSubmit(){"
        f'if(confirm("Open new sub-account for member {n}?")) document.forms[0].submit();'
        "}</SCRIPT>"
        f'{err}<FORM METHOD=POST ACTION="newsub.cgi?mid={n}">{hidden_token(tok)}'
        "<TABLE WIDTH=520 CELLPADDING=3 CELLSPACING=0>"
        '<TR><TD ALIGN=RIGHT WIDTH=140><FONT SIZE=1 COLOR="#666666">Product:</FONT></TD>'
        f"<TD><SELECT NAME=F7>{options}</SELECT></TD></TR>"
        '<TR><TD ALIGN=RIGHT><FONT SIZE=1 COLOR="#666666">Opening&nbsp;deposit:</FONT></TD>'
        "<TD><INPUT NAME=F8 SIZE=10></TD></TR>"
        '<TR><TD ALIGN=RIGHT><FONT SIZE=1 COLOR="#666666">Joint&nbsp;owner:</FONT></TD>'
        "<TD><INPUT TYPE=CHECKBOX NAME=F10 VALUE=Y></TD></TR>"
        '<TR><TD ALIGN=RIGHT><FONT SIZE=1 COLOR="#666666">Purpose:</FONT></TD>'
        "<TD><INPUT NAME=F9 SIZE=30></TD></TR>"
        '<TR><TD></TD><TD><INPUT TYPE=BUTTON VALUE="Submit" ONCLICK="doSubmit()"> '
        f'<INPUT TYPE=BUTTON VALUE="Cancel" ONCLICK="location=\'member.cgi?mid={n}\'">'
        "</TD></TR></TABLE></FORM>"
        '<FORM STYLE="display:none" METHOD=POST ACTION="print.cgi">'
        '<INPUT TYPE=SUBMIT VALUE="Submit"></FORM>'
    )
    return page("New sub-account", body)


def processing_page(ref: str) -> str:
    head = f'<META HTTP-EQUIV="refresh" CONTENT="2;url=/msv/done.cgi?ref={ref}">'
    body = "<P><B>PROCESSING - PLEASE WAIT</B></P><P>DO NOT CLICK BACK</P>"
    return page("Processing", body, head)


def done_page(ref: str, account_id: str, deposit_cents: int) -> str:
    body = (
        "<P><FONT SIZE=4>SUB-ACCOUNT OPENED</FONT></P>"
        "<TABLE CELLPADDING=3><TR><TD>"
        f"<TABLE CELLPADDING=2><TR><TD>Reference&nbsp;No:</TD><TD><B>{ref}</B></TD></TR>"
        f"<TR><TD>New&nbsp;Account:</TD><TD>{account_id}</TD></TR>"
        f"<TR><TD>Deposit:</TD><TD>{money(deposit_cents)}</TD></TR></TABLE>"
        "</TD></TR></TABLE>"
    )
    return page("Done", body)


def message_page(text: str) -> str:
    return page("Notice", banner(text))
