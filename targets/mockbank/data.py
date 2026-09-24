"""Synthetic seed data for MemberServ. Everything here is invalid or reserved by construction:
member numbers are made up, phones are in the fictional 555-01xx range, emails end in the
reserved .test TLD, and the SSN-shaped values start with area 000, which is never issued.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Account:
    id: str
    kind: str
    balance_cents: int


@dataclass
class Member:
    number: str
    name: str
    status: str
    branch: str
    since: str
    tin: str
    phone: str
    email: str
    accounts: list[Account] = field(default_factory=list)


def money(cents: int) -> str:
    """Legacy formatting: $2,480.15, and a negative shown as (12.00) with no dollar sign."""
    dollars, remainder = divmod(abs(cents), 100)
    text = f"{dollars:,}.{remainder:02d}"
    return f"({text})" if cents < 0 else f"${text}"


def seed_members() -> dict[str, Member]:
    members = [
        Member(
            "12345",
            "TESTERSON, ADA",
            "ACTIVE",
            "MAIN ST",
            "01/01/2001",
            "000-00-0001",
            "555-0101",
            "ada.testerson@example.test",
            [
                Account("SYN-12345-S01", "SHARE SAVINGS", 248015),
                Account("SYN-12345-C01", "CHECKING", 31240),
                Account("SYN-12345-L01", "OVERDRAFT LINE", -1200),
            ],
        ),
        Member(
            "12346",
            "PLACEHOLDER, GRACE",
            "FROZEN",
            "MAIN ST",
            "01/01/2003",
            "000-00-0002",
            "555-0102",
            "grace.placeholder@example.test",
            [Account("SYN-12346-S01", "SHARE SAVINGS", 10000)],
        ),
        Member(
            "12347",
            "SAMPLE, JORDAN",
            "ACTIVE",
            "ELM ST",
            "01/01/2005",
            "000-00-0003",
            "555-0103",
            "jordan.sample1@example.test",
            [Account("SYN-12347-S01", "SHARE SAVINGS", 5000)],
        ),
        Member(
            "12348",
            "SAMPLE, JORDAN",
            "ACTIVE",
            "ELM ST",
            "01/01/2007",
            "000-00-0004",
            "555-0104",
            "jordan.sample2@example.test",
            [Account("SYN-12348-S01", "SHARE SAVINGS", 7500)],
        ),
        Member(
            "12349",
            "O'NEIL-SAMPLE, ZOÉ",
            "ACTIVE",
            "ELM ST",
            "01/01/2009",
            "000-00-0005",
            "555-0105",
            "zoe.oneil@example.test",
            [Account("SYN-12349-S01", "SHARE SAVINGS", 10000)],
        ),
    ]
    return {m.number: m for m in members}


PRODUCTS = {"01": ("SHARE SAVINGS", "S"), "02": ("HOLIDAY CLUB", "H")}
MINIMUM_DEPOSIT_CENTS = 500
BUSINESS_DATE = "09/25/2026"
