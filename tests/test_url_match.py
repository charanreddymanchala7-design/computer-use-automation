"""URL patterns for checkpoints and expectations: a path with `:param` segments and `*`."""

from __future__ import annotations

import pytest

from cua.replay.match import url_matches


@pytest.mark.parametrize(
    ("pattern", "url", "expected"),
    [
        ("/msv/frameset.cgi", "http://127.0.0.1:4310/msv/frameset.cgi", True),
        ("/msv/frameset.cgi", "/msv/frameset.cgi", True),
        ("/msv/frameset.cgi", "http://h/msv/frameset.cgi?x=1", True),  # the query is not the path
        ("/members/:id", "http://h/members/12345", True),
        ("/members/:id", "http://h/members/12345/edit", False),
        ("/members/:id", "http://h/members/", False),
        ("/members/:id/accounts", "http://h/members/9/accounts", True),
        ("/msv/*.cgi", "http://h/msv/search.cgi", True),
        ("/msv/*", "http://h/msv/a/b.cgi", True),
        ("/msv/frameset.cgi", "http://h/msv/frameset.cgi.bak", False),
        ("/msv/frameset.cgi", "http://h/other/msv/frameset.cgi", False),
        ("/a.b", "http://h/axb", False),  # a dot is a dot, not a wildcard
        ("/", "http://h/", True),
    ],
)
def test_a_pattern_matches_the_path_of_a_url(pattern: str, url: str, expected: bool) -> None:
    assert url_matches(pattern, url) is expected
