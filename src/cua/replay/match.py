"""URL patterns for expectations and checkpoints.

A pattern matches the *path* of a URL (the query and host are not part of it). `:name` matches
one path segment, so `/members/:id` covers every member, and `*` matches anything. Everything
else is literal, so a dot is a dot.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit


def _segment(segment: str) -> str:
    if segment.startswith(":") and len(segment) > 1:
        return "[^/]+"
    return re.escape(segment).replace(r"\*", ".*")


def url_matches(pattern: str, url: str) -> bool:
    path = urlsplit(url).path or "/"
    regex = "/".join(_segment(part) for part in pattern.split("/"))
    return re.fullmatch(regex, path) is not None
