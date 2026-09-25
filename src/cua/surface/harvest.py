"""Locator harvesting and resolution for web pages.

Harvesting turns an element seen at discovery time into a *ranked bundle* of ways to find it
again. The rule that makes a bundle trustworthy: a candidate strategy is kept only if, resolved
against the live page right now, it finds exactly one element and that element is the one it was
built from. Nothing unverified reaches an artifact.

Ranking runs from the most meaningful to the most brittle: accessible role and name, label,
the neighbouring label text in the same row, stable attributes, visible text, a positional path
and, only when nothing else verified, a screen coordinate. Volatile ids are never used.

Resolution is the inverse and is what deterministic replay runs: strategies are tried in order,
the first that finds exactly one element wins, and an ambiguous strategy is skipped, never guessed.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterator, Mapping, Sequence
from urllib.parse import urlsplit

from playwright.sync_api import ElementHandle, Frame, Locator
from playwright.sync_api import Error as PlaywrightError

from cua.artifact import (
    AncestorAnchorLocator,
    AttributeFingerprintLocator,
    CoordinatesLocator,
    CssLocator,
    FrameSelector,
    LabelLocator,
    LocatorBundle,
    LocatorStrategy,
    RoleNameLocator,
    TextLocator,
    XPathLocator,
    fill_placeholders,
    parameterize,
)
from cua.surface.base import ElementInfo, HarvestError

_MIN_PARAM_LENGTH = 3  # see cua.artifact.parameterize
_NAMED_ROLES = frozenset({"link", "button", "checkbox", "radio", "tab", "menuitem", "option"})
_FIELD_TAGS = frozenset({"input", "select", "textarea"})
_TEXT_TAGS = frozenset({"a", "td", "th", "tr", "button", "span", "font", "b", "p", "div", "li"})
_STABLE_ATTRS = ("name", "type", "alt", "title", "src", "href", "placeholder")
_VALUE_TYPES = frozenset({"button", "submit", "reset", "radio", "checkbox"})

# the innermost element of each kind, so a nested layout table does not match its own rows
_CONTAINERS = {
    "row": "tr:not(:has(tr))",
    "cell": "td:not(:has(td)), th:not(:has(th))",
    "form": "form",
    "fieldset": "fieldset",
    "frame_body": "body",
    "any": "body",
}

_NTH_IN_ROW_JS = """(el) => {
  const row = el.closest('tr');
  if (!row) return null;
  const index = Array.from(row.querySelectorAll(el.tagName)).indexOf(el);
  return index < 0 ? null : index;
}"""

_POSITIONAL_CSS_JS = """(el) => {
  const parts = [];
  let node = el;
  while (node && node.nodeType === 1 && node !== document.documentElement) {
    const tag = node.tagName.toLowerCase();
    if (tag === 'body') { parts.unshift('body'); break; }
    if (!node.parentElement) return null;
    const index = Array.from(node.parentElement.children).indexOf(node) + 1;
    parts.unshift(tag + ':nth-child(' + index + ')');
    node = node.parentElement;
  }
  return parts.join(' > ');
}"""

_SAME_ELEMENT_JS = (
    "(found, [target, inside]) => found === target || (inside && target.contains(found))"
)

FIND_VALUE_JS = r"""([value, anchor]) => {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const want = norm(value);
  const anchorText = anchor ? norm(anchor).toLowerCase() : null;
  const shows = (el) => norm(el.innerText || el.textContent) === want;
  return Array.from(document.querySelectorAll('body *')).filter((el) => {
    if (!shows(el) || Array.from(el.children).some(shows)) return false;  // innermost only
    if (!anchorText) return true;
    const row = el.closest('tr');
    return !!row && norm(row.innerText || row.textContent).toLowerCase().includes(anchorText);
  });
}"""


# --- frames --------------------------------------------------------------------------------


def live_children(frame: Frame) -> list[Frame]:
    """Child frames still attached. After a frameset reloads, the client can keep listing the
    old, detached frames beside the new ones, and matching by name would pick a dead one."""
    return [child for child in frame.child_frames if not child.is_detached()]


def frame_path_of(frame: Frame) -> list[FrameSelector]:
    """The frame's address from the top window: by name where it has one, else by position."""
    hops: list[FrameSelector] = []
    node = frame
    while node.parent_frame is not None:
        parent = node.parent_frame
        hops.append(
            FrameSelector(name=node.name)
            if node.name
            else FrameSelector(index=live_children(parent).index(node))
        )
        node = parent
    return list(reversed(hops))


def find_frame(top: Frame, path: Sequence[FrameSelector]) -> Frame | None:
    frame = top
    for hop in path:
        match: Frame | None = None
        for position, child in enumerate(live_children(frame)):
            if hop.name is not None and child.name != hop.name:
                continue
            if hop.index is not None and hop.index != position:
                continue
            if hop.url_pattern is not None and not fnmatch.fnmatch(
                urlsplit(child.url).path, hop.url_pattern
            ):
                continue
            match = child
            break
        if match is None:
            return None
        frame = match
    return frame


# --- strategies -> locators ------------------------------------------------------------------


def _css_attr(name: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    operator = "$=" if name == "src" else "="  # src is stored as a basename
    return f'[{name}{operator}"{escaped}"]'


def _candidate_locator(
    frame: Frame, strategy: LocatorStrategy, values: Mapping[str, str]
) -> tuple[Locator | None, str]:
    """A locator for the strategy, or (None, outcome) when it cannot even be formed."""

    def text(raw: str) -> str:
        return fill_placeholders(raw, values)

    if isinstance(strategy, RoleNameLocator):
        return (
            frame.get_by_role(strategy.role, name=text(strategy.name), exact=strategy.exact),  # type: ignore[arg-type]
            "",
        )
    if isinstance(strategy, LabelLocator):
        return frame.get_by_label(text(strategy.text), exact=strategy.exact), ""
    if isinstance(strategy, TextLocator):
        return frame.get_by_text(text(strategy.text), exact=strategy.exact), ""
    if isinstance(strategy, AncestorAnchorLocator):
        containers = frame.locator(
            _CONTAINERS[strategy.container], has_text=text(strategy.anchor_text)
        )
        found = containers.count()
        if found != 1:
            return None, "no_match" if found == 0 else "ambiguous"
        inner = containers.first
        target = (
            inner.get_by_role(strategy.target_role)  # type: ignore[arg-type]
            if strategy.target_role
            else inner.locator(strategy.target_tag or "*")
        )
        return target.nth(strategy.nth), ""
    if isinstance(strategy, AttributeFingerprintLocator):
        attrs = "".join(_css_attr(k, text(v)) for k, v in strategy.attributes.items())
        return frame.locator(strategy.tag + attrs), ""
    if isinstance(strategy, CssLocator):
        return frame.locator(strategy.selector), ""
    if isinstance(strategy, XPathLocator):
        return frame.locator(f"xpath={strategy.expression}"), ""
    return None, "no_match"  # a coordinate is a point, not an element


def single_match(
    frame: Frame, strategy: LocatorStrategy, values: Mapping[str, str]
) -> tuple[ElementHandle | None, str]:
    """The one element a strategy finds, or why it did not: no_match or ambiguous."""
    locator, note = _candidate_locator(frame, strategy, values)
    if locator is None:
        return None, note
    try:
        count = locator.count()
        if count != 1:
            return None, "no_match" if count == 0 else "ambiguous"
        return locator.element_handle(timeout=1000), "matched"
    except PlaywrightError:
        return None, "no_match"


# --- harvesting -------------------------------------------------------------------------------


def _name_of(info: ElementInfo) -> str:
    return info.text or info.attrs.get("alt", "") or info.attrs.get("title", "")


def _fingerprint(info: ElementInfo, params: Mapping[str, str]) -> dict[str, str]:
    attrs = {k: info.attrs[k] for k in _STABLE_ATTRS if k in info.attrs}
    # what a person typed into a field is never part of what identifies it
    if info.tag == "input" and info.attrs.get("type") in _VALUE_TYPES and "value" in info.attrs:
        attrs["value"] = info.attrs["value"]
    return {k: parameterize(v, params) for k, v in attrs.items()}


def _candidates(
    handle: ElementHandle, info: ElementInfo, params: Mapping[str, str]
) -> Iterator[LocatorStrategy]:
    name = _name_of(info)
    if info.role in _NAMED_ROLES and name:
        yield RoleNameLocator(
            role=info.role,
            name=parameterize(name, params),
            exact=True,
            rationale="Accessible role and name: survives markup and layout changes while the "
            "visible name is unchanged",
        )
    if info.label_hint and info.tag in _FIELD_TAGS:
        yield LabelLocator(
            text=parameterize(info.label_hint, params),
            exact=True,
            rationale="A real <label> for the field; kept only where the page has one",
        )
        nth = handle.evaluate(_NTH_IN_ROW_JS)
        if nth is not None:
            yield AncestorAnchorLocator(
                anchor_text=parameterize(info.label_hint, params),
                container="row",
                target_tag=info.tag,
                nth=nth,
                rationale="Anchored on the neighbouring label text in the same row: survives "
                "volatile ids and added rows or columns elsewhere",
            )
    fingerprint = _fingerprint(info, params)
    if fingerprint:
        yield AttributeFingerprintLocator(
            tag=info.tag,
            attributes=fingerprint,
            rationale="Tag plus stable attributes, never the volatile id: survives layout "
            "changes, breaks only if the attributes are renamed",
        )
    if info.tag in _TEXT_TAGS and name:
        used = [n for n, v in params.items() if len(v) >= _MIN_PARAM_LENGTH and v in name]
        for param in used:
            yield TextLocator(
                text="{" + param + "}",
                exact=False,
                rationale="The text the caller searched for, so the same capability finds the "
                "row for any value",
            )
        if not used:
            yield TextLocator(
                text=name,
                exact=True,
                rationale="Visible text: readable, but breaks if the wording changes or a "
                "tenant relabels it",
            )


def _positional_css(handle: ElementHandle) -> CssLocator | None:
    path = handle.evaluate(_POSITIONAL_CSS_JS)
    if not path:
        return None
    return CssLocator(
        selector=path,
        rationale="Positional path: the last DOM-based resort, breaks when the layout changes",
    )


def _is_the_element(
    frame: Frame,
    strategy: LocatorStrategy,
    values: Mapping[str, str],
    target: ElementHandle,
    *,
    inside_ok: bool,
) -> bool:
    found, _ = single_match(frame, strategy, values)
    if found is None:
        return False
    return bool(found.evaluate(_SAME_ELEMENT_JS, [target, inside_ok]))


def _describe(info: ElementInfo, params: Mapping[str, str] | None = None) -> str:
    """A label for a person reviewing the artifact. An element found by what the caller supplied
    (a result row) is data-dependent: the rest of its text belongs to whichever record it was
    when discovered, so only the placeholder is kept."""
    label = info.text or info.label_hint or info.attrs.get("name", "")
    generic = parameterize(label, params or {})
    if generic != label:
        names = " ".join(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", generic))
        return f"{info.tag} containing {names}"
    return f"{info.tag} {label[:50]!r}".strip()


def harvest_element(
    handle: ElementHandle,
    info: ElementInfo,
    params: Mapping[str, str],
    *,
    for_click: bool,
    viewport: tuple[int, int],
) -> LocatorBundle:
    frame = handle.owner_frame()
    if frame is None:
        raise HarvestError("the element is no longer attached to a frame; observe again")
    values = dict(params)
    kept: list[LocatorStrategy] = [
        strategy
        for strategy in _candidates(handle, info, params)
        # a click on a descendant bubbles to the target, so a click may resolve to a child
        if _is_the_element(frame, strategy, values, handle, inside_ok=for_click)
    ]
    if not kept:
        positional = _positional_css(handle)
        if positional and _is_the_element(frame, positional, values, handle, inside_ok=for_click):
            kept.append(positional)
    if not kept and info.box is not None:
        x, y = info.box.center
        kept.append(
            CoordinatesLocator(
                x=x,
                y=y,
                viewport_width=viewport[0],
                viewport_height=viewport[1],
                rationale="No DOM anchor could be verified: last resort, tied to this viewport",
            )
        )
    if not kept:
        raise HarvestError(f"no reliable locator for {_describe(info)}: try a different element")
    return LocatorBundle(
        description=_describe(info, params), frame_path=frame_path_of(frame), strategies=kept
    )


def harvest_value(
    frames: Sequence[Frame],
    value: str,
    anchor_text: str | None,
    params: Mapping[str, str],
) -> LocatorBundle:
    """Locate a piece of text through the row that labels it (values to be read back on replay)."""
    found: list[tuple[Frame, ElementHandle]] = []
    for frame in frames:
        try:
            collected = frame.evaluate_handle(FIND_VALUE_JS, [value, anchor_text])
            for key, prop in collected.get_properties().items():
                element = prop.as_element() if key.isdigit() else None
                if element is not None:
                    found.append((frame, element))
        except PlaywrightError:
            continue
    if not found:
        raise HarvestError(
            f"{value!r} not found on the page: quote it exactly as it appears"
            + (f" in the row containing {anchor_text!r}" if anchor_text else "")
        )
    if len(found) > 1:
        raise HarvestError(
            f"{value!r} appears in more than one place: give anchor_text, the text of the row "
            "that labels the value you mean"
        )
    frame, handle = found[0]
    strategies: list[LocatorStrategy] = []
    if anchor_text:
        tag = str(handle.evaluate("(el) => el.tagName.toLowerCase()"))
        nth = handle.evaluate(_NTH_IN_ROW_JS)
        if nth is not None:
            anchored = AncestorAnchorLocator(
                anchor_text=parameterize(anchor_text, params),
                container="row",
                target_tag=tag,
                nth=nth,
                rationale="The row labelled by this text holds the value: survives added rows "
                "and volatile ids, and finds the value whatever it is on a later run",
            )
            if _is_the_element(frame, anchored, dict(params), handle, inside_ok=False):
                strategies.append(anchored)
    if not strategies:
        positional = _positional_css(handle)
        if positional and _is_the_element(frame, positional, dict(params), handle, inside_ok=False):
            strategies.append(positional)
    if not strategies:
        raise HarvestError(f"could not build a reliable locator for {value!r}")
    return LocatorBundle(
        description=f"value near {anchor_text!r}" if anchor_text else "value",
        frame_path=frame_path_of(frame),
        strategies=strategies,
    )
