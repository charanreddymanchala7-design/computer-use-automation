"""Frame-aware observation of a web page.

Why this walks frames instead of taking one snapshot: Playwright's accessibility snapshot of a
frame shows a child ``iframe`` as a bare ``- iframe`` and never inlines its content, and the
snapshot of a true ``<frameset>`` document's ``<body>`` (which is the FRAMESET element) does not
resolve at all. The only complete view of a legacy page is frame by frame (docs/spikes.md).

Interactive elements are found in the page with a DOM heuristic, not from the accessibility
tree, because in these apps the control is often just a clickable row, cell or image with no
role. Each one is given a short ref that stays valid until the next observation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import ElementHandle, Frame, Page
from playwright.sync_api import Error as PlaywrightError

from cua.surface.base import Box, ElementInfo, FrameHop, FrameObservation

TEXT_LIMIT = 4000
ARIA_LIMIT = 6000

# Normalised visible text of a frame: NBSP and runs of blanks collapse, blank lines are dropped.
TEXT_JS = r"""() => {
  const raw = document.body ? document.body.innerText : '';
  return raw.replace(/[ \t\u00a0]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();
}"""

# The text of one element (the value, for a field), normalised like a frame's text. A password
# field is never read.
ELEMENT_TEXT_JS = r"""(el) => {
  const type = (el.getAttribute('type') || '').toLowerCase();
  if (el.tagName === 'INPUT' && type === 'password') return '<masked>';
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return el.value || '';
  return (el.innerText || el.textContent || '')
    .replace(/[ \t\u00a0]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim();
}"""

# Visible interactive elements, in document order. A hidden decoy (display:none) is skipped
# because it has no box, and hidden inputs carry tokens rather than anything a person can use.
COLLECT_JS = r"""() => {
  const SEL = 'a[href], area[href], input, select, textarea, button, summary, [onclick], '
    + '[role=button], [role=link], [tabindex]';
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const st = getComputedStyle(el);
    return st.visibility !== 'hidden' && st.display !== 'none';
  };
  return Array.from(document.querySelectorAll(SEL)).filter(
    (el) => !(el.tagName === 'INPUT' && el.type === 'hidden') && visible(el)
  );
}"""

# The actual values of password fields in this frame, so they can be masked wherever they show up.
PASSWORD_VALUES_JS = r"""() => Array.from(document.querySelectorAll('input[type=password]'))
  .map((el) => el.value).filter((v) => v)"""

_TEXTBOX_VALUE = re.compile(r'^(\s*- (?:textbox|searchbox)(?: "[^"]*")?): .*$', re.MULTILINE)
_MIN_MASKED_LENGTH = 4  # shorter values would shred ordinary text

DESCRIBE_JS = r"""(els) => {
  const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const cut = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const inputRole = (el) => {
    const t = (el.getAttribute('type') || 'text').toLowerCase();
    if (['button', 'submit', 'reset', 'image'].includes(t)) return 'button';
    if (t === 'checkbox') return 'checkbox';
    if (t === 'radio') return 'radio';
    return 'textbox';
  };
  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a' || tag === 'area') return el.hasAttribute('href') ? 'link' : null;
    if (tag === 'button') return 'button';
    if (tag === 'input') return inputRole(el);
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    return null;
  };
  // Legacy pages have no <label>: the label is the neighbouring table cell or preceding text.
  const labelBefore = (el) => {
    const cell = el.closest('td,th');
    if (cell && cell.previousElementSibling) {
      const t = norm(cell.previousElementSibling.innerText);
      if (t) return t;
    }
    let text = '';
    let n = el.previousSibling;
    while (n && text.length < 60) {
      if (n.nodeType === 3) text = norm(n.textContent) + ' ' + text;
      else if (n.nodeType === 1 && !/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(n.tagName)) {
        text = norm(n.innerText) + ' ' + text;
      } else break;
      n = n.previousSibling;
    }
    return norm(text) || null;
  };
  const labelAfter = (el) => {
    const n = el.nextSibling;
    return n && n.nodeType === 3 ? norm(n.textContent) || null : null;
  };
  // Top-level viewport coordinates: add the offset of every frame element up the chain.
  const boxOf = (el) => {
    const r = el.getBoundingClientRect();
    let x = r.x, y = r.y, w = window;
    while (w !== w.top && w.frameElement) {
      const fr = w.frameElement.getBoundingClientRect();
      x += fr.x + w.frameElement.clientLeft;
      y += fr.y + w.frameElement.clientTop;
      w = w.parent;
    }
    return { x: x, y: y, width: r.width, height: r.height };
  };
  const textOf = (el) => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'select') {
      const o = el.options[el.selectedIndex];
      return o ? norm(o.text) : '';
    }
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (['button', 'submit', 'reset'].includes(t)) return norm(el.value);
      if (t === 'image') return norm(el.alt);
      return '';
    }
    if (tag === 'img') return norm(el.alt || el.title);
    return cut(norm(el.innerText || el.textContent), tag === 'tr' ? 160 : 100);
  };
  const attrsOf = (el) => {
    const out = {};
    for (const n of ['name', 'type', 'alt', 'title', 'placeholder', 'id', 'href']) {
      const v = el.getAttribute(n);
      if (v !== null && (v !== '' || n === 'alt')) out[n] = n === 'type' ? v.toLowerCase() : v;
    }
    const src = el.getAttribute('src');
    if (src) out.src = src.split('/').pop();
    const onclick = el.getAttribute('onclick');
    if (onclick) out.onclick = cut(onclick, 80);
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      // a password field's value is never reported, only that it has one
      if ((el.getAttribute('type') || '').toLowerCase() === 'password') {
        if (el.value) out.value = '<masked>';
      } else if (el.value) out.value = el.value;
    }
    return out;
  };
  return els.map((el) => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    return {
      tag: tag,
      role: roleOf(el),
      text: textOf(el),
      attrs: attrsOf(el),
      label_hint: ['input', 'select', 'textarea'].includes(tag) ? labelBefore(el) : null,
      label_after: ['radio', 'checkbox'].includes(type) ? labelAfter(el) : null,
      box: boxOf(el),
      checked: ['radio', 'checkbox'].includes(type) ? !!el.checked : null,
      disabled: !!el.disabled,
      options: tag === 'select' ? Array.from(el.options).slice(0, 20).map((o) => norm(o.text)) : [],
    };
  });
}"""


def _make_info(ref: str, frame_index: int, fact: dict[str, Any]) -> ElementInfo:
    box = fact["box"]
    return ElementInfo(
        ref=ref,
        frame=frame_index,
        tag=fact["tag"],
        role=fact["role"],
        text=fact["text"],
        attrs=dict(fact["attrs"]),
        label_hint=fact["label_hint"],
        label_after=fact["label_after"],
        box=Box(**box) if box else None,
        checked=fact["checked"],
        disabled=fact["disabled"],
        options=list(fact["options"]),
    )


@dataclass
class RefTable:
    """Refs issued by one observation, with the element handles they stand for."""

    handles: dict[str, ElementHandle] = field(default_factory=dict)
    infos: dict[str, ElementInfo] = field(default_factory=dict)
    _next: int = 1

    def issue(self, handle: ElementHandle, frame_index: int, fact: dict[str, Any]) -> ElementInfo:
        ref = f"e{self._next}"
        self._next += 1
        info = _make_info(ref, frame_index, fact)
        self.handles[ref] = handle
        self.infos[ref] = info
        return info


def _mask(text: str, values: Iterable[str]) -> str:
    for value in sorted({v for v in values if len(v) >= _MIN_MASKED_LENGTH}, key=len, reverse=True):
        text = text.replace(value, "<masked>")
    return text


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _aria(frame: Frame, body_tag: str) -> str:
    if not body_tag:
        return ""
    # A frameset's <body> is the FRAMESET element and its snapshot never resolves; the document
    # outline (html) does, and lists its child frames.
    target = "html" if body_tag == "FRAMESET" else "body"
    try:
        return _cut(frame.locator(target).aria_snapshot(timeout=2000), ARIA_LIMIT)
    except PlaywrightError:
        return ""


def _elements(frame: Frame, index: int, refs: RefTable) -> list[ElementInfo]:
    collected = frame.evaluate_handle(COLLECT_JS)
    numbered = sorted(
        ((int(key), prop) for key, prop in collected.get_properties().items() if key.isdigit()),
        key=lambda pair: pair[0],
    )
    handles = [prop.as_element() for _, prop in numbered]
    facts: list[dict[str, Any]] = frame.evaluate(DESCRIBE_JS, collected)
    return [
        refs.issue(handle, index, fact)
        for handle, fact in zip(handles, facts, strict=False)
        if handle is not None
    ]


def describe_handle(handle: ElementHandle, refs: RefTable, secrets: Iterable[str]) -> ElementInfo:
    """Give an element found some other way (a resolved locator) a ref and the same facts an
    observed element has, so it can be acted on and risk-classified identically."""
    frame = handle.owner_frame()
    if frame is None:
        raise PlaywrightError("the element is no longer attached to a frame")
    facts: list[dict[str, Any]] = frame.evaluate(DESCRIBE_JS, [handle])
    info = refs.issue(handle, -1, facts[0])  # -1: not part of any observation's frame list
    hidden = [*secrets, *frame.evaluate(PASSWORD_VALUES_JS)]
    info.text = _mask(info.text, hidden)
    info.attrs = {k: _mask(v, hidden) for k, v in info.attrs.items()}
    return info


def mask_text(text: str, secrets: Iterable[str]) -> str:
    return _mask(text, secrets)


def _observe_frame(
    frame: Frame,
    index: int,
    path: tuple[FrameHop, ...],
    refs: RefTable,
    secrets: tuple[str, ...],
) -> FrameObservation:
    text, aria, elements, unreadable = "", "", [], False
    try:
        hidden = [*secrets, *frame.evaluate(PASSWORD_VALUES_JS)]
        text = _mask(_cut(frame.evaluate(TEXT_JS), TEXT_LIMIT), hidden)
        outline = _aria(frame, frame.evaluate("document.body ? document.body.tagName : ''"))
        # An outline reports what is typed into a textbox (a password field's real value
        # included), so values are dropped: they belong on the element, where they are masked.
        aria = _mask(_TEXTBOX_VALUE.sub(r"\1", outline), hidden)
        elements = _elements(frame, index, refs)
        for element in elements:
            element.text = _mask(element.text, hidden)
            element.attrs = {k: _mask(v, hidden) for k, v in element.attrs.items()}
    except PlaywrightError:
        unreadable = True  # the frame navigated while we were reading it
    return FrameObservation(
        index=index,
        path=path,
        name=frame.name or None,
        url=frame.url,
        text=text,
        aria=aria,
        elements=elements,
        unreadable=unreadable,
    )


def observe_frames(
    page: Page, refs: RefTable, secrets: Iterable[str] = ()
) -> list[FrameObservation]:
    """Every frame of the page in document order, each with its own elements and refs."""
    hidden = tuple(secrets)
    frames: list[FrameObservation] = []

    def visit(frame: Frame, path: tuple[FrameHop, ...]) -> None:
        frames.append(_observe_frame(frame, len(frames), path, refs, hidden))
        live = [child for child in frame.child_frames if not child.is_detached()]
        for position, child in enumerate(live):
            hop = FrameHop(child.name or None, position, child.url)
            visit(child, (*path, hop))

    visit(page.main_frame, ())
    return frames
