"""Recording what a person does in the live session, without recording what they type.

A small script is installed in every frame while a person holds control. It reports trusted
clicks, keystrokes-as-a-count, selections and Enter presses to the surface through a binding.
Three rules keep regulated data out of the record:

* typed text is never sent, only that something was typed and how many characters (and for a
  password field not even that);
* a control is named by its own label, but a data cell, row or paragraph is not named at all,
  because its text is somebody's record;
* only *trusted* events count, so the page's own scripts cannot forge or flood the record.
"""

from __future__ import annotations

from typing import Any

from cua.surface.base import HumanAction

# Raw string: the JavaScript keeps its own backslashes.
CAPTURE_JS = r"""
(() => {
  if (window.__cuaCapture) return;
  window.__cuaCapture = true;
  const LIMIT = 60;
  const CONTROLS = 'a,button,input,select,textarea,label,img,summary,[role=button],[role=link]';
  const clip = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, LIMIT);
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clip(aria);
    if (el.labels && el.labels.length) return clip(el.labels[0].innerText);
    const alt = el.getAttribute('alt') || el.getAttribute('title');
    if (alt) return clip(alt);
    const tag = el.tagName;
    if (tag === 'BUTTON' || tag === 'A') return clip(el.innerText);
    if (tag === 'INPUT' && ['submit', 'button'].includes((el.type || '').toLowerCase())) {
      return clip(el.value);
    }
    return el.getAttribute('name') ? 'name=' + el.getAttribute('name') : '';
  };
  const describe = (raw) => {
    const el = raw.closest ? (raw.closest(CONTROLS) || raw) : raw;
    const isControl = el.matches && el.matches(CONTROLS);
    return {
      tag: el.tagName.toLowerCase(),
      type: (el.getAttribute && el.getAttribute('type') || '').toLowerCase(),
      name: isControl ? nameOf(el) : '',
    };
  };
  const send = (payload) => { try { window.__cua_capture(payload); } catch (e) {} };
  const opts = { capture: true };
  document.addEventListener('click', (ev) => {
    if (ev.isTrusted) send({ kind: 'click', ...describe(ev.target) });
  }, opts);
  document.addEventListener('input', (ev) => {
    if (ev.isTrusted && ev.target) ev.target.__cuaDirty = true;
  }, opts);
  const commit = (ev) => {
    const el = ev.target;
    if (!ev.isTrusted || !el || !el.tagName) return;
    if (el.tagName === 'SELECT') { send({ kind: 'selected', ...describe(el) }); return; }
    const type = (el.type || '').toLowerCase();
    if (type === 'checkbox' || type === 'radio') {
      send({ kind: 'toggled', ...describe(el) });
      return;
    }
    if (!el.__cuaDirty) return;
    el.__cuaDirty = false;
    const chars = type === 'password' ? null : (el.value || '').length;
    send({ kind: 'typed', ...describe(el), chars });
  };
  document.addEventListener('change', commit, opts);
  document.addEventListener('focusout', commit, opts);
  document.addEventListener('keydown', (ev) => {
    if (ev.isTrusted && ev.key === 'Enter') {
      send({ kind: 'pressed', tag: '', type: '', name: 'Enter' });
    }
  }, opts);
})();
"""

_INPUT_KINDS = {
    "": "text field",
    "text": "text field",
    "password": "password field",
    "checkbox": "checkbox",
    "radio": "radio button",
    "submit": "button",
    "button": "button",
    "image": "image button",
}
_TAG_KINDS = {"a": "link", "button": "button", "select": "dropdown", "textarea": "text area"}


def _target(tag: str, type_: str, name: str) -> str:
    kind = _INPUT_KINDS.get(type_, f"{type_} field") if tag == "input" else _TAG_KINDS.get(tag, tag)
    return f"{kind} {name!r}" if name else kind


def action_from_payload(payload: Any, frame: str) -> HumanAction | None:
    """Turn what the page reported into an action, or ``None`` for anything malformed."""
    if not isinstance(payload, dict):
        return None
    kind = payload.get("kind")
    if kind not in ("click", "typed", "selected", "toggled", "pressed"):
        return None
    tag, type_, name = (str(payload.get(k) or "") for k in ("tag", "type", "name"))
    chars = payload.get("chars")
    return HumanAction(
        kind,
        name if kind == "pressed" else _target(tag, type_, name),
        frame,
        chars if isinstance(chars, int) and not isinstance(chars, bool) else None,
    )
