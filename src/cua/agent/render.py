"""What the model is shown: a compact description of every frame, its text and its controls.

Each control gets a short ref (``e12``) that the model uses to act on it. Legacy pages give
controls no roles or labels, so the description says what it can: the role if there is one,
"(no role, clickable)" if not, the neighbouring-cell label, and the attributes that identify it.
Volatile ids are left out on purpose: they change on every render and would only mislead.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from cua.surface import ElementInfo, FrameObservation, Observation

_ATTR_ORDER = ("name", "type", "value", "alt", "title", "placeholder", "src", "href", "onclick")
_QUOTED_ATTRS = frozenset({"value"})
_ATTR_LIMIT = 70


def _path(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _element_line(element: ElementInfo) -> str:
    parts = [element.ref]
    if element.role:
        parts.append(element.role)
    else:
        parts.append(f"{element.tag} (no role, clickable)")
    if element.text:
        parts.append(f'"{element.text}"')
    if element.label_hint:
        parts.append(f'label="{element.label_hint}"')
    if element.label_after:
        parts.append(f'after="{element.label_after}"')
    for key in _ATTR_ORDER:
        if key in element.attrs:
            value = _cut(element.attrs[key], _ATTR_LIMIT)
            parts.append(f'{key}="{value}"' if key in _QUOTED_ATTRS else f"{key}={value}")
    if element.options:
        parts.append(f"options={json.dumps(element.options)}")
    if element.checked is not None:
        parts.append("checked" if element.checked else "unchecked")
    if element.disabled:
        parts.append("disabled")
    return "  " + " ".join(parts)


def _frame_header(frame: FrameObservation) -> str:
    header = f"[frame {frame.index}"
    if frame.name:
        header += f' "{frame.name}"'
    if len(frame.path) > 1:
        header += " path " + " > ".join(hop.name or f"#{hop.index}" for hop in frame.path)
    return header + f" {_path(frame.url)}]"


def _render_frame(frame: FrameObservation, *, is_frameset: bool, max_text: int) -> list[str]:
    if is_frameset:
        return [f"[frame {frame.index}] (a frameset: its content is in the frames below)"]
    header = _frame_header(frame)
    if frame.unreadable:
        return [header, "  (this frame changed while it was being read: observe again)"]
    if not frame.text and not frame.elements:
        return [f"{header} (empty)"]
    lines = [header]
    if frame.text:
        shown = frame.text[:max_text]
        suffix = " … (truncated)" if len(frame.text) > max_text else ""
        lines.append("  text: " + shown.replace("\n", " | ") + suffix)
    if frame.elements:
        lines.append("  elements:")
        lines.extend(_element_line(e) for e in frame.elements)
    return lines


def render_observation(obs: Observation, *, max_text: int = 1500) -> str:
    lines = [f"URL: {_path(obs.url)}  |  title: {obs.title}"]
    for frame in obs.frames:
        is_frameset = (
            frame.index == 0
            and len(obs.frames) > 1
            and not frame.text
            and not frame.elements
            and not frame.unreadable
        )
        lines.extend(_render_frame(frame, is_frameset=is_frameset, max_text=max_text))
    if obs.dialogs:
        handled = "; ".join(
            f'{d.kind} "{d.message}" -> {"accepted" if d.accepted else "dismissed"}'
            for d in obs.dialogs
        )
        lines.append(f"Dialogs handled: {handled}")
    return "\n".join(lines)
