# Spikes: what was actually verified

Facts below were observed, not assumed, with Playwright 1.62.0 and Chromium 151 against the
MemberServ mock (a true `<frameset>` with a nested `<iframe>`), on 2026-09-25. Each one
changed a design decision.

## 1. True `<frameset>` documents

| Question | Observed | Consequence |
|---|---|---|
| Are `<frame>` documents visible to Playwright? | Yes. `page.frames` lists the top document (name `''`) and `hdr`, `nav`, `main` by their `NAME`. | Frames can be identified by name; the artifact's `FrameSelector` keeps `name`, `url_pattern` and `index` because real legacy frames are often unnamed. |
| Does the nested `<iframe>` appear? | Yes, as a child of `main` (`acct`, path `main > acct`). `parent_frame.name` is correct. | A locator recorded at the wrong depth silently finds nothing, so every locator bundle stores its full `frame_path`. |
| Does `frame_locator("frame[name=main]")` work on a `<frame>`? | Yes. | Replay can scope a locator to a frame with the standard API. |
| What is the top document's `<body>`? | `document.body` is the `FRAMESET` element. | There is no body to read text from; the top document contributes only a frame list. |

The top URL is `/msv/frameset.cgi` for the whole session. **A URL cannot be a checkpoint
here**; checkpoints must be frame-scoped text or element checks.

## 2. Accessibility snapshots stop at frame boundaries

- `aria_snapshot` of a frame that contains an iframe shows a bare `- iframe` and **none** of the
  iframe's content (the balance `$2,480.15` is absent from `main`'s snapshot and present in
  `acct`'s). Observe therefore walks every frame and snapshots each one separately.
- `frame.locator("body").aria_snapshot()` on the frameset document **hangs until it times out**
  (its body is the FRAMESET element). Observe detects a FRAMESET body and snapshots `html`
  instead, which yields `- document: - iframe - iframe - iframe`.

## 3. Snapshots leak typed values, including passwords

Found by a test, not by inspection. After typing into the password field the outline contained
`- textbox: demo-only` **and** `- cell "demo-only"` (a table cell's accessible name is built
from its content, including the input's value). Anything that reads the outline, including a
model, would see the secret. Mitigation, in three layers: textbox values are stripped from the
outline; the real value of every password field is masked wherever it appears; every secret the
caller registered is masked wherever it appears. Non-secret values are reported on the element.

A limit worth knowing: masking matches whole values, so a secret typed into a field with a
`maxlength` shorter than the secret would leave a truncated prefix visible. Secrets are only
ever typed into their own fields.

## 4. Coordinates

An in-page computation of an element's box in top-level viewport coordinates (adding each frame
element's offset up the chain) matched Playwright's `bounding_box()` to the pixel
(`[649, 220, 30, 13]` against `{x: 649.19, y: 220, width: 29.67, height: 13}`). That makes the
coordinate fallback safe for elements inside nested frames.

## 5. The synchronous API allows one Playwright per thread

A second `sync_playwright()` in the same thread fails with "using Playwright Sync API inside
the asyncio loop". Consequences:

- The agent loop and replay are single-threaded, and the surface is owned by that thread.
- A human takes over by using **the same headed window**; nothing else touches the browser.
- Operator surfaces (a terminal prompt, a small web page) run on other threads and only read
  and write thread-safe lease state.
- A "second client" is another process (`connect_over_cdp` to the loopback debug port), which
  is how a real one would attach. Verified in `tests/test_surface_lifecycle.py`.

## 6. Small facts that cost time

- Attribute values keep their source case (`TYPE=IMAGE`), so `type` is lowercased on the way out.
- `<input type=image>` counts as a submit button, so Enter in a field submits the form.
- A hidden decoy (`display:none`) has no box and is dropped from the offered elements.
- Playwright dismisses dialogs nobody handles, so a `confirm` would silently return false. The
  surface registers a handler on the context before any action and records every dialog.

## Still unverified

- What happens to a pending native dialog while a human holds control of the session
  (decided in task 2.3.2, and recorded here then).
- Behaviour of `<select>` popups under a screencast (not needed: the chosen handoff uses the
  real headed window).
