# Submitting

The brief: push to a **public** GitHub repo and email the link to **assignments@interface.ai**, with the
repo URL on its own line, from the address used to apply, and no zip.

## Before making the repo public

```bash
scripts/prepublish_check.sh          # history secrets scan, evidence check, deliverables, fresh clone
```

Then open the PNGs under `evidence/` once. Screenshots cannot be grepped, and they are not
pixel-redacted (the mock's data is synthetic; the sign-in name shows in the login screenshot).

Make it public (this is the step that cannot be undone quietly, so it is yours to run):

```bash
gh repo edit charanreddymanchala7-design/computer-use-automation --visibility public --accept-visibility-change-consequences
```

## The email

Send it yourself, from the address you applied with. A draft:

> Subject: Take-home submission: Computer-Use Automation System
>
> Hi,
>
> Here is my submission for the Computer-Use Automation System take-home.
>
> https://github.com/charanreddymanchala7-design/computer-use-automation
>
> `README.md` has setup and the demo path, `REPORT.md` the design write-up, and `evidence/` the real
> discovery run and the replay, error, handoff and agent-call runs. The project was built with Claude
> Code assistance, which the README says too.
>
> Thanks,
> Charan

## Being ready to defend it

The brief says you own everything you submit. The three places I would expect questions, and where the
answer lives:

- **Why is the artifact shaped this way?** `REPORT.md`, "Artifact schema", and `src/cua/artifact/schema.py`
  (its module docstring lists the rules the schema enforces).
- **How does a run stop being safe to continue, and how does control move?** `REPORT.md`, "Escalation &
  handoff", `src/cua/control/lease.py` and `src/cua/control/handoff.py`.
- **What is not built?** `REPORT.md`, "Cuts". Say so plainly; the brief asks for it.
