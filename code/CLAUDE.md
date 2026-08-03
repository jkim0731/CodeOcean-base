# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this capsule is

This is a Code Ocean **capsule** — an interactive Ubuntu desktop workstation, not a conventional
application repo. It exists to orchestrate Code Ocean pipelines: searching/attaching data assets,
running pipeline monitors, and capturing results as derived data assets (see `README.md`). Current
use cases: 2p-3DFISH autocoregistration, pupil tracking with eyefoam detection.

There is no build/lint/test pipeline here — "development" means writing and running ad-hoc Python
(Jupyter, VS Code, or a shell) against the Code Ocean API.

## Code Ocean orchestration

@base-CLAUDE.md

Read it before writing any Code Ocean orchestration code. It covers auth/client setup, the capsule
registry (`code/CO_capsule_infos.xlsx` — source of truth for capsule ids/suffixes), data-asset
search, the attach+run+capture flow via the all-users pipeline monitor, docDB schema files, and a
CAUTIONS list (controlled-vocab `experiment type`, never `rglob` over `/data`, `aind-data-schema`
version pinning, root-owned `.git` under `/`, etc) that has cost real debugging time before.

## Capsule layout

- `code/` — world-writable working area for scripts/notebooks, plus the registry
  (`CO_capsule_infos.xlsx`) and `base-CLAUDE.md`.
- `data/` — read-only data assets attached to a run (empty until something is attached).
- `results/` — output of a **reproducible** run; captured as a derived data asset. **In this
  interactive workstation session it is ephemeral** — anything written here during ad-hoc
  Jupyter/script work is NOT persisted and can silently disappear (has happened before). For
  interactive output that needs to survive, write to `/scratch/<folder>` instead and, if it should
  become a data asset, capture it explicitly (see `base-CLAUDE.md` §4, "Interactive capture").
- `scratch/` (→ `/scratch`) — persistent interactive scratch space. Use this (`/scratch/tmp`) for
  temp files, never `/tmp` or `/` — the root overlay is small and filling it can lock up the env.
- `environment/` — capsule image definition: `Dockerfile`, `postInstall`, `start.sh` (runtime
  permission fixups + core-dump prevention), `update_firefox`, `vscode_setting.sh`.
- `.codeocean/` — resource class (`resources.json`) and secret *names* (`secrets.json`, e.g.
  `API_SECRET`, AWS role creds, `CLAUDE_CODE_OAUTH_TOKEN`) — never print/echo their values.
- `metadata/metadata.yml` — capsule name/description/author shown in the Code Ocean UI.

## Environment notes

- Base image is rebuilt onto Python 3.11 (`mamba install python=3.11`). Key installed packages:
  `codeocean==0.14.0`, `aind-codeocean-pipeline-monitor`, `aind-codeocean-utils`,
  `aind-data-access-api`, `aind-data-schema==1.2.0`, `aind_session`, `aind-log-utils`.
- `postInstall` additionally editable-installs two GitHub repos to `/`: `comb` (branch
  `for_gcamp_validation`) and `lamf-analysis` — the latter provides `code_ocean_utils`,
  `zstack_utils`, and `json_utils`, used throughout `code/base-CLAUDE.md`.


## Local Tools / Skills

Code Ocean orchestration skills (running/capturing computations, data assets, app panels, git
sync, …) come from the shared repo **`claude-code-skills-codeocean`**
(<https://github.com/jkim0731/claude-code-skills-codeocean>), which `environment/postInstall`
clones to **`/claude-code-skills-codeocean`**. Treat that clone as the source of truth: its
`README.md` lists the current skills, and the set changes over time — read it (or `ls` the clone)
rather than relying on a list here.

Claude Code only auto-loads skills from `<capsule>/.claude/skills/`, so each `codeocean-*` folder
in the clone must be **symlinked** there (README layout A). They are linked already; if the
directory is missing or a link dangles — e.g. after an env rebuild or a fresh checkout — recreate
them and restart Claude Code so discovery re-runs:

```bash
S=/claude-code-skills-codeocean; mkdir -p /root/capsule/.claude/skills
for d in "$S"/codeocean-*/; do n=$(basename "$d")
    ln -sfn "$S/$n" "/root/capsule/.claude/skills/$n"; done
```

Keep capsule-specific skills as **real folders** under `.claude/skills/` so they stay out of the
shared repo; only the `codeocean-*` entries are symlinks into it. Pull updates with
`git -C /claude-code-skills-codeocean pull` (links pick up new content; only a *newly added* skill
needs re-linking) — but see `base-CLAUDE.md` CAUTION #6: a `.git` under `/` is often root-owned.

These skills need `pip install codeocean` (already in this env) and a CO API token, taken from
`$CODEOCEAN_TOKEN` / `$API_SECRET` / `$CO_TOKEN` / `$CUSTOM_KEY` or `--token`, with the domain from
`$CODEOCEAN_DOMAIN` (default: the AIND deployment). Launching runs and creating data assets are
**billable** — confirm capsule id, asset ids/mounts, result name and tags first, and use a dry-run
before a real trigger.
