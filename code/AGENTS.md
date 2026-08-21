# AGENTS.md

Guidance for coding agents working in this Code Ocean capsule.

## What this repository is

This is an interactive Ubuntu research workstation, not a conventional application repository. It
supports two kinds of work:

- **Code Ocean orchestration:** Read `base-AGENTS.md` before searching or attaching data assets,
  running capsules or pipelines, or capturing derived assets.
- **Development and analysis:** Read `dev-AGENTS.md` before developing tools, methods, notebooks, or
  large analyses.

Read only the guides and project documents relevant to the task. Project-specific background lives
in `/root/capsule/code/background_information.md`.

## Core rules

- Never use `/`, `/root`, or `/tmp` for task data or caches; the root overlay is small. Use a
  task-named directory under `/scratch`, verify it is writable, and clean up disposable files.
  Use `/scratch/tmp` only when it is writable.
- Preserve intermediate or interactive outputs under `/scratch/<descriptive-name>`. Interactive
  `/results` is ephemeral; reproducible runs use `/results` for captured outputs.
- Install packages without caches. Prefer the existing environment and project tooling.
- Never print secret values. Before publishing results or transcripts, scrub credentials and
  remove credential files.
- Confirm identifiers, mounts, names, tags, and expected cost before billable Code Ocean actions.
  Use a dry run whenever supported.
- Inspect before editing, make focused changes, and validate the result.

## Code Ocean distinctions

- Attaching assets to a reproducible run targets a capsule and uses its `capsule_id`.
- Capturing files from this interactive workstation uses its `computation_id`.
- `code/CO_capsule_infos.xlsx` is the source of truth for capsule IDs, URLs, suffixes, tags, and
  required data types. Do not guess these values.

## Capsule layout

- `code/`: working scripts and notebooks, project guidance, and the capsule registry.
- `data/`: read-only assets attached to a run.
- `results/`: reproducible-run output; ephemeral during interactive work.
- `scratch/` -> `/scratch`: persistent interactive data and task storage.
- `environment/`: image and startup configuration.
- `.codeocean/`: resource settings and secret names. Never expose secret values.
- `metadata/metadata.yml`: capsule metadata shown in Code Ocean.

## Environment and tools

The Python 3.11 environment includes the Code Ocean client, AIND pipeline utilities, and
`aind-data-schema==1.2.0`. `lamf-analysis` provides `code_ocean_utils`, `zstack_utils`, and
`json_utils`; see `base-AGENTS.md` for their constraints.

Agent runtimes expose different tools and skill-discovery mechanisms. Use compatible skills when
available, but treat repository documentation and installed source as authoritative. Inspect
`environment/postInstall` and the installed skill repositories for their current locations and
capabilities rather than assuming a runtime-specific layout.
