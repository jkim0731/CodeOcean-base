# Base for agentic coding environments in CodeOcean
- Supports multiple coding agents, including Copilot, Codex, and Claude Code

## How to use
- `Create` -> under `Capsule`, choose `copy from Git` (not `clone`)
- Use this repo's url: https://github.com/jkim0731/CodeOcean-agentic-coding-base
- (optional) to track the new capsule from git, from the capsule, use `capsule` -> `Clone via Git...`, then in Github, `new` -> `import a repository`
- (optional) to track the contents of this repo (e.g., updated capsule infos, updated agent instructions), clone this repo in the capsule using postInstall.

## Features
- Runtime-neutral agent instructions (`code/AGENTS.md`)
- Setting up worktree (for Copilot; in start.sh)
- Linking skills (symlink skills in git repos to corresponding locations; in start.sh)
- Preventing root consumption (Work in progress; in start.sh)
    - Additional cleanup bash (cleanup_disk.sh)

## Agent instructions
- `AGENTS.md`: shared capsule rules and guide selection
- `dev-AGENTS.md`: development, analysis, and multi-agent workflow
- `base-AGENTS.md`: Code Ocean orchestration
- `CLAUDE.md`: compatibility import for `AGENTS.md`

## capsule infos
- updated: 260802
