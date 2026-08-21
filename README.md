# Base for agentic coding environments
- Supports multiple coding agents, including Copilot, Codex, and Claude Code

## Use cases
- Duplicate and make a new dev environment
- Duplicate for a new base to control capsules and pipelines
- Different type of capsule served from different branch: ubuntu, ubuntu-GPU, vscode, vscode-GPU

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
