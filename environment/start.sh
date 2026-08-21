#!/usr/bin/env bash
set -e

ROOT_HOME="${ROOT_HOME:-/root}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
CAPSULE_REPO="${CAPSULE_REPO:-/root/capsule}"
SESSION_BRANCH_FALLBACK_REF="${SESSION_BRANCH_FALLBACK_REF:-refs/heads/main}"

_startup_is_in_scratch() {
    local path="$1"
    local resolved_path resolved_scratch

    resolved_path=$(readlink -m -- "$path") || return 1
    resolved_scratch=$(readlink -m -- "$SCRATCH_ROOT") || return 1
    [[ "$resolved_path" == "$resolved_scratch" || "$resolved_path" == "$resolved_scratch/"* ]]
}

_startup_safe_rm_rf() {
    local path
    local status=0

    for path in "$@"; do
        if _startup_is_in_scratch "$path"; then
            echo "[startup] WARNING: refusing to delete $path because it resolves under $SCRATCH_ROOT"
            status=1
            continue
        fi

        echo "[startup] removing $path"
        if ! rm -rf --one-file-system -- "$path"; then
            echo "[startup] WARNING: failed to remove $path"
            status=1
        fi
    done

    return "$status"
}

disable_core_dumps() {
    local status=0

    echo "[startup] disabling core dumps"
    if ! echo '|/bin/false' > /proc/sys/kernel/core_pattern 2>/dev/null; then
        echo "[startup] WARNING: failed to update kernel core pattern"
        status=1
    fi
    if ! echo 'ulimit -c 0' > /etc/profile.d/no-core-dumps.sh 2>/dev/null; then
        echo "[startup] WARNING: failed to configure login-shell core limits"
        status=1
    fi
    if ! ulimit -c 0 2>/dev/null; then
        echo "[startup] WARNING: failed to disable core dumps in this shell"
        status=1
    fi

    return "$status"
}

configure_firefox_updates() {
    local policy_dir=/opt/firefox/distribution
    local policy_file="$policy_dir/policies.json"

    echo "[startup] checking Firefox update policy"
    if ! mkdir -p "$policy_dir"; then
        echo "[startup] WARNING: failed to create $policy_dir"
        return 1
    fi

    if [ -f "$policy_file" ]; then
        echo "[startup] Firefox update policy already exists"
        return 0
    fi

    if ! cat > "$policy_file" <<'FIREFOX_POLICY_EOF'
{"policies": {"DisableAppUpdate": true, "OverrideFirstRunPage": "", "OverridePostUpdatePage": ""}}
FIREFOX_POLICY_EOF
    then
        echo "[startup] WARNING: failed to write $policy_file"
        return 1
    fi
    echo "[startup] created Firefox update policy"
}

_redirect_to_scratch() {
    local src="$1"
    local dst="$2"
    local current_dir resolved_src

    if ! _startup_is_in_scratch "$dst"; then
        echo "[startup] WARNING: redirect destination $dst is not under $SCRATCH_ROOT"
        return 1
    fi
    if ! mkdir -p "$dst"; then
        echo "[startup] WARNING: failed to create $dst"
        return 1
    fi

    if [ -L "$src" ]; then
        echo "[startup] repointing $src to $dst"
        if ! ln -sfn "$dst" "$src"; then
            echo "[startup] WARNING: failed to repoint $src"
            return 1
        fi
    elif [ -d "$src" ]; then
        current_dir=$(pwd -P)
        resolved_src=$(readlink -m -- "$src")
        if [[ "$current_dir" == "$resolved_src" || "$current_dir" == "$resolved_src/"* ]]; then
            echo "[startup] WARNING: preserving $src because the current session is running inside it"
            return 1
        fi

        echo "[startup] copying $src into $dst"
        if ! cp -a --backup=numbered -- "$src/." "$dst/"; then
            echo "[startup] WARNING: copy failed; preserving $src and not creating the symlink"
            return 1
        fi
        echo "[startup] copy completed; replacing $src with a symlink"
        if ! _startup_safe_rm_rf "$src"; then
            echo "[startup] WARNING: preserving $src because it could not be safely removed"
            return 1
        fi
        if ! ln -sfn "$dst" "$src"; then
            echo "[startup] WARNING: copied data to $dst but failed to link $src"
            return 1
        fi
    elif [ -e "$src" ]; then
        echo "[startup] WARNING: preserving non-directory path $src; cannot redirect it"
        return 1
    else
        echo "[startup] linking absent $src to $dst"
        if ! mkdir -p "$(dirname "$src")" || ! ln -sfn "$dst" "$src"; then
            echo "[startup] WARNING: failed to link $src to $dst"
            return 1
        fi
    fi

    echo "[startup] redirect ready: $src -> $dst"
}

_workspace_value() {
    local key="$1"
    local workspace_file="$2"

    awk -v key="$key" 'index($0, key ": ") == 1 {
        sub("^[^:]+: ", "")
        print
        exit
    }' "$workspace_file"
}

_restore_session_branch() {
    local branch="$1"
    local local_ref="refs/heads/$branch"
    local remote_ref="refs/remotes/origin/$branch"

    if git -C "$CAPSULE_REPO" show-ref --verify --quiet "$local_ref"; then
        return 0
    fi
    if git -C "$CAPSULE_REPO" show-ref --verify --quiet "$remote_ref"; then
        echo "[startup] restoring session branch $branch from origin/$branch"
        git -C "$CAPSULE_REPO" update-ref "$local_ref" "$remote_ref"
        return
    fi

    if ! git -C "$CAPSULE_REPO" rev-parse --verify --quiet \
        "$SESSION_BRANCH_FALLBACK_REF^{commit}" >/dev/null; then
        echo "[startup] WARNING: cannot restore session branch $branch; neither origin/$branch nor $SESSION_BRANCH_FALLBACK_REF exists"
        return 1
    fi

    echo "[startup] WARNING: origin/$branch is missing; recreating $branch from $SESSION_BRANCH_FALLBACK_REF before restoring retained files"
    git -C "$CAPSULE_REPO" update-ref "$local_ref" \
        "$SESSION_BRANCH_FALLBACK_REF"
}

_recover_stale_worktree() {
    local target="$1"
    local branch="$2"
    local backup git_file copy_status=0

    if git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi
    if ! _restore_session_branch "$branch"; then
        return 1
    fi

    backup="${target}.recovery-$(date +%Y%m%d_%H%M%S)"
    if [ -e "$backup" ]; then
        echo "[startup] WARNING: recovery backup already exists: $backup"
        return 1
    fi

    echo "[startup] preserving stale worktree at $backup"
    if ! mv -- "$target" "$backup"; then
        echo "[startup] WARNING: failed to preserve stale worktree $target"
        return 1
    fi
    git -C "$CAPSULE_REPO" worktree prune
    if ! git -C "$CAPSULE_REPO" worktree add -- "$target" "$branch"; then
        echo "[startup] WARNING: failed to recreate worktree $target; restoring retained directory"
        mv -- "$backup" "$target" || echo "[startup] WARNING: retained directory remains at $backup"
        return 1
    fi

    git_file=$(cat "$target/.git")
    cp -a -- "$backup/." "$target/" || copy_status=$?
    if ! printf '%s\n' "$git_file" > "$target/.git"; then
        echo "[startup] WARNING: failed to restore Git metadata for $target"
        return 1
    fi
    if [ "$copy_status" -ne 0 ]; then
        echo "[startup] WARNING: failed to restore retained files from $backup"
        git -C "$CAPSULE_REPO" worktree remove --force "$target" \
            || echo "[startup] WARNING: failed to remove partial worktree $target"
        return 1
    fi

    if ! git -C "$target" status --short >/dev/null; then
        echo "[startup] WARNING: recreated worktree failed validation; retained files remain at $backup"
        git -C "$CAPSULE_REPO" worktree remove --force "$target" \
            || echo "[startup] WARNING: failed to remove invalid worktree $target"
        return 1
    fi
    echo "[startup] recovered worktree $target; retained backup kept at $backup"
}

restore_copilot_session_worktrees() {
    local state_root="$ROOT_HOME/.copilot/session-state"
    local workspace_file cwd branch name target
    local status=0

    [ -d "$state_root" ] || return 0
    if [ ! -d "$CAPSULE_REPO/.git" ]; then
        echo "[startup] WARNING: cannot restore session worktrees; $CAPSULE_REPO is not a Git repository"
        return 1
    fi

    for workspace_file in "$state_root"/*/workspace.yaml; do
        [ -f "$workspace_file" ] || continue
        cwd=$(_workspace_value cwd "$workspace_file")
        branch=$(_workspace_value branch "$workspace_file")
        case "$cwd" in
            "$ROOT_HOME/capsule.worktrees/"*) ;;
            *) continue ;;
        esac
        name=${cwd#"$ROOT_HOME/capsule.worktrees/"}
        case "$name" in
            ""|*/*)
                echo "[startup] WARNING: invalid session worktree path in $workspace_file"
                status=1
                continue
                ;;
        esac
        if [ -z "$branch" ]; then
            branch="agents/$name"
            echo "[startup] WARNING: deriving missing session branch as $branch for $workspace_file"
        fi
        if ! git check-ref-format "refs/heads/$branch"; then
            echo "[startup] WARNING: invalid session branch in $workspace_file"
            status=1
            continue
        fi

        _restore_session_branch "$branch" || {
            status=1
            continue
        }

        if [ "$(git -C "$cwd" branch --show-current 2>/dev/null)" = "$branch" ]; then
            continue
        fi

        target="$SCRATCH_ROOT/.capsule-worktrees/$name"
        if [ -d "$target" ]; then
            _recover_stale_worktree "$target" "$branch" || status=1
        else
            echo "[startup] recreating missing session worktree $target"
            if ! git -C "$CAPSULE_REPO" worktree add -- "$target" "$branch"; then
                echo "[startup] WARNING: failed to recreate missing worktree $target"
                status=1
            fi
        fi
    done

    return "$status"
}

symlink_agent_skills() {
    local claude_skills="$CAPSULE_REPO/.claude/skills"
    local copilot_skills="$CAPSULE_REPO/.github/skills"
    local source_dir skill_dir skill_name destination
    local status=0

    echo "[startup] symlinking agent skills for Claude and Copilot"
    if ! mkdir -p "$claude_skills" "$copilot_skills"; then
        echo "[startup] WARNING: failed to create agent skill directories"
        return 1
    fi

    for source_dir in \
        /claude-code-skills-codeocean \
        /ctl-claude-skills-pipeline-runs \
        /claude-code-skills-misc \
        /lamf-analysis/src/lamf_analysis/agent-skills
    do
        [ -d "$source_dir" ] || continue
        for skill_dir in "$source_dir"/*/; do
            [ -f "${skill_dir}SKILL.md" ] || continue
            skill_name=$(basename "$skill_dir")
            for destination in "$claude_skills" "$copilot_skills"; do
                if ! ln -sfn "$source_dir/$skill_name" "$destination/$skill_name"; then
                    echo "[startup] WARNING: failed to link $skill_name into $destination"
                    status=1
                fi
            done
        done
    done

    return "$status"
}

redirect_root_directories() {
    local status=0

    echo "[startup] redirecting overlay-heavy root directories to $SCRATCH_ROOT"
    _redirect_to_scratch "$ROOT_HOME/.vscode/extensions" "$SCRATCH_ROOT/.vscode-extensions" || status=1
    _redirect_to_scratch "$ROOT_HOME/.cache/pip" "$SCRATCH_ROOT/.cache/pip" || status=1
    _redirect_to_scratch "$ROOT_HOME/.local/share/Trash" "$SCRATCH_ROOT/.root-trash" || status=1
    _redirect_to_scratch "$ROOT_HOME/.copilot" "$SCRATCH_ROOT/.copilot" || status=1
    _redirect_to_scratch "$ROOT_HOME/.codex" "$SCRATCH_ROOT/.codex" || status=1
    _redirect_to_scratch "$ROOT_HOME/.local/share/claude" "$SCRATCH_ROOT/.local-share-claude" || status=1
    _redirect_to_scratch "$ROOT_HOME/capsule.worktrees" "$SCRATCH_ROOT/.capsule-worktrees" || status=1
    restore_copilot_session_worktrees || status=1
    return "$status"
}

main() {
    disable_core_dumps || echo "[startup] WARNING: core-dump policy completed with errors"
    configure_firefox_updates || echo "[startup] WARNING: Firefox policy completed with errors"
    redirect_root_directories || echo "[startup] WARNING: scratch redirects completed with errors"

    # Keep root-overlay cleanup reusable during a running session:
    # bash /root/capsule/environment/cleanup_disk.sh
    echo "[startup] running reusable disk cleanup..."
    bash /root/capsule/environment/cleanup_disk.sh \
        || echo "[startup] WARNING: disk cleanup completed with errors"

    # --- disk diagnosis: baseline snapshot + hourly cron ---
    echo "[startup] starting cron + taking disk baseline snapshot..."
    service cron start 2>/dev/null || true
    bash /root/capsule/code/diskmon.sh snap "session_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

    bash /root/capsule/environment/vscode_setting.sh

    # bash /root/capsule/environment/claude_login_bypass.sh
    # I don't need it, as long as firefox is updated during postInstall

    symlink_agent_skills \
        || echo "[startup] WARNING: agent skill linking completed with errors"

    # --- restore Claude Science after a capsule hold ---
    # A hold wipes the container overlay, deleting the claude-science binary
    # (/root/.local/bin), the from-source bwrap (/usr/local/bin) and apt's socat.
    # All session state -- conversations, artifacts, conda envs, encryption key,
    # OAuth token -- lives on /scratch and survives, so relinking the staged
    # binaries and restarting the daemon restores the session as it was.
    # Opt out of the autostart with: touch /scratch/cs-home/.no-autostart
    CS_BIN=/scratch/cs-home/.local/bin
    if [ -x "$CS_BIN/claude-science" ]; then
        echo "[startup] restoring Claude Science binaries..."
        mkdir -p /root/.local/bin
        ln -sfn "$CS_BIN/claude-science" /root/.local/bin/claude-science
        ln -sfn "$CS_BIN/bwrap" /usr/local/bin/bwrap
        ln -sfn "$CS_BIN/socat" /usr/local/bin/socat
        if [ ! -f /scratch/cs-home/.no-autostart ]; then
            echo "[startup] starting Claude Science daemon..."
            bash /scratch/cs-home/cs start >/dev/null 2>&1 \
                || echo "[startup] WARNING: claude-science autostart failed"
        fi
    else
        echo "[startup] WARNING: Claude Science binaries missing from $CS_BIN"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi