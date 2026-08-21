#!/usr/bin/env bash

set -u

ROOT_HOME="${ROOT_HOME:-/root}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
TMP_ROOT="${TMP_ROOT:-/tmp}"

_is_in_scratch() {
    local path="$1"
    local resolved_path resolved_scratch

    resolved_path=$(readlink -m -- "$path") || return 1
    resolved_scratch=$(readlink -m -- "$SCRATCH_ROOT") || return 1
    [[ "$resolved_path" == "$resolved_scratch" || "$resolved_path" == "$resolved_scratch/"* ]]
}

_safe_rm_rf() {
    local path
    local status=0

    for path in "$@"; do
        if _is_in_scratch "$path"; then
            echo "[disk-cleanup] WARNING: refusing to delete $path because it resolves under $SCRATCH_ROOT"
            status=1
            continue
        fi

        echo "[disk-cleanup] removing $path"
        if ! rm -rf --one-file-system -- "$path"; then
            echo "[disk-cleanup] WARNING: failed to remove $path"
            status=1
        fi
    done

    return "$status"
}

_find_and_remove() {
    local root="$1"
    shift
    local -a paths=()
    local find_fd find_pid
    local path
    local status=0

    if _is_in_scratch "$root"; then
        echo "[disk-cleanup] WARNING: refusing cleanup because $root resolves under $SCRATCH_ROOT"
        return 1
    fi

    coproc DISK_CLEANUP_FIND { find "$root" -xdev "$@" -print0; }
    find_fd="${DISK_CLEANUP_FIND[0]}"
    find_pid="$DISK_CLEANUP_FIND_PID"
    mapfile -d '' paths <&"$find_fd"
    if ! wait "$find_pid"; then
        echo "[disk-cleanup] WARNING: failed to search $root"
        return 1
    fi

    if ((${#paths[@]} == 0)); then
        echo "[disk-cleanup] no matching stale paths under $root"
        return 0
    fi

    for path in "${paths[@]}"; do
        _safe_rm_rf "$path" || status=1
    done
    return "$status"
}

clean_stale_tmp() {
    local status=0

    echo "[disk-cleanup] cleaning stale temporary files"
    _safe_rm_rf "$TMP_ROOT/core" "$TMP_ROOT"/core.* || status=1
    _find_and_remove "$TMP_ROOT" -maxdepth 1 -name 'tmp*' ! -name 'claude-*' -mmin +60 || status=1

    if [ -e "$TMP_ROOT/claude-0" ]; then
        _find_and_remove "$TMP_ROOT/claude-0" -mindepth 1 -maxdepth 2 -type d -mmin +1440 || status=1
    else
        echo "[disk-cleanup] skipping absent $TMP_ROOT/claude-0"
    fi

    _find_and_remove "$TMP_ROOT" -maxdepth 3 -type f -size +50M -mmin +60 || status=1
    return "$status"
}

truncate_speech_dispatcher_log() {
    local log_path="$ROOT_HOME/.cache/speech-dispatcher/log/speech-dispatcher.log"

    if [ ! -f "$log_path" ]; then
        echo "[disk-cleanup] skipping absent speech-dispatcher log"
        return 0
    fi
    if _is_in_scratch "$log_path"; then
        echo "[disk-cleanup] WARNING: refusing to truncate $log_path because it resolves under $SCRATCH_ROOT"
        return 1
    fi

    echo "[disk-cleanup] capping speech-dispatcher log at 1 MB"
    if ! truncate -s 1M "$log_path"; then
        echo "[disk-cleanup] WARNING: failed to truncate $log_path"
        return 1
    fi
}

clear_firefox_cache() {
    local -a cache_paths=()

    shopt -s nullglob
    cache_paths=(
        "$ROOT_HOME"/.cache/mozilla/firefox/*/cache2
        "$ROOT_HOME"/.cache/mozilla/firefox/*/startupCache
        "$ROOT_HOME"/.cache/mozilla/firefox/*/safebrowsing
        "$ROOT_HOME"/.config/mozilla/firefox/*/storage/default
    )
    shopt -u nullglob

    if ((${#cache_paths[@]} == 0)); then
        echo "[disk-cleanup] no Firefox caches to clear"
        return 0
    fi

    echo "[disk-cleanup] clearing Firefox caches"
    _safe_rm_rf "${cache_paths[@]}"
}

main() {
    local status=0

    clean_stale_tmp || status=1
    truncate_speech_dispatcher_log || status=1
    clear_firefox_cache || status=1

    if ((status == 0)); then
        echo "[disk-cleanup] cleanup completed successfully"
    else
        echo "[disk-cleanup] WARNING: cleanup completed with one or more errors"
    fi
    return "$status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
