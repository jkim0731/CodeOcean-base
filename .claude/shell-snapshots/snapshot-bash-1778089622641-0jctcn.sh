# Snapshot file
# Unset all aliases to avoid conflicts with functions
unalias -a 2>/dev/null || true
# Functions
eval "$(echo 'X19jb25kYV9hY3RpdmF0ZSAoKSAKeyAKICAgIGlmIFsgLW4gIiR7Q09OREFfUFMxX0JBQ0tVUDor
eH0iIF07IHRoZW4KICAgICAgICBQUzE9IiRDT05EQV9QUzFfQkFDS1VQIjsKICAgICAgICBcdW5z
ZXQgQ09OREFfUFMxX0JBQ0tVUDsKICAgIGZpOwogICAgXGxvY2FsIGFza19jb25kYTsKICAgIGFz
a19jb25kYT0iJChQUzE9IiR7UFMxOi19IiBfX2NvbmRhX2V4ZSBzaGVsbC5wb3NpeCAiJEAiKSIg
fHwgXHJldHVybjsKICAgIFxldmFsICIkYXNrX2NvbmRhIjsKICAgIF9fY29uZGFfaGFzaHIKfQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9leGUgKCkgCnsgCiAgICAoICIkQ09OREFfRVhFIiAkX0NFX00gJF9DRV9DT05EQSAi
JEAiICkKfQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9oYXNociAoKSAKeyAKICAgIGlmIFsgLW4gIiR7WlNIX1ZFUlNJT046K3h9IiBdOyB0
aGVuCiAgICAgICAgXHJlaGFzaDsKICAgIGVsc2UKICAgICAgICBpZiBbIC1uICIke1BPU0hfVkVS
U0lPTjoreH0iIF07IHRoZW4KICAgICAgICAgICAgOjsKICAgICAgICBlbHNlCiAgICAgICAgICAg
IFxoYXNoIC1yOwogICAgICAgIGZpOwogICAgZmkKfQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'X19jb25kYV9yZWFjdGl2YXRlICgpIAp7IAogICAgXGxvY2FsIGFza19jb25kYTsKICAgIGFza19j
b25kYT0iJChQUzE9IiR7UFMxOi19IiBfX2NvbmRhX2V4ZSBzaGVsbC5wb3NpeCByZWFjdGl2YXRl
KSIgfHwgXHJldHVybjsKICAgIFxldmFsICIkYXNrX2NvbmRhIjsKICAgIF9fY29uZGFfaGFzaHIK
fQo=' | base64 -d)" > /dev/null 2>&1
eval "$(echo 'Y29uZGEgKCkgCnsgCiAgICBcbG9jYWwgY21kPSIkezEtX19taXNzaW5nX199IjsKICAgIGNhc2Ug
IiRjbWQiIGluIAogICAgICAgIGFjdGl2YXRlIHwgZGVhY3RpdmF0ZSkKICAgICAgICAgICAgX19j
b25kYV9hY3RpdmF0ZSAiJEAiCiAgICAgICAgOzsKICAgICAgICBpbnN0YWxsIHwgdXBkYXRlIHwg
dXBncmFkZSB8IHJlbW92ZSB8IHVuaW5zdGFsbCkKICAgICAgICAgICAgX19jb25kYV9leGUgIiRA
IiB8fCBccmV0dXJuOwogICAgICAgICAgICBfX2NvbmRhX3JlYWN0aXZhdGUKICAgICAgICA7Owog
ICAgICAgICopCiAgICAgICAgICAgIF9fY29uZGFfZXhlICIkQCIKICAgICAgICA7OwogICAgZXNh
Ywp9Cg==' | base64 -d)" > /dev/null 2>&1
# Shell Options
shopt -u autocd
shopt -u assoc_expand_once
shopt -u cdable_vars
shopt -u cdspell
shopt -u checkhash
shopt -u checkjobs
shopt -s checkwinsize
shopt -s cmdhist
shopt -u compat31
shopt -u compat32
shopt -u compat40
shopt -u compat41
shopt -u compat42
shopt -u compat43
shopt -u compat44
shopt -s complete_fullquote
shopt -u direxpand
shopt -u dirspell
shopt -u dotglob
shopt -u execfail
shopt -u expand_aliases
shopt -u extdebug
shopt -u extglob
shopt -s extquote
shopt -u failglob
shopt -s force_fignore
shopt -s globasciiranges
shopt -u globstar
shopt -u gnu_errfmt
shopt -u histappend
shopt -u histreedit
shopt -u histverify
shopt -s hostcomplete
shopt -u huponexit
shopt -u inherit_errexit
shopt -s interactive_comments
shopt -u lastpipe
shopt -u lithist
shopt -u localvar_inherit
shopt -u localvar_unset
shopt -s login_shell
shopt -u mailwarn
shopt -u no_empty_cmd_completion
shopt -u nocaseglob
shopt -u nocasematch
shopt -u nullglob
shopt -s progcomp
shopt -u progcomp_alias
shopt -s promptvars
shopt -u restricted_shell
shopt -u shift_verbose
shopt -s sourcepath
shopt -u xpg_echo
set -o braceexpand
set -o hashall
set -o interactive-comments
set -o monitor
set -o onecmd
shopt -s expand_aliases
# Aliases
# Check for rg availability
if ! (unalias rg 2>/dev/null; command -v rg) >/dev/null 2>&1; then
  function rg {
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/home/claude-user/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command rg "$@"; return; fi
  if [[ -n $ZSH_VERSION ]]; then
    ARGV0=rg "$_cc_bin" "$@"
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=rg "$_cc_bin" "$@"
  elif [[ $BASHPID != $$ ]]; then
    exec -a rg "$_cc_bin" "$@"
  else
    (exec -a rg "$_cc_bin" "$@")
  fi
}
fi
# Shadow find/grep with embedded bfs/ugrep
unalias find 2>/dev/null || true
unalias grep 2>/dev/null || true
function find {
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/home/claude-user/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command find "$@"; return; fi
  if [[ -n $ZSH_VERSION ]]; then
    ARGV0=bfs "$_cc_bin" -S dfs -regextype findutils-default "$@"
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=bfs "$_cc_bin" -S dfs -regextype findutils-default "$@"
  elif [[ $BASHPID != $$ ]]; then
    exec -a bfs "$_cc_bin" -S dfs -regextype findutils-default "$@"
  else
    (exec -a bfs "$_cc_bin" -S dfs -regextype findutils-default "$@")
  fi
}
function grep {
  local _cc_bin="${CLAUDE_CODE_EXECPATH:-}"
  [[ -x $_cc_bin ]] || _cc_bin=/home/claude-user/.local/bin/claude
  if [[ ! -x $_cc_bin ]]; then command grep "$@"; return; fi
  if [[ -n $ZSH_VERSION ]]; then
    ARGV0=ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl "$@"
  elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ARGV0=ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl "$@"
  elif [[ $BASHPID != $$ ]]; then
    exec -a ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl "$@"
  else
    (exec -a ugrep "$_cc_bin" -G --ignore-files --hidden -I --exclude-dir=.git --exclude-dir=.svn --exclude-dir=.hg --exclude-dir=.bzr --exclude-dir=.jj --exclude-dir=.sl "$@")
  fi
}
export PATH=/opt/conda/condabin:/usr/local/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
