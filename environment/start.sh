#!/usr/bin/env bash
set -e

# echo "[startup] fixing permissions for claude-user..."

# chown -R claude-user:claude-user /root/capsule/.claude || echo "[startup] 
#   WARNING: chown .claude failed"
# su claude-user -c "rm -rfv /scratch/.Trash*" || echo "[startup] WARNING: rm 
#   .Trash failed"
# find /scratch -maxdepth 2 -user root -exec chmod a+rwX {} + 2>/dev/null ||
#   echo "[startup] WARNING: chmod /scratch failed"
# chmod -R a+rwX /root/capsule/code || echo "[startup] WARNING: chmod code 
#   failed"
# chmod -R a+rwX /results || echo "[startup] WARNING: chmod /results failed"
# chmod o+rx /root/capsule/data || echo "[startup] WARNING: chmod /data failed"
# chmod a+rwX / || echo "[startup] WARNING: chmod / failed"


# --- prevent disk fills from crashing/killed jobs ---
# A crash dumps full process memory to core_pattern (was /tmp/core.<pid>, ~1GB each) and
# abandons its /tmp/tmpXXXX working dir; both pile up on the small root overlay until restart.
echo "[startup] disabling core dumps + sweeping stale /tmp..."
echo '|/bin/false' > /proc/sys/kernel/core_pattern 2>/dev/null || true   # global: no core ever written
echo 'ulimit -c 0' > /etc/profile.d/no-core-dumps.sh 2>/dev/null || true # backup for new login shells
ulimit -c 0 2>/dev/null || true
rm -f /tmp/core /tmp/core.* 2>/dev/null || true                          # clear leftover cores
find /tmp -maxdepth 1 -name 'tmp*' ! -name 'claude-*' -mmin +60 -exec rm -rf {} + 2>/dev/null || true

bash /root/capsule/environment/vscode_setting.sh

# bash /root/capsule/environment/claude_login_bypass.sh
# I don't need it, as long as firefox is updated during postInstall