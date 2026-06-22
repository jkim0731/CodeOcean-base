#!/usr/bin/env bash
set -e

echo "[startup] fixing permissions..."

chown -R claude-user:claude-user /root/capsule/.claude
chmod -R a+wX /scratch
chmod -R a+wX /root/capsule/code
chmod -R a+wX /results
chmod o+rx /root/capsule/data
#!/usr/bin/env bash
set -e

echo "[startup] fixing permissions..."

chown -R claude-user:claude-user /root/capsule/.claude
chmod -R a+wX /scratch
chmod -R a+wX /root/capsule/code
chmod -R a+wX /results
chmod o+rx /root/capsule/data
chmod a+wX /

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
