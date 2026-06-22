#!/usr/bin/env bash
set -e

echo "Writing VS Code profile settings for stable kernel detection..."
mkdir -p /root/capsule/scratch/.vscode-root/User
cat > /root/capsule/scratch/.vscode-root/User/settings.json << 'EOF'
{
  "python.useEnvironmentsExtension": false,
  "python.defaultInterpreterPath": "/opt/conda/bin/python3",
  "python.condaPath": "/opt/conda/bin/conda",
  "jupyter.jupyterServerType": "local",
  "jupyter.kernels.excludePythonEnvironments": [],
  "python-envs.workspaceSearchPaths": [
    "./.venv"
  ],
  "python-envs.globalSearchPaths": [
    "/opt/conda",
    "/opt/conda/envs",
    "/root/.conda/envs",
    "/root/.virtualenvs"
  ],
  "python.terminal.activateEnvironment": true,
  "python.experiments.enabled": false,
  "python.testing.pytestEnabled": false,
  "claudeCode.preferredLocation": "panel",
  "search.followSymlinks": false,
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/data/**": true,
    "/data/**": true,
    "**/results/**": true,
    "**/scratch/**": true
  },
  "search.exclude": {
    "**/.git/**": true,
    "**/data/**": true,
    "/data/**": true,
    "**/results/**": true,
    "**/scratch/**": true
  }
}
EOF
