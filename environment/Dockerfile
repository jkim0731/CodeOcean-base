# hash:sha256:d039531d3ef8f0155446eb1a6c09fd9495775c2b6fa9f1bfa15342ca712ddde6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-igv-fiji-mambaforge:23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

ARG CACHE_BUST=2026-05-12

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# Update python to 3.11
RUN CONDA_NO_PLUGINS=true mamba install -y -c conda-forge python=3.11 "zarr>=3.0.0" pyqt=5

# VS Code, numlockx gthumb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgbm1 libasound2 libnss3 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 \
    libxcb-xfixes0 libxcb-shape0 libxcb-cursor0 \
    numlockx gthumb vlc \
    && curl -L "https://update.code.visualstudio.com/latest/linux-x64/stable" -o /tmp/vscode.tar.gz \
    && mkdir -p /opt/vscode \
    && tar -xzf /tmp/vscode.tar.gz -C /opt/vscode --strip-components=1 \
    && rm /tmp/vscode.tar.gz \
    && ln -sf /opt/vscode/bin/code /usr/bin/code
# Bash alias
RUN echo "alias code='/opt/vscode/bin/code --no-sandbox --user-data-dir=/root/capsule/scratch/.vscode-root --password-store=basic'" >> /etc/bash.bashrc

# installing tools for cellpose and napari (or other qc apps using pyqt)
RUN apt-get install -y --no-install-recommends \
    qtbase5-dev qt5-qmake qttools5-dev qttools5-dev-tools qtwebengine5-dev libqt5svg5-dev libqt5websockets5-dev

# Updating firefox for interactive login (required for the first time)
# Alternative is to bypass logon information from previously successful login, but this way is better 
# (until we have an updated base image with firefox >=125)
RUN <<'SCRIPT'
set -eux

# 1) Install the latest Firefox into /opt (image layer; survives the runtime mount)
rm -rf /opt/firefox
curl -fSL -o /tmp/firefox.tar.xz "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US"
tar -xJf /tmp/firefox.tar.xz -C /opt
rm -f /tmp/firefox.tar.xz

# 2) Purge the old distro Firefox (reclaims ~200 MB); NO autoremove, so shared libs stay
OLD_PKG="$(dpkg -S "$(readlink -f /usr/bin/firefox)" 2>/dev/null | cut -d: -f1 || true)"
[ -n "${OLD_PKG:-}" ] && { apt-get remove -y --purge "$OLD_PKG"; rm -rf /var/lib/apt/lists/*; }

# 3) Point shell + launcher paths at the new binary
ln -sf /opt/firefox/firefox /usr/local/bin/firefox
ln -sf /opt/firefox/firefox /usr/bin/firefox

# 4) Restore the apps-menu entry the purge removed
cat > /usr/share/applications/firefox.desktop <<'EOF'
[Desktop Entry]
Name=Firefox
Comment=Web Browser
Exec=/opt/firefox/firefox %u
Terminal=false
Type=Application
Icon=/opt/firefox/browser/chrome/icons/default/default128.png
Categories=Network;WebBrowser;
MimeType=text/html;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;
StartupNotify=true
StartupWMClass=firefox
EOF
update-desktop-database /usr/share/applications 2>/dev/null || true

# 5) Build-time sanity check (fails the build loudly if the binary is broken)
/opt/firefox/firefox --version
SCRIPT

ENV BROWSER=/usr/local/bin/firefox

# clean apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    codeocean==0.14.0 \
    aind-codeocean-utils \
    aind-data-access-api \
    aind-ophys-utils \
    aind_session \
    aind-data-schema==1.2.0 \
    aind-log-utils \
    joblib \
    jupyterlab \
    matplotlib \
    pydantic==2.8.2 \
    pydantic_settings \
    pyqtgraph 
#    aind-qcportal-schema \
#    statsmodels \

# Register Python kernel for Jupyter
RUN python3 -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"

# Create VS Code data directory and install extensions
RUN /opt/vscode/bin/code --install-extension ms-python.python --force --user-data-dir=/root/capsule/scratch/.vscode-root --no-sandbox || true \
    && /opt/vscode/bin/code --install-extension ms-toolsai.jupyter --force --user-data-dir=/root/capsule/scratch/.vscode-root --no-sandbox || true \
    && /opt/vscode/bin/code --install-extension GitHub.copilot-chat --force --user-data-dir=/root/capsule/scratch/.vscode-root --no-sandbox || true \
    && /opt/vscode/bin/code --install-extension anthropic.claude-code --force --user-data-dir=/root/capsule/scratch/.vscode-root --no-sandbox || true

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
