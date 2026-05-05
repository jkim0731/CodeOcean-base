# hash:sha256:d039531d3ef8f0155446eb1a6c09fd9495775c2b6fa9f1bfa15342ca712ddde6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/ubuntu-desktop-igv-fiji-mambaforge:23.1.0-4-python3.10.12-fiji20231107-1617-firefox120.0-igv2.16.1-guacamole1.5.2-ubuntu22.04

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
    numlockx gthumb \
    && curl -L "https://update.code.visualstudio.com/latest/linux-x64/stable" -o /tmp/vscode.tar.gz \
    && mkdir -p /opt/vscode \
    && tar -xzf /tmp/vscode.tar.gz -C /opt/vscode --strip-components=1 \
    && rm /tmp/vscode.tar.gz \
    && ln -sf /opt/vscode/bin/code /usr/bin/code
# Bash alias
RUN echo "alias code='/opt/vscode/bin/code --no-sandbox --user-data-dir=/root/capsule/scratch/.vscode-root'" >> /etc/bash.bashrc

# installing tools for cellpose and napari
#RUN apt-get install -y --no-install-recommends \
#    qtbase5-dev qt5-qmake qttools5-dev qttools5-dev-tools qtwebengine5-dev libqt5svg5-dev libqt5websockets5-dev

# clean apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ophys-utils \
    aind-data-schema==1.2.0 \
    aind-log-utils \
    aind-qcportal-schema \
    jax \
    joblib \
    jupyterlab \
    matplotlib \
    pydantic==2.8.2 \
    pydantic_settings \
    statsmodels


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
