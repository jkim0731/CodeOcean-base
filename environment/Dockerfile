# hash:sha256:f021ab0082d32d2954b63da31e988d47a581ca94879eb15b3627e51ed45c93a3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION="6.0"
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-ophys-dff

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-ophys-utils \
    aind-data-schema==1.2.0 \
    aind-log-utils \
    aind-qcportal-schema \
    jax \
    joblib \
    matplotlib \
    pydantic==2.8.2 \
    pydantic_settings \
    statsmodels
