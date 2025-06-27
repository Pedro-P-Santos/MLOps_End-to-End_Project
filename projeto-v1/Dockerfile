# Stage 1: Runtime environment with dependencies
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime-environment

# System and Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -U "pip>=21.2,<23.2"
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Create dedicated user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

# Ensure mlflow folders exist with full permissions
RUN mkdir -p /ml_artifacts /mlruns && chmod -R 777 /ml_artifacts /mlruns

WORKDIR /home/kedro_docker
USER kedro_docker

# Stage 2: Copy project code
FROM runtime-environment

ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888

CMD ["kedro", "run"]
