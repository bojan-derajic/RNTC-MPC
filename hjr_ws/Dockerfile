# Base image: slim Python 3.13 to keep the image size small
FROM python:3.13-slim

ARG USERNAME=vscode

# Suppress interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Create the workspace directory that will be bind-mounted at runtime
RUN mkdir /workspace
WORKDIR /workspace

# Install system build tools and git (needed for the hj_reachability submodule)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openssh-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching the VS Code Dev Container convention
RUN useradd -ms /bin/bash $USERNAME
RUN usermod -aG sudo $USERNAME

USER $USERNAME

# Install Python dependencies (includes JAX with CUDA 12 support)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Note: hj_reachability is installed separately at container startup
# (via postCreateCommand in devcontainer.json or in generate_data.sh)
# because it is a bind-mounted git submodule, not baked into the image.

CMD ["/bin/bash"]
