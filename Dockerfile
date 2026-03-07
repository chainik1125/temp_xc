FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    curl git wget bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 (LTS) for Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create conda env with Python 3.12, then install pip packages
RUN conda create -n torchgpu python=3.12 -y
COPY requirements_torchgpu.txt /tmp/requirements_torchgpu.txt
RUN conda run -n torchgpu pip install --no-cache-dir \
    -r /tmp/requirements_torchgpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create non-root home dir usable by any UID (for --user flag)
# Claude Code refuses --dangerouslySkipPermissions as root,
# so we run with --user $(id -u) at launch time
RUN mkdir -p /home/sandbox /workspace \
    && chmod 777 /home/sandbox /workspace
# Let any user use conda
RUN chmod -R a+rX /opt/conda

# Workspace setup
WORKDIR /workspace
ENV TQDM_DISABLE=1
ENV PYTHONPATH=/workspace
ENV PATH="/opt/conda/bin:$PATH"

CMD ["bash"]
