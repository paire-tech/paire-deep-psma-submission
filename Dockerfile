ARG PYTORCH="2.6.0"
ARG CUDA="12.4"
ARG CUDNN="9"
ARG BUILD_MODE=prod

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user
RUN useradd --uid 1000 appuser

# Required environment variables for nnUNet
ENV nnUNet_raw="/app/nnUNet_raw"
ENV nnUNet_preprocessed="/app/nnUNet_preprocessed"
ENV nnUNet_results="/app/nnUNet_results"

# Remove unnecessary packages and clean cache
RUN apt-get update && apt-get install -y tree && rm -rf /var/lib/apt/lists/*

# Set app directory
RUN mkdir -p /app && chown -R appuser:appuser /app
WORKDIR /app

# List the pre-installed CUDA dependencies
RUN pip list --format=columns | grep -E "(torch|cuda|nvidia)"

# Copy project files
COPY --chown=appuser:appuser .python-version pyproject.toml uv.lock README.md src/ ./

# Setup directories / ckpt for nnUNet
COPY --chown=appuser:appuser ./data/nnUNet_raw/ /app/nnUNet_raw/
COPY --chown=appuser:appuser ./data/nnUNet_preprocessed/ /app/nnUNet_preprocessed/
# PSMA Ensembling
COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset801_PSMA_PET/ /app/nnUNet_results/Dataset801_PSMA_PET/
# COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset901_PSMA_PET/ /app/nnUNet_results/Dataset901_PSMA_PET/
COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset911_PSMA_PET/ /app/nnUNet_results/Dataset911_PSMA_PET/
COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset921_PSMA_PET/ /app/nnUNet_results/Dataset921_PSMA_PET/
# FDG Ensembling
COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset802_FDG_PET/ /app/nnUNet_results/Dataset802_FDG_PET/
# COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset902_FDG_PET/ /app/nnUNet_results/Dataset902_FDG_PET/
# COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset912_FDG_PET/ /app/nnUNet_results/Dataset912_FDG_PET/
COPY --chown=appuser:appuser ./data/nnUNet_results/Dataset922_FDG_PET/ /app/nnUNet_results/Dataset922_FDG_PET/

RUN tree /app/nnUNet_results -L 2 --sort name

# Export requirements, using uv
RUN if [ "$BUILD_MODE" = "dev" ]; then \
        uv export --no-hashes --dev --format requirements-txt > requirements.txt; \
    else \
        uv export --no-hashes --no-editable --no-dev --format requirements-txt > requirements.txt; \
    fi

# Install (dev) dependencies in the conda environment, 
# to avoid re-installing CUDA dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Launch the app
ENTRYPOINT ["paire-deep-psma-submission"]
