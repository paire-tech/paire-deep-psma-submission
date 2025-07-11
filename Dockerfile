FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Build argument for environment
ARG BUILD_MODE=prod

# Remove unnecessary packages and clean cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && uv cache clean

WORKDIR /app

# List the pre-installed CUDA dependencies
RUN pip list --format=columns | grep -E "(torch|cuda|nvidia)"

# Copy project files
COPY .python-version pyproject.toml uv.lock README.md ./

# Export requirements, using uv
RUN if [ "$BUILD_MODE" = "dev" ]; then \
        uv export --no-hashes --no-editable --no-emit-project --format requirements-txt > requirements.txt; \
    else \
        uv export --no-hashes --no-editable --no-dev --no-emit-project --format requirements-txt > requirements.txt; \
    fi

# Install (dev) dependencies in the conda environment, 
# to avoid re-installing CUDA dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add source code
COPY src/ main.py ./

# Launch the app
ENTRYPOINT ["python", "main.py"]
