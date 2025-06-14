# Stage 1: Build dependencies in a virtual environment
FROM python:3.10-slim-bullseye AS builder

# Set environment variables for uv
ENV UV_HOME=/opt/uv
ENV PATH="$UV_HOME/bin:$PATH"

# Install uv - the fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
WORKDIR /app
RUN uv venv

# Copy dependency definitions
COPY pyproject.toml uv.lock* ./

# Install dependencies into the virtual environment
# Using --no-cache to keep the layer small
RUN . .venv/bin/activate && uv pip sync --no-cache

# Stage 2: Create the final, lean production image
FROM python:3.10-slim-bullseye AS final

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser
USER appuser

# Copy the virtual environment with dependencies from the builder stage
COPY --chown=appuser:appuser --from=builder /app/.venv ./.venv

# Copy application source code
COPY --chown=appuser:appuser server.py .
COPY --chown=appuser:appuser .env.example .

# Expose the application port
EXPOSE 8082

# Set the command to run the application using uvicorn
# Note: --host 0.0.0.0 is crucial for Docker networking
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]