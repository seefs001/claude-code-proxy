# Stage 1: Builder
FROM python:3.11-slim-bullseye AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files and install dependencies with uv
COPY pyproject.toml uv.lock* ./
RUN uv pip sync --no-cache --system

# Stage 2: Final
FROM python:3.11-slim-bullseye AS final

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application files
COPY server.py .
COPY .env.example .

# Expose the application port
EXPOSE 8082

# Set the command to run the application
CMD ["python", "server.py"]