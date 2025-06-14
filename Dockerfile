# Stage 1: Builder
FROM python:3.11-slim-bullseye AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
COPY CLAUDE.md .

# Expose the application port
EXPOSE 8082

# Set the command to run the application
CMD ["python", "server.py"]