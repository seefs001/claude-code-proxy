# Use a single stage for simplicity
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off

# Set working directory
WORKDIR /app

# Install uv, the fast Python package installer
RUN pip install uv

# Copy all necessary files
COPY pyproject.toml uv.lock* ./
COPY server.py .
COPY .env.example .

# Install dependencies using uv
# --system modifies the Python environment directly
RUN uv pip install --no-cache -r pyproject.toml --system

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Expose the application port
EXPOSE 8082

# Set the command to run the application using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]