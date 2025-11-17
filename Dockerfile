# Dockerfile for Geo6D-Lite Pose Estimation Model
# Usage: docker build -t geo6d-lite . && docker run -p 5000:5000 geo6d-lite

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for API
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Start API server by default
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "5000"]
