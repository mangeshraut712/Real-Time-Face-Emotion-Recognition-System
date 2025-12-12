# Multi-stage build for optimal image size
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY src/web/frontend/package*.json ./
RUN npm ci
COPY src/web/frontend/ ./
RUN npm run build

# Python runtime
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/../static /app/src/web/static

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/status')"

# Run application
CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "5000"]
