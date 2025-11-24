# -----------------------------
# Stage 1: Builder for Python packages
# -----------------------------
    FROM python:3.11-slim as builder

    WORKDIR /app
    
    # Install build dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements first for caching
    COPY requirements.txt .
    
    # Upgrade pip and install Python packages
    RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
        pip install --no-cache-dir -r requirements.txt
    
    # -----------------------------
    # Stage 2: Final runtime image
    # -----------------------------
    FROM python:3.11-slim
    
    WORKDIR /app
    
    # Install runtime dependencies only
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy Python packages from builder
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    COPY --from=builder /usr/local/bin /usr/local/bin
    
    # Copy application code
    COPY . .
    
    # Create non-root user
    RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
    USER appuser
    
    # Expose port
    EXPOSE 8000
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
        CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
    
    # Start server
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    