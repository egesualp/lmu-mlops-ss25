# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only the required Python packages for sentiment monitoring
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    pandas==2.3.0 \
    evidently==0.6.7 \
    google-cloud-storage>=2.0.0 \
    nltk>=3.8 \
    anyio==4.9.0

# Copy only the monitoring application
COPY api/sentiment_monitoring.py ./api/


# Create necessary directories
RUN mkdir -p /app/data/processed /app/output_folder/processed

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('words', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080


# Run the application
CMD ["uvicorn", "api.sentiment_monitoring:app", "--host", "0.0.0.0", "--port", "8080"]
