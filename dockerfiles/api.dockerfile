FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY api/ api/
COPY conf/ conf/
COPY src/ src/

# Set workdir to project root
WORKDIR /

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

# Expose API port
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
