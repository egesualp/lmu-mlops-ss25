# Use official PyTorch image with CUDA 11.8 and Python 3.10 (choose CPU-only if no GPU)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install any additional dependencies (if needed)
COPY requirements.txt .
COPY data

RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src/ ./src/

# Set unbuffered output for easier log viewing
ENV PYTHONUNBUFFERED=1

# Default command (adjust as needed)
CMD ["python", "src/train.py", "--help"]
