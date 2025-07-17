# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY serving/requirements_backend.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend app and model files
COPY serving/backend.py ./backend.py
COPY onnx/ ./onnx/

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
