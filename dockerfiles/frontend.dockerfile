# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY serving/requirements_frontend.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY serving/app.py ./app.py

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables (optional: avoid Streamlit telemetry)
ENV STREAMLIT_TELEMETRY_DISABLED=true

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 