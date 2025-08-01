# Base image
FROM python:3.12-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY conf/ conf/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/train_hf.py"]
