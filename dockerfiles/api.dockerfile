FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY api/ api/
COPY conf/ conf/
COPY src/ src/


WORKDIR /
RUN pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]