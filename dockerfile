FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p src/agents src/data vector_store

COPY src/ src/
COPY data/ data/

COPY .env .env
ENV PYTHONPATH="/app/src:${PYTHONPATH}"


# Команда запуска
CMD ["python", "-m", "src.main"]