FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tesseract-ocr \
        tesseract-ocr-vie \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY DominicBE/requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY rag-core /tmp/rag-core
RUN pip install /tmp/rag-core

COPY DominicBE/alembic.ini ./alembic.ini
COPY DominicBE/startup.sh ./startup.sh
COPY DominicBE/alembic ./alembic
COPY DominicBE/app ./app
COPY DominicBE/docker/entrypoint.sh ./docker/entrypoint.sh

RUN chmod +x ./startup.sh ./docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker/entrypoint.sh"]
