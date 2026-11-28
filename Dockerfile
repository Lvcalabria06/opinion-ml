FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt
COPY config ./config
COPY src ./src
COPY embedded ./embedded
COPY api ./api
COPY app ./app
# o pacote opinion (src/) é usado sem instalar; OPINION_HOME diz onde estão config/ e data/
ENV PYTHONPATH=/app/src:/app \
    OPINION_HOME=/app
