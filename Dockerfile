FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config ./config
COPY src ./src
COPY embedded ./embedded
# o pacote opinion (src/) é usado sem instalar; OPINION_HOME diz onde estão config/ e data/
ENV PYTHONPATH=/app/src \
    OPINION_HOME=/app
