FROM apache/airflow:3.1.0-python3.12
COPY requirements.txt /tmp/requirements.txt
# Fixar o apache-airflow na mesma instalação impede que as dependências do modelo troquem
# versões de pacotes que o Airflow usa (FastAPI, SQLAlchemy...). mlflow-skinny basta para
# registrar o modelo no servidor MLflow do compose. O scikit-learn tem a MESMA versão de
# requirements-ml.txt: a API carrega o pickle que o Airflow treina.
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        "apache-airflow==3.1.0" "scikit-learn==1.9.1" "mlflow-skinny>=3.5" "matplotlib>=3.8"
