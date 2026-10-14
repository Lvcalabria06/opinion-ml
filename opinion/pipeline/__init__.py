"""
Pipeline de dados do OpiniOn (CC2).

Etapas (cada uma vira uma task da DAG do Airflow em airflow/dags/opinion_pipeline.py):

  1. ingestao       extrai do banco operacional (o que a ingestão MQTT gravou + histórico do app)
                    para uma área de staging, com checagens de consistência
  2. compatibilidade similaridade de gosto entre os dois usuários de cada encontro,
                    usando só as avaliações feitas ANTES do encontro
  3. features       encontro, contexto, perfil e relação — tudo "até o instante do encontro"
  4. label          conectou = 1 se houve interação entre os dois em até 48 h após o fim
  5. preparacao     junta tudo, split temporal, purga da fronteira, imputação, codificação,
                    checagens de qualidade e grava o dataset do modelo

Para rodar sem Airflow:  python -m opinion.pipeline.executar
"""
import os
from pathlib import Path

from opinion.mundo import RAIZ

PASTA_DADOS = Path(os.getenv("OPINION_DATA_DIR", RAIZ / "data"))
PASTA_STAGING = PASTA_DADOS / "staging"
PASTA_PROCESSADOS = PASTA_DADOS / "processed"
CONFIG_PIPELINE = RAIZ / "config" / "pipeline.yaml"
