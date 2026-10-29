"""
Baseline do modelo (CC3).

  python -m opinion.modelo.treinar       treina, compara com a regra simples, registra no MLflow
  python -m opinion.checkpoint2          Checkpoint 2: encontro novo recebe a probabilidade do modelo registrado
"""
from opinion.mundo import RAIZ
from opinion.pipeline import PASTA_PROCESSADOS

CONFIG_MODELO = RAIZ / "config" / "modelo.yaml"

__all__ = ["CONFIG_MODELO", "PASTA_PROCESSADOS"]
