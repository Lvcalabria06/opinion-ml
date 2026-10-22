"""
CHECKPOINT 2 (Sprint 3 / CC3): o baseline explicável está no MLflow, foi treinado com
split temporal e supera a regra simples de compatibilidade no teste — inclusive entre
desconhecidos, o público da pulseira.

    python -m opinion.checkpoint2
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from opinion.modelo import CONFIG_MODELO, PASTA_PROCESSADOS
from opinion.modelo.treinar import _imprimir, carregar, config, mascara_desconhecidos, metricas, treinar


def main():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = config()
    if not (Path(PASTA_PROCESSADOS) / "dataset_modelo.parquet").exists():
        sys.exit("Falta o dataset do modelo. Rode: python -m opinion.pipeline.executar --fonte arquivos")

    print("1) Dataset do modelo (split temporal treino / validação / teste)")
    dm, meta = carregar()
    for s, v in meta["por_split"].items():
        print(f"   {s:<10} {v['encontros']:>7,} encontros  {v['taxa_conexao']:.1%} conectou")

    print("2) Treinando o baseline (Regressão Logística e Árvore de Decisão) e registrando no MLflow")
    r = treinar(registrar=True)
    _imprimir(r)

    print("3) Recarregando o modelo pelo alias do registry")
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    uri = f"models:/{cfg['mlflow']['modelo_registrado']}@{cfg['mlflow']['alias']}"
    modelo = mlflow.sklearn.load_model(uri)
    te = dm[dm.split == "teste"]
    p = modelo.predict_proba(te[meta["colunas_features"]])[:, 1]
    m = metricas(te.conectou, p, mascara_desconhecidos(te))
    regra = r["regra_simples"]["teste"]
    print(f"   {uri}")
    print(f"   PR-AUC desconhecidos (recarregado) {m['pr_auc_desconhecidos']:.3f}  "
          f"regra {regra['pr_auc_desconhecidos']:.3f}")
    if m["pr_auc_desconhecidos"] <= regra["pr_auc_desconhecidos"]:
        sys.exit("   FALHOU: o modelo recarregado não supera a regra simples entre desconhecidos")
    print("\nCHECKPOINT 2 OK ✔")


if __name__ == "__main__":
    main()
