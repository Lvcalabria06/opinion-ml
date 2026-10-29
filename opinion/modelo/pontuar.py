"""
Pontuação de encontros novos com o baseline registrado no MLflow (Checkpoint 2; base da API).

Um encontro chega no formato do contrato MQTT (IDs anônimos e rotativos), é traduzido
como a ingestão traduz e passa pelas MESMAS funções do pipeline do CC2 (compatibilidade,
features, completar/codificar), usando só o que existia antes do início dele. O modelo
devolve a probabilidade de conexão em 48 h e o limiar escolhido na validação decide o sinal.
"""
from __future__ import annotations

import os

import pandas as pd

from opinion import contrato as C
from opinion.pipeline import compatibilidade, features, preparacao


def carregar_modelo(cfg: dict):
    """Modelo pelo alias do registry + o que o run de treino registrou (limiares, métricas)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    nome, alias = cfg["mlflow"]["modelo_registrado"], cfg["mlflow"]["alias"]
    cliente = MlflowClient()
    versao = cliente.get_model_version_by_alias(nome, alias)
    run = cliente.get_run(versao.run_id)
    modelo = mlflow.sklearn.load_model(f"models:/{nome}@{alias}")
    info = {"uri": f"models:/{nome}@{alias}", "versao": versao.version, "run_id": versao.run_id,
            "algoritmo": run.data.params.get("algoritmo"),
            "limiar": float(run.data.params["limiar"]), "limiar_regra": float(run.data.params["limiar_regra"]),
            "metricas": run.data.metrics}
    return modelo, info


def traduzir(payload: dict, usuarios: list[str]) -> dict:
    """Valida a mensagem contra o contrato MQTT e resolve os IDs rotativos (como opinion.ingestao)."""
    erro = C.validar("encontro", payload)
    if erro:
        raise ValueError(f"encontro fora do contrato: {erro}")
    inicio = pd.Timestamp(payload["inicio"])
    resolv = C.ResolvedorTokens(usuarios)
    a = resolv.resolver(payload["usuario_a"], inicio.to_pydatetime())
    b = resolv.resolver(payload["usuario_b"], inicio.to_pydatetime())
    if a is None or b is None:
        raise ValueError("ID rotativo não reconhecido (segredo diferente ou usuário fora da base)")
    return {"encontro_id": payload["encontro_id"], "pulseira_id": payload.get("pulseira_id"),
            "usuario_a": a, "usuario_b": b, "inicio": inicio,
            "fim": inicio + pd.Timedelta(seconds=float(payload["duracao_s"])),
            "duracao_s": int(payload["duracao_s"]), "rssi_medio": float(payload["rssi_medio"]),
            "local": payload["local"]}


def matriz(encontro: dict, f: dict[str, pd.DataFrame], cfg_pipeline: dict, colunas: list[str]):
    """Features do encontro no formato do treino. Só entra o que existia antes do início."""
    novo = pd.DataFrame([encontro])
    t = novo.inicio.iloc[0]
    anteriores = f["encontros"][(f["encontros"].inicio < t) & (f["encontros"].encontro_id != encontro["encontro_id"])]
    e = pd.concat([anteriores[novo.columns.intersection(anteriores.columns)], novo], ignore_index=True)
    g = dict(f, encontros=e)
    feats = features.calcular(e, g, cfg_pipeline)
    feats = feats[feats.encontro_id == encontro["encontro_id"]]
    compat = compatibilidade.calcular(novo, f["avaliacoes"], f["obras"], cfg_pipeline)
    df = preparacao.completar(feats.merge(compat, on="encontro_id"))
    return preparacao.matriz_features(df, colunas), df


def pontuar(encontro: dict, f, modelo, info: dict, cfg_pipeline: dict, colunas: list[str]) -> dict:
    X, df = matriz(encontro, f, cfg_pipeline, colunas)
    p = float(modelo.predict_proba(X)[:, 1][0])
    linha = df.iloc[0]
    return {"encontro_id": encontro["encontro_id"], "usuario_a": encontro["usuario_a"],
            "usuario_b": encontro["usuario_b"], "inicio": str(encontro["inicio"]), "local": encontro["local"],
            "probabilidade": round(p, 4), "limiar": round(info["limiar"], 4), "sinal_modelo": p >= info["limiar"],
            "compat_score": float(linha.compat_score), "sinal_regra": float(linha.compat_score) >= info["limiar_regra"],
            "algum_segue": int(linha.algum_segue), "modelo": f"{info['uri']} (v{info['versao']})", "_X": X}
