"""
API de inferência do OpiniOn (CC7).

    uvicorn api.main:app --host 0.0.0.0 --port 8000        # da raiz do repositório
    http://localhost:8000/docs                             # documentação interativa (Swagger)

Rotas
  GET  /saude                  estado da API, do banco e do modelo
  GET  /modelo                 modelo servido (MLflow), limiar e métricas de validação/teste
  POST /prever                 encontro no formato do contrato MQTT (IDs rotativos) → chance de interação
  POST /prever/usuarios        o mesmo com os IDs reais dos usuários (demonstração e dashboard)
  GET  /inferencias            últimas inferências (consumidor MQTT e API)
  GET  /encontros/recentes     últimos encontros gravados pela ingestão, com o aviso de afinidade
  GET  /indicadores            taxa de conversão, precisão dos sinais e taxa de silenciamento

As features são calculadas pelas mesmas funções do pipeline (CC2), só com dados anteriores
ao início do encontro; o modelo é o de produção registrado no MLflow (opinion-producao@producao,
com recuo para o otimizado do CC5).
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from opinion import contrato as C
from opinion.db import Banco
from opinion.indicadores import Indicadores
from opinion.inferencia import Inferencia
from opinion.inferencia_mqtt import gravar

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [api] %(message)s")

app = FastAPI(title="OpiniOn — API de inferência",
              description="Chance de dois usuários que se encontraram interagirem no OpiniOn em até 48 h.",
              version="1.0.0")
_inf: Inferencia | None = None
_banco: Banco | None = None
_ind: Indicadores | None = None


def inf() -> Inferencia:
    global _inf
    if _inf is None:
        _inf = Inferencia()
    return _inf


def banco() -> Banco:
    global _banco
    if _banco is None:
        _banco = Banco()
        _banco.criar_esquema()
    return _banco


def indicadores() -> Indicadores:
    global _ind
    if _ind is None:
        _ind = Indicadores(banco(), ttl=float(os.getenv("OPINION_INDICADORES_TTL_S", 60)))
    return _ind


# ---------------------------------------------------------------- esquemas
Local = Literal["convencao_games", "cinema", "livraria", "campus"]


class EncontroMQTT(BaseModel):
    """Mensagem `encontro` da pulseira (docs/contrato_mqtt.md)."""
    v: int = Field(C.VERSAO, description="versão do contrato")
    encontro_id: str = Field(default_factory=lambda: f"e-api-{uuid.uuid4().hex[:8]}")
    usuario_a: str = Field(..., description="ID anônimo do dia, u-xxxxxxxxxx (dono da pulseira)")
    usuario_b: str = Field(..., description="ID anônimo do dia da outra pessoa")
    inicio: str = Field(..., description="ISO-8601")
    duracao_s: float = Field(..., gt=0)
    rssi_medio: float = Field(..., ge=-110, le=0)
    local: Local
    pulseira_id: str | None = None

    model_config = {"json_schema_extra": {"examples": [{
        "v": 1, "encontro_id": "e-demo-001", "usuario_a": "u-3f9a1c2b7e", "usuario_b": "u-0b7c41d2aa",
        "inicio": "2026-09-30T20:15:00", "duracao_s": 600, "rssi_medio": -58, "local": "cinema"}]}}


class EncontroUsuarios(BaseModel):
    """Encontro com os IDs reais (usr-xxxxx). Útil para demonstração; a pulseira usa /prever."""
    usuario_a: str = Field(..., examples=["usr-00694"])
    usuario_b: str = Field(..., examples=["usr-01029"])
    local: Local = "cinema"
    duracao_s: float = Field(600, gt=0)
    rssi_medio: float = Field(-58, ge=-110, le=0)
    inicio: str | None = Field(None, description="ISO-8601; padrão: agora")
    encontro_id: str | None = None


def _executar(fn, registrar: bool):
    try:
        r = fn()
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    except RuntimeError as ex:
        raise HTTPException(503, str(ex))
    if registrar:
        gravar(banco(), r, False, "api")
    return r


# ---------------------------------------------------------------- rotas
@app.get("/")
def raiz():
    return {"servico": "OpiniOn — API de inferência", "documentacao": "/docs",
            "rotas": ["/saude", "/modelo", "/prever", "/prever/usuarios", "/inferencias", "/encontros/recentes",
                      "/indicadores"]}


@app.get("/saude")
def saude():
    estado = {"api": "ok", "hora": datetime.now().isoformat(timespec="seconds")}
    try:
        banco().um("SELECT 1")
        estado["banco"] = "ok"
    except Exception as ex:
        estado["banco"] = f"erro: {ex}"
    try:
        _, info = inf().modelo()
        estado["modelo"] = f"{info['nome']} v{info['versao']}"
    except Exception as ex:
        estado["modelo"] = f"indisponível: {ex}"
    return estado


@app.get("/modelo")
def modelo():
    try:
        _, info = inf().modelo()
    except RuntimeError as ex:
        raise HTTPException(503, str(ex))
    return {k: v for k, v in info.items() if k != "colunas"} | {"n_features": len(info["colunas"]),
                                                                 "faixa_perceptivel_fator": inf().fator_perceptivel}


@app.post("/prever")
def prever(encontro: EncontroMQTT, registrar: bool = Query(False, description="grava em `inferencias`")):
    """Recebe o encontro como a pulseira o publica (IDs anônimos do dia) e devolve a chance de interação."""
    payload = encontro.model_dump()
    return _executar(lambda: inf().prever_payload(payload), registrar)


@app.post("/prever/usuarios")
def prever_usuarios(e: EncontroUsuarios, registrar: bool = Query(False)):
    inicio = pd.Timestamp(e.inicio) if e.inicio else pd.Timestamp.now().floor("s")
    dono = inf().dados()["usuarios"].set_index("usuario_id").pulseira_id
    enc = {"encontro_id": e.encontro_id or f"e-api-{uuid.uuid4().hex[:8]}", "pulseira_id": dono.get(e.usuario_a),
           "usuario_a": e.usuario_a, "usuario_b": e.usuario_b, "inicio": inicio,
           "fim": inicio + pd.Timedelta(seconds=e.duracao_s), "duracao_s": int(e.duracao_s),
           "rssi_medio": float(e.rssi_medio), "local": e.local}
    return _executar(lambda: inf().prever(enc), registrar)


@app.get("/inferencias")
def inferencias(limite: int = Query(50, ge=1, le=1000)):
    cols = ["encontro_id", "pulseira_id", "usuario_a", "usuario_b", "inicio", "fim", "local", "probabilidade",
            "limiar", "sinal", "nivel", "midias_aviso", "compat_score", "algum_segue", "modelo", "latencia_ms",
            "feedback_enviado", "origem", "criado_em"]
    linhas = banco().todos(f"SELECT {', '.join(cols)} FROM inferencias ORDER BY criado_em DESC LIMIT {int(limite)}")
    return [{k: (str(v) if isinstance(v, datetime) else v) for k, v in zip(cols, l)} for l in linhas]


@app.get("/encontros/recentes")
def encontros_recentes(limite: int = Query(50, ge=1, le=1000)):
    cols = ["encontro_id", "pulseira_id", "usuario_a", "usuario_b", "inicio", "duracao_s", "rssi_medio", "local",
            "recebido_em", "midias_aviso"]
    linhas = banco().todos(
        "SELECT e.encontro_id, e.pulseira_id, e.usuario_a, e.usuario_b, e.inicio, e.duracao_s, e.rssi_medio, e.local, "
        "e.recebido_em, a.midias FROM encontros e LEFT JOIN avisos_proximidade a ON a.encontro_id = e.encontro_id "
        f"ORDER BY e.recebido_em DESC, e.inicio DESC LIMIT {int(limite)}")
    return [{k: (str(v) if isinstance(v, datetime) else v) for k, v in zip(cols, l)} for l in linhas]


@app.get("/indicadores")
def get_indicadores():
    return indicadores().obter()
