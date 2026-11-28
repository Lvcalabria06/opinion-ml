"""CC7/CC9: inferência de encontros novos, consumidor MQTT, API e indicadores."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from opinion import contrato as C
from opinion.db import Banco
from opinion.inferencia import Inferencia, mensagem_feedback
from opinion.mundo import PASTA_BOOTSTRAP
from opinion.pipeline import compatibilidade, features, ingestao, preparacao
from opinion.pipeline.executar import config as config_pipeline


class ModeloFalso:
    """Probabilidade = compat_score: basta para testar o caminho sem depender do MLflow."""

    def __init__(self, colunas):
        self.feature_names_in_ = np.array(colunas)

    def predict_proba(self, X):
        p = X["compat_score"].to_numpy(dtype=float)
        return np.c_[1 - p, p]


@pytest.fixture(scope="module")
def fontes():
    return ingestao.carregar_de_arquivos(PASTA_BOOTSTRAP)


@pytest.fixture(scope="module")
def colunas(fontes):
    cfg = config_pipeline()
    e = fontes["encontros"].head(200)
    feats = features.calcular(e, dict(fontes, encontros=e), cfg)
    compat = compatibilidade.calcular(e, fontes["avaliacoes"], fontes["obras"], cfg)
    _, cols = preparacao.codificar(preparacao.completar(feats.merge(compat, on="encontro_id")))
    return cols


def _inferencia(fontes, colunas, limiar=0.4):
    inf = Inferencia(fonte="arquivos", recarregar_s=1e9)
    inf._f, inf._f_em = fontes, time.time()
    inf._resolvedor = C.ResolvedorTokens(fontes["usuarios"].usuario_id.tolist())
    inf._modelo = ModeloFalso(colunas)
    inf._info = {"nome": "teste", "versao": "1", "algoritmo": "falso", "limiar": limiar, "colunas": colunas}
    inf._checado_em = time.time()
    return inf


def test_inferencia_usa_as_mesmas_features_do_treino(fontes, colunas):
    """O recorte aos encontros do par não muda nenhuma feature (comparado ao cálculo com todos)."""
    cfg = config_pipeline()
    e = fontes["encontros"]
    amostra = e.iloc[np.random.default_rng(4).choice(len(e), 4, replace=False)]
    feats = features.calcular(e, fontes, cfg)
    compat = compatibilidade.calcular(amostra, fontes["avaliacoes"], fontes["obras"], cfg)
    lote = preparacao.completar(feats.merge(compat, on="encontro_id"))
    X_ref, cols = preparacao.codificar(lote)
    X_ref.index = lote.encontro_id
    inf = _inferencia(fontes, cols)
    for enc in amostra.to_dict("records"):
        r = inf.prever(enc)
        assert r["compat_score"] == pytest.approx(X_ref.loc[enc["encontro_id"], "compat_score"], abs=1e-4)
        assert r["encontros_anteriores_par"] == X_ref.loc[enc["encontro_id"], "n_encontros_anteriores_par"]
        assert r["algum_segue"] == X_ref.loc[enc["encontro_id"], "algum_segue"]
        assert r["nivel"] == inf.faixa(r["probabilidade"], 0.4)


def test_faixas_e_mensagem_de_feedback():
    inf = Inferencia.__new__(Inferencia)
    inf.fator_perceptivel = 1.5
    assert inf.faixa(0.09, 0.1) == "nenhum"
    assert inf.faixa(0.12, 0.1) == "periferico_leve"
    assert inf.faixa(0.16, 0.1) == "periferico_perceptivel"
    m = mensagem_feedback({"encontro_id": "e1", "nivel": "periferico_leve", "probabilidade": 0.2, "midias_aviso": []})
    assert m["v"] == C.VERSAO and m["tipo"] == "previsao_48h" and m["padrao"] == "luz_suave"


def test_consumidor_pontua_grava_e_sinaliza(fontes, colunas, monkeypatch):
    from opinion.inferencia_mqtt import ConsumidorInferencia

    banco = Banco("sqlite:///:memory:")
    inf = _inferencia(fontes, colunas, limiar=0.0)            # todo encontro recebe sinal
    c = ConsumidorInferencia(banco, inf)
    enviados = []
    monkeypatch.setattr(c.cliente, "publish", lambda t, m, qos: enviados.append((t, json.loads(m))))
    u = fontes["usuarios"]
    a, b = u.usuario_id.iloc[10], u.usuario_id.iloc[20]
    pulseira = u.pulseira_id.iloc[10]
    t = pd.Timestamp("2026-10-02T19:00:00")
    msg = {"v": 1, "encontro_id": "e-live-1", "usuario_a": C.token_do_dia(a, t.date()),
           "usuario_b": C.token_do_dia(b, t.date()), "inicio": t.isoformat(), "duracao_s": 300,
           "rssi_medio": -60, "local": "cinema"}
    r = c.processar(C.topico(pulseira, "encontro"), json.dumps(msg))
    assert r["usuario_a"] == a and r["usuario_b"] == b
    assert c.processar(C.topico(pulseira, "encontro"), json.dumps(msg)) is None          # reentrega
    assert banco.um("SELECT COUNT(*), MAX(feedback_enviado) FROM inferencias") == (1, 1)
    assert enviados[0][0] == C.topico(pulseira, "feedback") and enviados[0][1]["encontro_id"] == "e-live-1"
    with pytest.raises(ValueError, match="dono"):
        c.processar(C.topico(u.pulseira_id.iloc[30], "encontro"), json.dumps({**msg, "encontro_id": "e-live-2"}))
    # pulseira pausada: pontua e grava, mas não sinaliza
    banco.executar("INSERT INTO status_pulseira (pulseira_id, usuario_id, modo, ts) VALUES (?,?,?,?)",
                   (pulseira, a, "pausado", t.isoformat()))
    c.processar(C.topico(pulseira, "encontro"), json.dumps({**msg, "encontro_id": "e-live-3"}))
    assert len(enviados) == 1
    assert banco.um("SELECT feedback_enviado FROM inferencias WHERE encontro_id = 'e-live-3'")[0] == 0


def test_api_preve_e_lista(fontes, colunas, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api.main as api

    inf = _inferencia(fontes, colunas)
    monkeypatch.setattr(api, "_inf", inf)
    monkeypatch.setattr(api, "_banco", Banco(f"sqlite:///{tmp_path / 'api.db'}"))
    api._banco.criar_esquema()
    monkeypatch.setattr(api, "_ind", None)
    cli = TestClient(api.app)
    u = fontes["usuarios"].usuario_id
    r = cli.post("/prever/usuarios?registrar=true", json={"usuario_a": u.iloc[1], "usuario_b": u.iloc[2],
                                                           "local": "livraria", "inicio": "2026-10-01T15:00:00"})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["probabilidade"] <= 1
    t = pd.Timestamp("2026-10-01T15:00:00")
    corpo = {"usuario_a": C.token_do_dia(u.iloc[1], t.date()), "usuario_b": C.token_do_dia(u.iloc[3], t.date()),
             "inicio": t.isoformat(), "duracao_s": 420, "rssi_medio": -61, "local": "cinema"}
    r = cli.post("/prever", json=corpo)
    assert r.status_code == 200 and r.json()["usuario_b"] == u.iloc[3]
    assert cli.post("/prever", json={**corpo, "usuario_b": "u-0000000000"}).status_code == 422
    assert cli.post("/prever", json={**corpo, "local": "praia"}).status_code == 422
    assert len(cli.get("/inferencias").json()) == 1
    assert cli.get("/modelo").json()["limiar"] == 0.4
    assert cli.get("/saude").json()["banco"] == "ok"


def test_indicadores_conversao_precisao_e_silenciamento():
    from opinion.indicadores import calcular, carregar

    b = Banco("sqlite:///:memory:")
    b.criar_esquema()
    enc = [("e1", "p1", "a", "b", "2026-01-01T10:00:00", 600, "cinema"),
           ("e2", "p1", "a", "c", "2026-01-01T11:00:00", 600, "campus"),
           ("e3", "p2", "b", "c", "2026-01-01T12:00:00", 600, "campus")]
    b.executar_varios("INSERT INTO encontros (encontro_id, pulseira_id, usuario_a, usuario_b, inicio, duracao_s, local, "
                      "token_a, token_b, fim) VALUES (?,?,?,?,?,?,?,'t','t','x')", enc)
    b.executar_varios("INSERT INTO interacoes (interacao_id, origem, destino, tipo, criado_em) VALUES (?,?,?,?,?)",
                      [("i1", "b", "a", "seguir", "2026-01-02T09:00:00"),
                       ("i2", "x", "y", "seguir", "2026-01-05T00:00:00")])
    b.executar_varios("INSERT INTO avisos_proximidade (encontro_id, pulseira_id, midias, inicio) VALUES (?,?,?,?)",
                      [("e1", "p1", "musica", "2026-01-01T10:00:00"), ("e2", "p1", "filme", "2026-01-01T11:00:00")])
    b.executar("INSERT INTO status_pulseira (pulseira_id, usuario_id, modo, ts) VALUES (?,?,?,?)",
               ("p1", "a", "pausado", "2026-01-01T13:00:00"))
    r = calcular(carregar(b))
    assert r["encontros"] == 3 and r["taxa_conversao"] == pytest.approx(1 / 3)
    assert r["aviso_afinidade"]["sinais"] == 2 and r["aviso_afinidade"]["precisao"] == pytest.approx(0.5)
    assert r["silenciamento"]["seguidos_de_pausa_24h"] == 2 and r["silenciamento"]["taxa"] == pytest.approx(1.0)
    assert r["silenciamento"]["pulseiras_pausadas_agora"] == 1
