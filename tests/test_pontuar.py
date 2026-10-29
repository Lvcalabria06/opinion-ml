"""Checkpoint 2: um encontro novo é traduzido e pontuado com as MESMAS features do treino."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opinion import contrato as C
from opinion.modelo.pontuar import matriz, traduzir
from opinion.mundo import PASTA_BOOTSTRAP
from opinion.pipeline import compatibilidade, features, ingestao, preparacao
from opinion.pipeline.executar import config


@pytest.fixture(scope="module")
def fontes():
    return ingestao.carregar_de_arquivos(PASTA_BOOTSTRAP)


def test_pontuacao_reproduz_features_do_pipeline(fontes):
    cfg = config()
    e = fontes["encontros"]
    amostra = e.iloc[np.random.default_rng(1).choice(len(e), 3, replace=False)]
    feats = features.calcular(e, fontes, cfg)
    compat = compatibilidade.calcular(amostra, fontes["avaliacoes"], fontes["obras"], cfg)
    lote = preparacao.completar(feats.merge(compat, on="encontro_id"))   # caminho do treino
    X_treino, colunas = preparacao.codificar(lote)
    X_treino.index = lote.encontro_id
    for enc in amostra.to_dict("records"):
        X, _ = matriz(enc, fontes, cfg, colunas)                         # caminho da pontuação
        assert list(X.columns) == colunas
        np.testing.assert_allclose(X.iloc[0].to_numpy(dtype=float),
                                   X_treino.loc[enc["encontro_id"]].to_numpy(dtype=float))


def test_traduz_ids_rotativos_e_valida_contrato():
    t = pd.Timestamp("2026-10-01T10:00:00")
    usuarios = ["usr-00001", "usr-00002", "usr-00003"]
    p = {"v": C.VERSAO, "encontro_id": "e-x", "usuario_a": C.token_do_dia("usr-00001", t.date()),
         "usuario_b": C.token_do_dia("usr-00003", t.date()), "inicio": t.isoformat(),
         "duracao_s": 300, "rssi_medio": -60.0, "local": "cinema"}
    e = traduzir(p, usuarios)
    assert (e["usuario_a"], e["usuario_b"]) == ("usr-00001", "usr-00003")
    assert e["fim"] == t + pd.Timedelta(seconds=300)
    with pytest.raises(ValueError, match="contrato"):
        traduzir({**p, "local": "praia"}, usuarios)
    with pytest.raises(ValueError, match="rotativo"):
        traduzir({**p, "usuario_a": "u-0000000000"}, usuarios)
