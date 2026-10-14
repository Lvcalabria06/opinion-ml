"""
Testes do pipeline (CC2).

O principal é o TESTE DE VAZAMENTO: para encontros sorteados, recalcula features e
compatibilidade apagando tudo o que aconteceu a partir do início do encontro. Se o
resultado mudar, alguma feature estava "olhando para o futuro".

    python -m pytest tests -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opinion.mundo import PASTA_BOOTSTRAP
from opinion.pipeline import compatibilidade, features, ingestao, label
from opinion.pipeline.executar import config
from opinion.pipeline.tempo import contar


@pytest.fixture(scope="module")
def fontes():
    return ingestao.carregar_de_arquivos(PASTA_BOOTSTRAP)


@pytest.fixture(scope="module")
def cfg():
    return config()


def _truncar(f: dict, t: pd.Timestamp, encontro_id: str) -> dict:
    """Só o que existia antes de t (o próprio encontro é mantido para ser pontuado)."""
    g = dict(f)
    e = f["encontros"]
    g["encontros"] = e[(e.inicio < t) | (e.encontro_id == encontro_id)]
    g["avaliacoes"] = f["avaliacoes"][f["avaliacoes"].criado_em < t]
    g["seguidores"] = f["seguidores"][f["seguidores"].desde < t]
    g["interacoes"] = f["interacoes"][f["interacoes"].criado_em < t]
    g["status"] = f["status"][f["status"].ts <= t]
    return g


def test_sem_vazamento_do_futuro(fontes, cfg):
    e = fontes["encontros"]
    rng = np.random.default_rng(0)
    # sorteia encontros de várias semanas, incluindo pares que se repetem
    amostra = e.iloc[rng.choice(len(e), 30, replace=False)]
    completo_f = features.calcular(e, fontes, cfg).set_index("encontro_id")
    completo_c = compatibilidade.calcular(e, fontes["avaliacoes"], fontes["obras"], cfg).set_index("encontro_id")
    for _, enc in amostra.iterrows():
        g = _truncar(fontes, enc.inicio, enc.encontro_id)
        so = g["encontros"][g["encontros"].encontro_id == enc.encontro_id]
        tf = features.calcular(g["encontros"], g, cfg).set_index("encontro_id").loc[enc.encontro_id]
        tc = compatibilidade.calcular(so, g["avaliacoes"], fontes["obras"], cfg).set_index("encontro_id").loc[enc.encontro_id]
        pd.testing.assert_series_equal(completo_f.loc[enc.encontro_id], tf, check_names=False, check_dtype=False)
        pd.testing.assert_series_equal(completo_c.loc[enc.encontro_id], tc, check_names=False, check_dtype=False)


def test_label_bate_com_referencia(fontes, cfg):
    r, _ = label.calcular(fontes["encontros"], fontes["interacoes"], cfg)
    ref = pd.read_csv(PASTA_BOOTSTRAP / "rotulos_referencia.csv")
    m = r.merge(ref, on="encontro_id", suffixes=("", "_ref"))
    assert len(m) == len(ref)
    assert (m.conectou == m.conectou_ref).all()


def test_label_janela_48h(cfg):
    """Interação 47 h depois conta; 49 h depois não conta; antes do fim não conta."""
    t = pd.Timestamp("2026-08-01 20:00")
    enc = pd.DataFrame({"encontro_id": ["x", "y", "z"], "usuario_a": ["u1", "u3", "u5"], "usuario_b": ["u2", "u4", "u6"],
                        "inicio": [t] * 3, "fim": [t + pd.Timedelta(minutes=10)] * 3})
    fim = enc.fim[0]
    inter = pd.DataFrame({"origem": ["u2", "u3", "u5", "u9"], "destino": ["u1", "u4", "u6", "u9"],
                          "tipo": ["seguir", "curtir_review", "abrir_perfil", "seguir"],
                          "criado_em": [fim + pd.Timedelta(hours=47), fim + pd.Timedelta(hours=49),
                                        fim - pd.Timedelta(minutes=1), fim + pd.Timedelta(days=10)]})
    r, _ = label.calcular(enc, inter, cfg)
    assert r.set_index("encontro_id").conectou.to_dict() == {"x": 1, "y": 0, "z": 0}


def test_contar_e_estrito():
    ev_k, ev_t = ["a", "a", "a", "b"], pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01"])
    q = contar(ev_k, ev_t, ["a", "a", "b", "c"], pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-01", "2026-01-09"]))
    assert q.tolist() == [1, 3, 0, 0]   # evento no mesmo instante NÃO conta como "antes"
