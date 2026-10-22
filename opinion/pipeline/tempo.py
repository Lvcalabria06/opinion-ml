"""
Utilitários "point-in-time": contar eventos de uma chave ANTES de um instante,
de forma vetorizada (sem laço por encontro).

Usados para garantir a regra da seção 4.4: toda feature usa apenas o que existia
até o momento do encontro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ns(t) -> np.ndarray:
    return pd.to_datetime(pd.Series(t)).to_numpy(dtype="datetime64[ns]").astype("int64")


def contar(ev_chave, ev_tempo, q_chave, q_tempo, incluir_igual: bool = False) -> np.ndarray:
    """
    Para cada consulta i: quantos eventos com chave == q_chave[i] e tempo < q_tempo[i]
    (ou <= se incluir_igual). Resolve tudo com uma ordenação + soma acumulada por chave.
    """
    n = len(q_chave)
    ev = pd.DataFrame({"k": np.asarray(ev_chave, dtype=object), "t": _ns(ev_tempo),
                       "e": 1, "o": -1 if incluir_igual else 1, "i": -1})
    q = pd.DataFrame({"k": np.asarray(q_chave, dtype=object), "t": _ns(q_tempo),
                      "e": 0, "o": 0, "i": np.arange(n)})
    tudo = pd.concat([ev, q], ignore_index=True).sort_values(["k", "t", "o"], kind="mergesort")
    tudo["c"] = tudo.groupby("k", sort=False)["e"].cumsum()
    r = tudo[tudo.e == 0]
    out = np.zeros(n, dtype=np.int64)
    out[r.i.to_numpy()] = r.c.to_numpy()
    return out


def contar_na_janela(ev_chave, ev_tempo, q_chave, t0, t1) -> np.ndarray:
    """Eventos com tempo em (t0, t1]."""
    return contar(ev_chave, ev_tempo, q_chave, t1, incluir_igual=True) - \
        contar(ev_chave, ev_tempo, q_chave, t0, incluir_igual=True)


def ultimo_valor(ev_chave, ev_tempo, ev_valor, q_chave, q_tempo, padrao=None) -> np.ndarray:
    """Último valor registrado para a chave até o instante (inclusive)."""
    n = len(q_chave)
    q = pd.DataFrame({"k": np.asarray(q_chave, dtype=object), "t": pd.to_datetime(pd.Series(q_tempo)).to_numpy(),
                      "i": np.arange(n)}).sort_values("t")
    ev = pd.DataFrame({"k": np.asarray(ev_chave, dtype=object), "t": pd.to_datetime(pd.Series(ev_tempo)).to_numpy(),
                       "v": np.asarray(ev_valor, dtype=object)}).sort_values("t")
    m = pd.merge_asof(q, ev, on="t", by="k", direction="backward")
    out = np.full(n, padrao, dtype=object)
    out[m.i.to_numpy()] = m.v.where(m.v.notna(), padrao).to_numpy()
    return out


def chave_par(a, b) -> np.ndarray:
    """Identificador do par sem ordem: 'usr-00001|usr-00002' == 'usr-00002|usr-00001'."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    return np.where(a < b, a + "|" + b, b + "|" + a)
