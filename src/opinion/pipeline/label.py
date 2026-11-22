"""
Etapa 4 — Label (seção 4.4).

conectou = 1 se, em até 48 h após o FIM do encontro, pelo menos um dos dois seguiu o
outro, abriu o perfil do outro, curtiu/comentou uma review do outro ou salvou uma obra
avaliada pelo outro. Caso contrário, 0.

O label só existe quando as 48 h já passaram. O "corte" é o instante mais recente que
os dados cobrem (última interação registrada). Encontros cujo prazo ainda não venceu
ficam com label_disponivel = 0 e não entram no treino — no sistema real são os
encontros "de hoje", que o modelo só vai pontuar.
"""
from __future__ import annotations

import pandas as pd

from opinion.pipeline.tempo import chave_par, contar_na_janela


def calcular(encontros: pd.DataFrame, interacoes: pd.DataFrame, cfg: dict,
             seguidores: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if cfg["label"].get("tipo", "interacao_48h") == "amizade":
        return _por_amizade(encontros, seguidores)
    return _por_interacao(encontros, interacoes, cfg)


def _por_amizade(encontros: pd.DataFrame, seguidores: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """conectou = 1 se os dois já são amigos na rede (Last.fm). O rótulo é estático: sempre disponível."""
    if seguidores is None:
        raise ValueError("label.tipo = amizade exige a tabela de seguidores")
    e = encontros.reset_index(drop=True)
    amigos = set(chave_par(seguidores.seguidor, seguidores.seguido))
    par = chave_par(e.usuario_a, e.usuario_b)
    out = pd.DataFrame({
        "encontro_id": e.encontro_id,
        "n_interacoes_48h": 0,
        "label_disponivel": 1,
        "conectou": pd.Series(par).isin(amigos).astype(int),
    })
    return out, e.fim.max()


def _por_interacao(encontros: pd.DataFrame, interacoes: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Timestamp]:
    janela = pd.Timedelta(hours=cfg["label"]["janela_horas"])
    it = interacoes[interacoes.tipo.isin(cfg["label"]["tipos_interacao"])]
    corte = interacoes.criado_em.max()
    e = encontros.reset_index(drop=True)
    par = chave_par(e.usuario_a, e.usuario_b)
    n = contar_na_janela(chave_par(it.origem, it.destino), it.criado_em, par, e.fim, e.fim + janela)
    out = pd.DataFrame({
        "encontro_id": e.encontro_id,
        "n_interacoes_48h": n,
        "label_disponivel": ((e.fim + janela) <= corte).astype(int),
    })
    out["conectou"] = (out.n_interacoes_48h > 0).astype(int).where(out.label_disponivel == 1)
    return out, corte
