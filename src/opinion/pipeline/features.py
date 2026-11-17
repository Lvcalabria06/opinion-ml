"""
Etapa 3 — Features do encontro (seção 4.4). Tudo calculado "até o instante do início".

Como usuario_a/usuario_b é uma ordem arbitrária (quem publicou), as features de perfil
são simétricas: guardamos o MÍNIMO e o MÁXIMO dos dois.

  Encontro      duracao_s, log_duracao, rssi_medio,
                n_encontros_anteriores_par, horas_desde_ultimo_encontro_par, primeiro_encontro_par
  Contexto      local, dia_semana, hora, fim_de_semana, periodo, algum_modo_discreto
  Perfil        atividade_min/max   (avaliações nos últimos 28 dias)
                historico_min/max   (avaliações desde sempre)
                conexoes_min/max    (seguidores + seguindo)
  Relação       algum_segue, seguem_mutuamente, interacoes_anteriores_par, interagiram_antes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opinion.pipeline.tempo import chave_par, contar, ultimo_valor


def _por_usuario(enc: pd.DataFrame, fn) -> tuple[np.ndarray, np.ndarray]:
    """Aplica fn(usuarios, instantes) aos dois lados e devolve (mínimo, máximo)."""
    a, b = fn(enc.usuario_a, enc.inicio), fn(enc.usuario_b, enc.inicio)
    return np.minimum(a, b), np.maximum(a, b)


def calcular(encontros: pd.DataFrame, f: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    e = encontros.reset_index(drop=True)
    par = chave_par(e.usuario_a, e.usuario_b)
    out = pd.DataFrame({"encontro_id": e.encontro_id, "par_id": par})

    # ------------------------------------------------------------ encontro
    out["duracao_s"] = e.duracao_s.astype(int)
    out["log_duracao"] = np.log(e.duracao_s.clip(lower=1)).round(4)
    out["rssi_medio"] = e.rssi_medio.astype(float)
    out["n_encontros_anteriores_par"] = contar(par, e.inicio, par, e.inicio)
    # último encontro do mesmo par que TERMINOU antes deste começar
    ordem = pd.DataFrame({"par": par, "inicio": e.inicio, "fim": e.fim, "i": np.arange(len(e))})
    ult = pd.merge_asof(ordem.sort_values("inicio"), ordem[["par", "fim"]].rename(columns={"fim": "fim_ant"})
                        .assign(t=lambda d: d.fim_ant).sort_values("t"),
                        left_on="inicio", right_on="t", by="par", direction="backward", allow_exact_matches=False)
    horas = np.full(len(e), np.nan)
    horas[ult.i.to_numpy()] = ((ult.inicio - ult.fim_ant).dt.total_seconds() / 3600).to_numpy()
    out["primeiro_encontro_par"] = np.isnan(horas).astype(int)
    out["horas_desde_ultimo_encontro_par"] = np.round(np.nan_to_num(horas, nan=24 * 7 * 12), 2)

    # ------------------------------------------------------------ contexto
    out["local"] = e.local
    out["dia_semana"] = e.inicio.dt.dayofweek
    out["hora"] = e.inicio.dt.hour
    out["fim_de_semana"] = (out.dia_semana >= 5).astype(int)
    out["periodo"] = pd.cut(out.hora, [-1, 11, 17, 23], labels=["manha", "tarde", "noite"]).astype(str)
    st = f["status"]
    modo_a = ultimo_valor(st.usuario_id, st.ts, st.modo, e.usuario_a, e.inicio, padrao="aberto")
    modo_b = ultimo_valor(st.usuario_id, st.ts, st.modo, e.usuario_b, e.inicio, padrao="aberto")
    out["algum_modo_discreto"] = ((modo_a == "discreto") | (modo_b == "discreto")).astype(int)

    # -------------------------------------------------------------- perfil
    av = f["avaliacoes"]
    janela = pd.Timedelta(days=cfg["features"]["janela_atividade_dias"])
    hist = lambda u, t: contar(av.usuario_id, av.criado_em, u, t)
    ativ = lambda u, t: hist(u, t) - contar(av.usuario_id, av.criado_em, u, t - janela)
    sg = f["seguidores"]
    conex = lambda u, t: contar(sg.seguido, sg.desde, u, t) + contar(sg.seguidor, sg.desde, u, t)
    out["atividade_min"], out["atividade_max"] = _por_usuario(e, ativ)
    out["historico_min"], out["historico_max"] = _por_usuario(e, hist)
    out["conexoes_min"], out["conexoes_max"] = _por_usuario(e, conex)

    # -------------------------------------------------------------- relação
    seg = sg.assign(k=sg.seguidor + ">" + sg.seguido)
    a_b = contar(seg.k, seg.desde, e.usuario_a + ">" + e.usuario_b, e.inicio) > 0
    b_a = contar(seg.k, seg.desde, e.usuario_b + ">" + e.usuario_a, e.inicio) > 0
    out["algum_segue"] = (a_b | b_a).astype(int)
    out["seguem_mutuamente"] = (a_b & b_a).astype(int)
    it = f["interacoes"]
    it_par = chave_par(it.origem, it.destino)
    out["interacoes_anteriores_par"] = contar(it_par, it.criado_em, par, e.inicio)
    out["interagiram_antes"] = (out.interacoes_anteriores_par > 0).astype(int)
    return out
