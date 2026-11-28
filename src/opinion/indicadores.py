"""
Indicadores de produto (seção 3.2 do documento do projeto), calculados do banco operacional.

  * taxa de conversão de encontros: fração dos encontros cujo par é amigo na rede — o mesmo
    rótulo do modelo no experimento Last.fm (estático: amizades não têm data);
  * precisão dos sinais da pulseira: entre os encontros que receberam sinal, a fração que
    converteu. Calculada para o aviso de afinidade (na proximidade) e para o sinal do modelo
    (após o encontro), separadamente;
  * taxa de silenciamento (fadiga de notificação): fração dos sinais recebidos seguidos de uma
    troca da pulseira para o modo "pausado" em até 24 h.

Usado pela API (/indicadores) e, por ela, pelo dashboard.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from opinion.pipeline import label as label_mod
from opinion.pipeline.executar import config as config_pipeline

JANELA_SILENCIO = pd.Timedelta(hours=24)


def _df(banco, sql, cols):
    return pd.DataFrame(banco.todos(sql), columns=cols)


def carregar(banco) -> dict[str, pd.DataFrame]:
    d = {
        "encontros": _df(banco, "SELECT encontro_id, pulseira_id, usuario_a, usuario_b, inicio, duracao_s, local FROM encontros",
                         ["encontro_id", "pulseira_id", "usuario_a", "usuario_b", "inicio", "duracao_s", "local"]),
        "interacoes": _df(banco, "SELECT origem, destino, tipo, criado_em FROM interacoes",
                          ["origem", "destino", "tipo", "criado_em"]),
        "seguidores": _df(banco, "SELECT seguidor, seguido, desde FROM seguidores", ["seguidor", "seguido", "desde"]),
        "avisos": _df(banco, "SELECT encontro_id, pulseira_id, midias, inicio FROM avisos_proximidade",
                      ["encontro_id", "pulseira_id", "midias", "inicio"]),
        "status": _df(banco, "SELECT pulseira_id, usuario_id, modo, ts FROM status_pulseira",
                      ["pulseira_id", "usuario_id", "modo", "ts"]),
        "inferencias": _df(banco, "SELECT encontro_id, pulseira_id, inicio, fim, probabilidade, sinal, nivel, "
                                  "feedback_enviado, criado_em FROM inferencias",
                           ["encontro_id", "pulseira_id", "inicio", "fim", "probabilidade", "sinal", "nivel",
                            "feedback_enviado", "criado_em"]),
    }
    for nome, cols in (("encontros", ["inicio"]), ("interacoes", ["criado_em"]), ("avisos", ["inicio"]),
                       ("status", ["ts"]), ("inferencias", ["inicio", "fim", "criado_em"])):
        for c in cols:
            d[nome][c] = pd.to_datetime(d[nome][c], format="ISO8601")
    e = d["encontros"]
    e["duracao_s"] = e.duracao_s.astype(float)
    e["fim"] = e.inicio + pd.to_timedelta(e.duracao_s, unit="s")
    return d


def calcular(d: dict[str, pd.DataFrame]) -> dict:
    cfg = config_pipeline()
    e, av, st, inf = d["encontros"], d["avisos"], d["status"], d["inferencias"]
    out: dict = {"calculado_em": pd.Timestamp.now().isoformat(timespec="seconds")}
    if e.empty:
        return out | {"encontros": 0}
    amizade = cfg["label"].get("tipo", "interacao_48h") == "amizade"
    if amizade or not d["interacoes"].empty:
        rot, corte = label_mod.calcular(e, d["interacoes"], cfg, d.get("seguidores"))
    else:
        rot, corte = None, None
    if rot is not None:
        e = e.merge(rot[["encontro_id", "conectou", "label_disponivel"]], on="encontro_id", how="left")
    else:
        e = e.assign(conectou=np.nan, label_disponivel=0)
    com_label = e[e.label_disponivel == 1]
    out |= {"rotulo": "amizade" if amizade else "interacao_48h",
            "encontros": int(len(e)), "encontros_com_label": int(len(com_label)),
            "dados_de_interacao_ate": None if corte is None else str(corte),
            "taxa_conversao": _taxa(com_label.conectou)}
    out["conversao_por_local"] = {k: _taxa(g.conectou) for k, g in com_label.groupby("local")}

    # aviso de afinidade, emitido na proximidade
    a = av.merge(e[["encontro_id", "conectou", "label_disponivel"]], on="encontro_id", how="left")
    a_lab = a[a.label_disponivel == 1]
    out["aviso_afinidade"] = {"sinais": int(len(a)), "com_label": int(len(a_lab)), "precisao": _taxa(a_lab.conectou),
                              "por_midia": {m: int(a.midias.str.contains(m).sum())
                                            for m in ("filme", "serie", "livro", "jogo", "musica")}}
    # sinal do modelo, emitido após o encontro
    s = inf[inf.feedback_enviado == 1].merge(e[["encontro_id", "conectou", "label_disponivel"]], on="encontro_id", how="left")
    s_lab = s[s.label_disponivel == 1]
    out["sinal_modelo"] = {"encontros_pontuados": int(len(inf)), "sinais": int(len(s)), "com_label": int(len(s_lab)),
                           "precisao": _taxa(s_lab.conectou),
                           "taxa_de_sinal": _taxa(inf.feedback_enviado) if len(inf) else None,
                           "probabilidade_media": None if inf.empty else float(inf.probabilidade.mean())}

    # silenciamento: sinal seguido de pausa da mesma pulseira em até 24 h
    sinais = pd.concat([av[["pulseira_id", "inicio"]].rename(columns={"inicio": "ts"}),
                        inf.loc[inf.feedback_enviado == 1, ["pulseira_id", "criado_em"]].rename(columns={"criado_em": "ts"})],
                       ignore_index=True).dropna()
    pausas = st[st.modo == "pausado"][["pulseira_id", "ts"]].rename(columns={"ts": "pausa"}).dropna()
    if sinais.empty or pausas.empty:
        silenciados = 0
    else:
        sinais["pulseira_id"] = sinais.pulseira_id.astype(str)
        pausas["pulseira_id"] = pausas.pulseira_id.astype(str)
        sinais["ts"], pausas["pausa"] = pd.to_datetime(sinais.ts), pd.to_datetime(pausas.pausa)
        x = pd.merge_asof(sinais.sort_values("ts"), pausas.sort_values("pausa"), left_on="ts", right_on="pausa",
                          by="pulseira_id", direction="forward", tolerance=JANELA_SILENCIO)
        silenciados = int(x.pausa.notna().sum())
    ultimo = st.sort_values("ts").groupby("pulseira_id").modo.last() if not st.empty else pd.Series(dtype=str)
    out["silenciamento"] = {"sinais": int(len(sinais)), "seguidos_de_pausa_24h": silenciados,
                            "taxa": silenciados / len(sinais) if len(sinais) else None,
                            "trocas_para_pausado": int((st.modo == "pausado").sum()),
                            "pulseiras_pausadas_agora": int((ultimo == "pausado").sum()),
                            "pulseiras_com_status": int(len(ultimo))}
    por_dia = e.groupby(e.inicio.dt.date).agg(encontros=("encontro_id", "size"), conectou=("conectou", "mean"))
    avd = av.groupby(av.inicio.dt.date).size()
    por_dia["avisos"] = avd.reindex(por_dia.index).fillna(0).astype(int)
    out["por_dia"] = [{"dia": str(k), "encontros": int(v.encontros), "avisos": int(v.avisos),
                       "taxa_conversao": None if pd.isna(v.conectou) else float(v.conectou)}
                      for k, v in por_dia.iterrows()]
    return out


def _taxa(s) -> float | None:
    s = pd.Series(s).dropna()
    return float(s.mean()) if len(s) else None


class Indicadores:
    """Cache simples: recalcula no máximo a cada `ttl` segundos."""

    def __init__(self, banco, ttl: float = 60):
        self.banco, self.ttl = banco, ttl
        self._valor, self._em = None, 0.0

    def obter(self) -> dict:
        if self._valor is None or time.time() - self._em > self.ttl:
            self._valor, self._em = calcular(carregar(self.banco)), time.time()
        return self._valor
