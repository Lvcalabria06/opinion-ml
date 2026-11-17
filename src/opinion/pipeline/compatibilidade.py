"""
Etapa 2 — Compatibilidade entre os dois usuários de cada encontro (seção 4.4).

Usa SOMENTE as avaliações criadas antes do início do encontro.

Colunas geradas
  sim_generos           cosseno entre os "vetores de gosto" dos dois (-1 a 1). O vetor de
                        cada usuário soma, por gênero, (nota - 3) das obras que avaliou:
                        5 estrelas puxa a favor do gênero, 1 estrela puxa contra.
  generos_top3_comum    quantos dos 3 gêneros preferidos os dois têm em comum (0–3)
  n_obras_comuns        obras que os dois já avaliaram
  sim_notas             concordância das notas nas mesmas obras: 1 - |diferença média| / 4
                        (vazio se não há obra em comum)
  favoritas_comuns      obras que os dois avaliaram com nota >= 4
  compat_score          pontuação única 0–1, usada pela REGRA SIMPLES de referência
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opinion.pipeline.tempo import contar


def _matriz_generos(obras: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    generos = sorted({g for gs in obras.generos.dropna() for g in gs.split("|")})
    return generos, {o: np.isin(generos, gs.split("|")).astype(float) for o, gs in zip(obras.obra_id, obras.generos)}


def _vetores_ate(enc_usuario, enc_inicio, aval: pd.DataFrame, vet_obra: dict, n_gen: int) -> np.ndarray:
    """Vetor de gosto de cada usuário considerando só avaliações anteriores ao instante."""
    a = aval.sort_values(["usuario_id", "criado_em"], kind="mergesort").reset_index(drop=True)
    pesos = (a.nota.to_numpy() - 3)[:, None] * np.stack([vet_obra[o] for o in a.obra_id])
    acum = np.zeros_like(pesos)
    for _, idx in a.groupby("usuario_id", sort=False).indices.items():   # soma acumulada por usuário
        acum[idx] = np.cumsum(pesos[idx], axis=0)
    inicio_usuario = a.groupby("usuario_id", sort=False).indices
    offset = {u: idx[0] for u, idx in inicio_usuario.items()}
    k = contar(a.usuario_id, a.criado_em, enc_usuario, enc_inicio)       # nº de avaliações antes
    out = np.zeros((len(k), n_gen))
    for i, (u, kk) in enumerate(zip(enc_usuario, k)):
        if kk > 0:
            out[i] = acum[offset[u] + kk - 1]
    return out


def _obras_em_comum(enc: pd.DataFrame, aval: pd.DataFrame, nota_fav: int) -> pd.DataFrame:
    av = aval[["usuario_id", "obra_id", "nota", "criado_em"]]
    lados = []
    for lado in ("a", "b"):
        m = enc[["encontro_id", f"usuario_{lado}", "inicio"]].merge(
            av, left_on=f"usuario_{lado}", right_on="usuario_id")
        m = m[m.criado_em < m.inicio]
        lados.append(m[["encontro_id", "obra_id", "nota"]].rename(columns={"nota": f"nota_{lado}"}))
    c = lados[0].merge(lados[1], on=["encontro_id", "obra_id"])
    c["dif"] = (c.nota_a - c.nota_b).abs()
    c["fav"] = (c.nota_a >= nota_fav) & (c.nota_b >= nota_fav)
    g = c.groupby("encontro_id").agg(n_obras_comuns=("obra_id", "size"), dif_media=("dif", "mean"),
                                     favoritas_comuns=("fav", "sum"))
    g["sim_notas"] = 1 - g.dif_media / 4
    return g.drop(columns="dif_media")


def calcular(encontros: pd.DataFrame, avaliacoes: pd.DataFrame, obras: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    generos, vet = _matriz_generos(obras)
    va = _vetores_ate(encontros.usuario_a, encontros.inicio, avaliacoes, vet, len(generos))
    vb = _vetores_ate(encontros.usuario_b, encontros.inicio, avaliacoes, vet, len(generos))
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where((na > 0) & (nb > 0), (va * vb).sum(1) / (na * nb), 0.0)

    def top3(v):  # máscara dos 3 gêneros de maior peso positivo
        m = np.zeros_like(v, dtype=bool)
        idx = np.argsort(-v, axis=1)[:, :3]
        np.put_along_axis(m, idx, True, axis=1)
        return m & (v > 0)

    out = pd.DataFrame({
        "encontro_id": encontros.encontro_id.to_numpy(),
        "sim_generos": np.round(sim, 4),
        "generos_top3_comum": (top3(va) & top3(vb)).sum(1),
    })
    comum = _obras_em_comum(encontros, avaliacoes, cfg["features"]["nota_favorita"])
    out = out.merge(comum, on="encontro_id", how="left")
    out["n_obras_comuns"] = out.n_obras_comuns.fillna(0).astype(int)
    out["favoritas_comuns"] = out.favoritas_comuns.fillna(0).astype(int)
    out["sim_notas"] = out.sim_notas.round(4)

    p = cfg["compat_score"]
    out["compat_score"] = (p["peso_generos"] * (out.sim_generos + 1) / 2
                           + p["peso_notas"] * out.sim_notas.fillna(0.5)
                           + p["peso_favoritas"] * np.minimum(out.favoritas_comuns, 3) / 3).round(4)
    return out
