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
  compat_score_<midia>  pontuação 0–1 por mídia, usada no aviso por categoria
  compat_score          máximo dos scores por mídia; referência do baseline geral
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opinion.pipeline.tempo import contar

MIDIAS = ("filme", "serie", "livro", "jogo", "musica")
INDICADORES = ("historico_comum", "compat_score", "match")
COLUNAS_MIDIA = [f"{campo}_{midia}" for midia in MIDIAS for campo in INDICADORES]

def _matriz_generos(obras: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    generos_item = obras.generos.fillna("")
    generos = sorted({g for gs in generos_item for g in gs.split("|") if g and not g.startswith("outros_")})
    vetores = {gs: np.isin(generos, gs.split("|")).astype(float) for gs in generos_item.unique()}
    return generos, dict(zip(obras.obra_id, generos_item.map(vetores)))


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


def _categoria(encontros: pd.DataFrame, avaliacoes: pd.DataFrame, obras: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    generos, vet = _matriz_generos(obras)
    va = _vetores_ate(encontros.usuario_a, encontros.inicio, avaliacoes, vet, len(generos))
    vb = _vetores_ate(encontros.usuario_b, encontros.inicio, avaliacoes, vet, len(generos))
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where((na > 0) & (nb > 0), (va * vb).sum(1) / (na * nb), 0.0)

    def top3(v):  # máscara dos 3 gêneros de maior peso positivo
        m = np.zeros_like(v, dtype=bool)
        idx = np.argsort(-v, axis=1, kind="stable")[:, :3]
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


def motivos_do_sinal(linha, cfg: dict) -> list[str]:
    """Cada mídia precisa, sozinha, atingir o corte de afinidade do protótipo."""
    corte = cfg["compat_score"]["afinidade_alta"]
    return [m for m in MIDIAS if linha[f"match_{m}"] and linha[f"compat_score_{m}"] >= corte]


def calcular(encontros: pd.DataFrame, avaliacoes: pd.DataFrame, obras: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Um encontro, cinco comparações independentes. Ausência de histórico não gera afinidade."""
    out = encontros[["encontro_id"]].reset_index(drop=True).copy()
    out["sim_generos"] = 0.0
    out["generos_top3_comum"] = 0
    out["n_obras_comuns"] = 0
    out["favoritas_comuns"] = 0
    soma_notas = np.zeros(len(out))
    for midia in MIDIAS:
        catalogo = obras[obras.midia == midia]
        notas = avaliacoes[avaliacoes.obra_id.isin(catalogo.obra_id)]
        historico = f"historico_comum_{midia}"
        score = f"compat_score_{midia}"
        match = f"match_{midia}"
        if notas.empty:
            out[historico], out[score], out[match] = 0, 0.0, 0
            continue
        por_usuario = notas.groupby("usuario_id").criado_em.min()
        elegivel = ((encontros.usuario_a.map(por_usuario) < encontros.inicio)
                    & (encontros.usuario_b.map(por_usuario) < encontros.inicio)).to_numpy()
        out[historico] = elegivel.astype(int)
        out[score], out[match] = 0.0, 0
        if not elegivel.any():
            continue
        c = _categoria(encontros[elegivel], notas, catalogo, cfg).set_index("encontro_id")
        c = c.reindex(out.encontro_id)
        c.index = out.index
        tem_genero = (c.generos_top3_comum > 0) & (c.sim_generos > 0)
        combina = tem_genero | (c.favoritas_comuns > 0)
        out[match] = combina.astype(int)
        out[score] = c.compat_score.where(combina, 0.0)
        out["sim_generos"] = np.maximum(out.sim_generos, c.sim_generos.fillna(0))
        out["generos_top3_comum"] = np.maximum(out.generos_top3_comum, c.generos_top3_comum.fillna(0).astype(int))
        comuns = c.n_obras_comuns.fillna(0).astype(int)
        out["n_obras_comuns"] += comuns
        out["favoritas_comuns"] += c.favoritas_comuns.fillna(0).astype(int)
        soma_notas += c.sim_notas.fillna(0) * comuns
    out["sim_notas"] = soma_notas / out.n_obras_comuns.replace(0, np.nan)
    out["compat_score"] = out[[f"compat_score_{m}" for m in MIDIAS]].max(axis=1)
    return out
