"""
Etapa 5 — Preparação do dataset do modelo.

  * junta compatibilidade + features + label
  * marca o split temporal (treino 1–8 / validação 9–10 / teste 11–12) e a purga da fronteira
  * imputação simples (com indicador) e one-hot das categóricas
  * checagens de qualidade (a DAG falha se algo estiver fora do esperado)
  * grava:
      data/processed/features_encontros.parquet  todos os encontros (inclusive sem label ainda)
      data/processed/dataset_modelo.parquet      só os rotulados e dentro do período do split
      data/processed/metadados.json              lista de features, tamanhos, taxas, corte

Padronização (média/desvio) NÃO é feita aqui de propósito: ela precisa ser ajustada só no
treino de cada fold, dentro do pipeline do modelo (CC3 em diante), para não vazar informação.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAS = ["local", "periodo"]
NUMERICAS = [
    # encontro
    "log_duracao", "rssi_medio", "n_encontros_anteriores_par", "horas_desde_ultimo_encontro_par",
    "primeiro_encontro_par",
    # contexto
    "dia_semana", "hora", "fim_de_semana", "algum_modo_discreto",
    # compatibilidade
    "sim_generos", "generos_top3_comum", "n_obras_comuns", "sim_notas", "tem_obras_comuns", "favoritas_comuns",
    "compat_score",
    # perfil
    "atividade_min", "atividade_max", "historico_min", "historico_max", "conexoes_min", "conexoes_max",
    # relação
    "algum_segue", "seguem_mutuamente", "interacoes_anteriores_par", "interagiram_antes",
]
GRUPOS = {
    "encontro": NUMERICAS[0:5], "contexto": NUMERICAS[5:9] + ["local", "periodo"],
    "compatibilidade": NUMERICAS[9:16], "perfil": NUMERICAS[16:22], "relacao": NUMERICAS[22:26],
}


def marcar_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    s = pd.Series("fora_do_periodo", index=df.index, dtype=object)
    for nome in ("treino", "validacao", "teste"):
        ini, fim = cfg["split"][nome]
        s[(df.inicio >= pd.Timestamp(ini)) & (df.inicio < pd.Timestamp(fim))] = nome
    if cfg["split"].get("purgar_fronteira"):
        fim_treino = pd.Timestamp(cfg["split"]["treino"][1])
        janela = pd.Timedelta(hours=cfg["label"]["janela_horas"])
        s[(s == "treino") & (df.fim + janela > fim_treino)] = "purgado"
    return s


def montar(encontros: pd.DataFrame, compat: pd.DataFrame, feats: pd.DataFrame, rot: pd.DataFrame,
           cfg: dict) -> pd.DataFrame:
    base = encontros[["encontro_id", "usuario_a", "usuario_b", "inicio", "fim"]]
    df = base.merge(feats, on="encontro_id").merge(compat, on="encontro_id").merge(rot, on="encontro_id")
    inicio0 = pd.Timestamp(cfg["split"]["treino"][0])
    df["semana"] = ((df.inicio - inicio0).dt.days // 7 + 1).astype(int)
    df["split"] = marcar_split(df, cfg)
    return completar(df).sort_values("inicio").reset_index(drop=True)


def completar(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas derivadas e imputação — as mesmas no treino e na pontuação de encontros novos."""
    df["tem_obras_comuns"] = (df.n_obras_comuns > 0).astype(int)
    df["sim_notas"] = df.sim_notas.fillna(0.5)     # neutro quando não há obra em comum (+ indicador acima)
    return df


def codificar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(df[CATEGORICAS], prefix=CATEGORICAS, dtype=int)
    X = pd.concat([df[NUMERICAS], dummies], axis=1)
    return X, list(X.columns)


def matriz_features(df: pd.DataFrame, colunas_x: list[str]) -> pd.DataFrame:
    """X no formato exato do treino (mesmas colunas e ordem). Categoria ausente no lote vira 0."""
    X, _ = codificar(df)
    return X.reindex(columns=colunas_x, fill_value=0)


def checar_qualidade(dm: pd.DataFrame, colunas_x: list[str], cfg: dict) -> list[str]:
    q, erros = cfg["qualidade"], []
    nulos = dm[colunas_x].isna().sum()
    if nulos.any():
        erros.append(f"features com valores nulos: {nulos[nulos > 0].to_dict()}")
    if dm.encontro_id.duplicated().any():
        erros.append("encontro_id duplicado no dataset")
    taxa = dm.conectou.mean()
    if not q["taxa_conexao_min"] <= taxa <= q["taxa_conexao_max"]:
        erros.append(f"taxa de conexão {taxa:.3f} fora da faixa esperada")
    for s in ("treino", "validacao", "teste"):
        n = (dm.split == s).sum()
        if n < q["min_encontros_por_split"]:
            erros.append(f"split {s} com só {n} encontros")
        elif dm.loc[dm.split == s, "conectou"].nunique() < 2:
            erros.append(f"split {s} tem uma única classe")
    if not np.isfinite(dm[colunas_x].to_numpy(dtype=float)).all():
        erros.append("valores infinitos nas features")
    return erros


def executar(encontros, compat, feats, rot, corte, cfg, pasta_saida: Path, referencia: Path | None = None) -> dict:
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    df = montar(encontros, compat, feats, rot, cfg)
    df.to_parquet(pasta_saida / "features_encontros.parquet", index=False)

    dm = df[(df.label_disponivel == 1) & df.split.isin(["treino", "validacao", "teste"])].copy()
    dm["conectou"] = dm.conectou.astype(int)
    X, colunas_x = codificar(dm)
    ident = ["encontro_id", "par_id", "usuario_a", "usuario_b", "inicio", "fim", "semana", "split", "local"]
    saida = pd.concat([dm[ident].rename(columns={"local": "local_original"}), X, dm[["conectou"]]], axis=1)

    erros = checar_qualidade(saida, colunas_x, cfg)
    concordancia = None
    if referencia and Path(referencia).exists():   # só existe com dados simulados
        ref = pd.read_csv(referencia)
        m = df[df.label_disponivel == 1][["encontro_id", "conectou"]].merge(ref, on="encontro_id", suffixes=("", "_ref"))
        concordancia = float((m.conectou == m.conectou_ref).mean()) if len(m) else None
        if concordancia is not None and concordancia < 0.999:
            erros.append(f"label difere da referência da simulação em {(1 - concordancia):.2%} dos encontros")
    if erros:
        raise ValueError("Checagem de qualidade falhou: " + " | ".join(erros))

    saida.to_parquet(pasta_saida / "dataset_modelo.parquet", index=False)
    por_split = saida.groupby("split").agg(encontros=("encontro_id", "size"), taxa_conexao=("conectou", "mean"),
                                           pares=("par_id", "nunique"))
    meta = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "corte_dos_dados": str(corte),
        "encontros_total": int(len(df)),
        "encontros_sem_label_ainda": int((df.label_disponivel == 0).sum()),
        "encontros_fora_do_periodo": int((df.split == "fora_do_periodo").sum()),
        "encontros_purgados_fronteira": int((df.split == "purgado").sum()),
        "dataset_modelo": int(len(saida)),
        "taxa_conexao": round(float(saida.conectou.mean()), 4),
        "por_split": {k: {"encontros": int(v.encontros), "taxa_conexao": round(float(v.taxa_conexao), 4),
                          "pares": int(v.pares)} for k, v in por_split.iterrows()},
        "colunas_features": colunas_x,
        "grupos_features": GRUPOS,
        "coluna_label": "conectou",
        "coluna_grupo_cv": "par_id",
        "concordancia_label_referencia": concordancia,
    }
    (pasta_saida / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
