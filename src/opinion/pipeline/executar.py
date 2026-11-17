"""
Funções de cada etapa do pipeline, no formato que a DAG do Airflow chama
(cada uma lê do staging e grava o seu resultado lá), e um executor de linha de
comando para rodar tudo sem Airflow.

    python -m opinion.pipeline.executar                    # fonte = banco (DATABASE_URL)
    python -m opinion.pipeline.executar --fonte arquivos   # direto do data/bootstrap
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from opinion.mundo import PASTA_BOOTSTRAP
from opinion.pipeline import CONFIG_PIPELINE, PASTA_PROCESSADOS, PASTA_STAGING
from opinion.pipeline import compatibilidade, features, ingestao, label, preparacao


def config() -> dict:
    return yaml.safe_load(Path(CONFIG_PIPELINE).read_text(encoding="utf-8"))


def etapa_ingestao(staging=PASTA_STAGING, fonte="banco", pasta_bootstrap=None) -> dict:
    return ingestao.executar(Path(staging), fonte, pasta_bootstrap)


def etapa_compatibilidade(staging=PASTA_STAGING) -> dict:
    f = ingestao.ler_staging(staging)
    c = compatibilidade.calcular(f["encontros"], f["avaliacoes"], f["obras"], config())
    c.to_parquet(Path(staging) / "compatibilidade.parquet", index=False)
    return {"encontros": len(c), "com_obras_em_comum": int((c.n_obras_comuns > 0).sum()),
            "compat_score_medio": round(float(c.compat_score.mean()), 4)}


def etapa_features(staging=PASTA_STAGING) -> dict:
    f = ingestao.ler_staging(staging)
    x = features.calcular(f["encontros"], f, config())
    x.to_parquet(Path(staging) / "features_base.parquet", index=False)
    return {"encontros": len(x), "colunas": len(x.columns) - 2}


def etapa_label(staging=PASTA_STAGING) -> dict:
    f = ingestao.ler_staging(staging)
    r, corte = label.calcular(f["encontros"], f["interacoes"], config())
    r.to_parquet(Path(staging) / "label.parquet", index=False)
    (Path(staging) / "_corte.json").write_text(json.dumps({"corte": str(corte)}))
    return {"rotulados": int(r.label_disponivel.sum()), "pendentes": int((r.label_disponivel == 0).sum()),
            "taxa_conexao": round(float(r.conectou.mean()), 4), "corte": str(corte)}


def etapa_preparacao(staging=PASTA_STAGING, saida=PASTA_PROCESSADOS, referencia=None) -> dict:
    s = Path(staging)
    f = ingestao.ler_staging(s)
    corte = pd.Timestamp(json.loads((s / "_corte.json").read_text())["corte"])
    ref = referencia if referencia is not None else PASTA_BOOTSTRAP / "rotulos_referencia.csv"
    return preparacao.executar(f["encontros"], pd.read_parquet(s / "compatibilidade.parquet"),
                               pd.read_parquet(s / "features_base.parquet"), pd.read_parquet(s / "label.parquet"),
                               corte, config(), Path(saida), ref)


def rodar_tudo(fonte="banco", staging=PASTA_STAGING, saida=PASTA_PROCESSADOS, pasta_bootstrap=None,
               referencia=None, verbose=True) -> dict:
    etapas = [("ingestao", lambda: etapa_ingestao(staging, fonte, pasta_bootstrap)),
              ("compatibilidade", lambda: etapa_compatibilidade(staging)),
              ("features", lambda: etapa_features(staging)),
              ("label", lambda: etapa_label(staging)),
              ("preparacao", lambda: etapa_preparacao(staging, saida, referencia))]
    resultado = {}
    for nome, fn in etapas:
        t = time.time()
        resultado[nome] = fn()
        if verbose:
            r = resultado[nome]
            resumo = {k: v for k, v in r.items() if k not in ("colunas_features", "grupos_features", "linhas", "por_split")}
            print(f"[{nome:<15}] {time.time() - t:5.1f} s  {resumo}")
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Pipeline de dados do OpiniOn (sem Airflow)")
    ap.add_argument("--fonte", choices=["banco", "arquivos"], default="banco")
    ap.add_argument("--bootstrap", help="pasta do bootstrap (fonte=arquivos)")
    a = ap.parse_args()
    r = rodar_tudo(a.fonte, pasta_bootstrap=a.bootstrap)
    print("\nDataset do modelo:", json.dumps(r["preparacao"]["por_split"], ensure_ascii=False))
    print(f"Arquivos em {PASTA_PROCESSADOS}")


if __name__ == "__main__":
    main()
