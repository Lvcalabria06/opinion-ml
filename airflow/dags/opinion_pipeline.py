"""
DAG opinion_pipeline — CC2

    ingestao ──┬── compatibilidade ──┐
               ├── features ─────────┼── preparacao ── resumo
               └── label ────────────┘

ingestao         extrai do banco operacional (encontros recebidos via MQTT + histórico do app)
                 para data/staging/, com checagens de consistência
compatibilidade  similaridade de gosto entre os dois usuários, só com avaliações anteriores
features         encontro, contexto, perfil e relação, "até o instante do encontro"
label            conectou = interação entre os dois em até 48 h após o fim
preparacao       split temporal + purga, imputação, one-hot, checagens de qualidade,
                 grava data/processed/dataset_modelo.parquet (entrada do CC3)
resumo           registra no log o tamanho e a taxa de conexão de cada split

Roda todo dia (encontros novos chegam o tempo todo pela pulseira). Para disparar na mão:
botão "Trigger" na interface, ou  airflow dags test opinion_pipeline
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# permite importar o pacote `opinion` quando a pasta do projeto não está no PYTHONPATH
RAIZ_PROJETO = Path(os.getenv("OPINION_HOME", Path(__file__).resolve().parents[2]))
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

try:                                    # Airflow 3
    from airflow.sdk import dag, task
except ImportError:                     # Airflow 2.x
    from airflow.decorators import dag, task

FONTE = os.getenv("OPINION_FONTE", "banco")   # "banco" (padrão) ou "arquivos" (direto do bootstrap)


@dag(
    dag_id="opinion_pipeline",
    description="OpiniOn — encontros da pulseira -> features sem informação do futuro -> dataset do modelo",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "cc-opinion", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["opinion", "cc2", "ml"],
)
def opinion_pipeline():

    @task
    def ingestao() -> dict:
        from opinion.pipeline.executar import etapa_ingestao
        r = etapa_ingestao(fonte=FONTE)
        print(f"Ingestão: {r['linhas']}")
        return {k: v for k, v in r.items() if k != "linhas"}

    @task
    def compatibilidade(_: dict) -> dict:
        from opinion.pipeline.executar import etapa_compatibilidade
        return etapa_compatibilidade()

    @task
    def features(_: dict) -> dict:
        from opinion.pipeline.executar import etapa_features
        return etapa_features()

    @task
    def label(_: dict) -> dict:
        from opinion.pipeline.executar import etapa_label
        return etapa_label()

    @task
    def preparacao(c: dict, f: dict, lb: dict) -> dict:
        from opinion.pipeline.executar import etapa_preparacao
        meta = etapa_preparacao()
        return {k: meta[k] for k in ("dataset_modelo", "taxa_conexao", "por_split", "corte_dos_dados",
                                     "encontros_sem_label_ainda", "encontros_purgados_fronteira",
                                     "concordancia_label_referencia")}

    @task
    def resumo(meta: dict) -> None:
        print("=" * 60)
        print(f"Dataset do modelo: {meta['dataset_modelo']:,} encontros | taxa de conexão {meta['taxa_conexao']:.1%}")
        for s, v in meta["por_split"].items():
            print(f"  {s:<10} {v['encontros']:>7,} encontros  {v['taxa_conexao']:.1%} conectou  {v['pares']:,} pares")
        print(f"Aguardando 48 h para ter label: {meta['encontros_sem_label_ainda']:,} encontros")
        print(f"Removidos na fronteira treino/validação: {meta['encontros_purgados_fronteira']:,}")
        print("=" * 60)

    base = ingestao()
    resumo(preparacao(compatibilidade(base), features(base), label(base)))


opinion_pipeline()
