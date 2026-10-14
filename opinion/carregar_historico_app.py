"""
Carrega no banco o HISTÓRICO DO APP gerado pelo bootstrap: usuários (com a pulseira
de cada um), obras, avaliações, seguidores e interações.

No sistema real esses dados já estariam no banco do aplicativo; aqui eles vêm do
simulador. Os dados da pulseira NÃO são carregados por aqui — eles entram pelo
MQTT (pulseira virtual -> broker -> serviço de ingestão).

    python -m opinion.carregar_historico_app
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opinion.db import Banco
from opinion.mundo import PASTA_BOOTSTRAP

TABELAS = [
    ("usuarios", "usuarios.csv", ["usuario_id", "pulseira_id", "data_cadastro"]),
    ("obras", "obras.csv", ["obra_id", "titulo", "midia", "generos", "popularidade"]),
    ("avaliacoes", "avaliacoes.csv", ["avaliacao_id", "usuario_id", "obra_id", "nota", "criado_em"]),
    ("seguidores", "seguidores.csv", ["seguidor", "seguido", "desde"]),
    ("interacoes", "interacoes.csv", ["interacao_id", "origem", "destino", "tipo", "obra_id", "avaliacao_id", "criado_em"]),
]


def carregar(banco: Banco, pasta: Path):
    banco.criar_esquema()
    for tabela, arquivo, cols in TABELAS:
        existentes = banco.um(f"SELECT COUNT(*) FROM {tabela}")[0]
        df = pd.read_csv(pasta / arquivo)[cols]
        if existentes >= len(df):
            print(f"  {tabela:<11} já carregada ({existentes:,} linhas)")
            continue
        df = df.astype(object).where(df.notna(), None)  # tipos nativos do Python; vazio -> NULL
        linhas = list(df.itertuples(index=False, name=None))
        marcadores = ",".join("?" * len(cols))
        banco.executar_varios(f"INSERT INTO {tabela} ({','.join(cols)}) VALUES ({marcadores}) ON CONFLICT DO NOTHING", linhas)
        print(f"  {tabela:<11} {len(linhas):>8,} linhas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default=str(PASTA_BOOTSTRAP / "historico_app"))
    a = ap.parse_args()
    b = Banco()
    print(f"Carregando histórico do app em {b.url}")
    carregar(b, Path(a.dados))


if __name__ == "__main__":
    main()
