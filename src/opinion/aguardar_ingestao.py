"""
Espera a ingestão terminar de gravar o replay antes de liberar o pipeline (Docker Compose).

A pulseira virtual publica mais rápido do que a ingestão grava: quando o replay termina, ainda há
mensagens na fila do broker. Sem esta espera, o Airflow começaria com o banco pela metade.

    python -m opinion.aguardar_ingestao              # até o banco ter todos os encontros do bootstrap
    python -m opinion.aguardar_ingestao --limite-s 1800

Termina com sucesso quando o banco tem todos os encontros do bootstrap ou quando a contagem fica
parada por `--estavel-s` segundos (caso alguma mensagem tenha sido rejeitada de propósito).
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from opinion.db import Banco
from opinion.mundo import PASTA_BOOTSTRAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite-s", type=int, default=1800)
    ap.add_argument("--estavel-s", type=int, default=60)
    a = ap.parse_args()
    esperado = len(pd.read_csv(PASTA_BOOTSTRAP / "pulseira" / "encontros.csv", usecols=["encontro_id"]))
    banco = Banco()
    inicio, ultimo, parado_desde = time.time(), -1, time.time()
    while time.time() - inicio < a.limite_s:
        n = banco.um("SELECT COUNT(*) FROM encontros")[0]
        if n >= esperado:
            print(f"Ingestão completa: {n:,} encontros no banco.")
            return
        if n != ultimo:
            ultimo, parado_desde = n, time.time()
        elif time.time() - parado_desde > a.estavel_s:
            print(f"Contagem parada em {n:,} de {esperado:,} encontros há {a.estavel_s} s; seguindo.")
            return
        print(f"Aguardando a ingestão: {n:,} de {esperado:,} encontros...", flush=True)
        time.sleep(5)
    sys.exit("A ingestão não terminou dentro do limite.")


if __name__ == "__main__":
    main()
