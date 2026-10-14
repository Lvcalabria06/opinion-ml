"""
Confere se o replay do bootstrap chegou inteiro e correto no banco:
  * quantidade de encontros, interesses e trocas de modo: arquivo vs. banco
  * os IDs rotativos foram traduzidos para os usuários CERTOS (compara com a verdade da simulação)

    python -m opinion.verificar_replay
"""
from __future__ import annotations

import sys

import pandas as pd

from opinion.db import Banco
from opinion.mundo import PASTA_BOOTSTRAP


def main():
    b = Banco()
    P = PASTA_BOOTSTRAP / "pulseira"
    ok = True
    print("Arquivo do bootstrap  x  banco")
    for nome, arq, sql in [
        ("encontros", "encontros.csv", "SELECT COUNT(*) FROM encontros WHERE encontro_id NOT LIKE 'e-live-%' AND encontro_id NOT LIKE 'e-checkpoint-%'"),
        ("interesses", "interesses.csv", "SELECT COUNT(*) FROM interesses WHERE encontro_id NOT LIKE 'e-live-%'"),
        ("status", "status.csv", "SELECT COUNT(*) FROM status_pulseira WHERE ts < '2026-10-01'"),
    ]:
        n_arq, n_db = len(pd.read_csv(P / arq)), b.um(sql)[0]
        bate = n_arq == n_db
        ok &= bate
        print(f"  {nome:<11}{n_arq:>8,} {n_db:>8,}   {'OK' if bate else 'DIFERENTE'}")

    verdade = pd.read_csv(PASTA_BOOTSTRAP / "verdade_simulacao_NAO_USAR_COMO_FEATURE" / "encontros_ocultos.csv",
                          usecols=["encontro_id", "usuario_publicou", "usuario_outro"])
    banco = pd.DataFrame(b.todos("SELECT encontro_id, usuario_a, usuario_b FROM encontros"),
                         columns=["encontro_id", "usuario_a", "usuario_b"])
    m = verdade.merge(banco, on="encontro_id")
    certos = ((m.usuario_publicou == m.usuario_a) & (m.usuario_outro == m.usuario_b)).sum()
    ok &= certos == len(m) and len(m) > 0
    print(f"\nIDs rotativos traduzidos corretamente: {certos:,} de {len(m):,} encontros")
    print("\nRESULTADO: " + ("tudo certo ✔" if ok else "há diferenças ✘"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
