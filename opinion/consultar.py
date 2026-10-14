"""
Consulta simples do que está no banco (é a "consulta simples" do Checkpoint 1).

    python -m opinion.consultar             # resumo + últimos 10 encontros
    python -m opinion.consultar --n 30
"""
from __future__ import annotations

import argparse

from opinion.db import Banco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    b = Banco()
    b.criar_esquema()
    print(f"Banco: {b.url}\n")
    print("Histórico do app                 Pulseira (via MQTT)")
    esq = ["usuarios", "obras", "avaliacoes", "seguidores", "interacoes"]
    dir_ = ["mqtt_mensagens", "encontros", "interesses", "status_pulseira", ""]
    for t1, t2 in zip(esq, dir_):
        n1 = b.um(f"SELECT COUNT(*) FROM {t1}")[0]
        lado2 = f"{t2:<17}{b.um(f'SELECT COUNT(*) FROM {t2}')[0]:>9,}" if t2 else ""
        print(f"  {t1:<15}{n1:>9,}        {lado2}")

    ok, rej = b.um("SELECT SUM(CASE WHEN valida = 1 THEN 1 ELSE 0 END), "
                   "SUM(CASE WHEN valida = 0 THEN 1 ELSE 0 END) FROM mqtt_mensagens")
    print(f"\nMensagens MQTT: {ok or 0:,} aceitas, {rej or 0:,} rejeitadas")
    print("Ocorrências (duplicadas são aceitas e ignoradas; as demais foram rejeitadas):")
    for erro, n in b.todos("SELECT erro, COUNT(*) FROM mqtt_mensagens WHERE erro IS NOT NULL GROUP BY erro ORDER BY 2 DESC"):
        print(f"  - {erro}: {n:,}")

    print("\nEncontros por local:")
    for local, n, dur in b.todos("SELECT local, COUNT(*), AVG(duracao_s) FROM encontros GROUP BY local ORDER BY 2 DESC"):
        print(f"  {local:<16}{n:>8,}   duração média {dur / 60:5.1f} min")

    print("\nModo de descoberta atual das pulseiras:")
    for modo, n in b.todos(
            "SELECT s.modo, COUNT(*) FROM status_pulseira s JOIN (SELECT pulseira_id, MAX(ts) AS ts FROM status_pulseira "
            "GROUP BY pulseira_id) u ON u.pulseira_id = s.pulseira_id AND u.ts = s.ts GROUP BY s.modo ORDER BY 2 DESC"):
        print(f"  {modo:<10}{n:>6,}")

    print(f"\nÚltimos {a.n} encontros recebidos:")
    print(f"  {'encontro':<16} {'pulseira':<9} {'usuário A':<10} {'usuário B':<10} {'início':<20} {'dur(s)':>6} {'RSSI':>6}  local")
    for r in b.todos("SELECT encontro_id, pulseira_id, usuario_a, usuario_b, inicio, duracao_s, rssi_medio, local "
                     "FROM encontros ORDER BY recebido_em DESC, inicio DESC LIMIT ?", (a.n,)):
        print(f"  {r[0]:<16} {r[1]:<9} {r[2]:<10} {r[3]:<10} {str(r[4])[:19]:<20} {r[5]:>6} {r[6]:>6.1f}  {r[7]}")


if __name__ == "__main__":
    main()
