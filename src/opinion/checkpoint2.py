"""
CHECKPOINT 2 (Sprint 3 / CC3): "um encontro novo passa pelo pipeline e recebe a
probabilidade do baseline".

O teste:
  1. carrega o baseline registrado no MLflow (opinion-baseline-cc3@champion) — sem retreinar;
     se ainda não há modelo registrado, treina uma vez
  2. confere que a pontuação ao vivo usa o MESMO pipeline do treino: recalcula um encontro
     do teste e compara com a linha do dataset_modelo
  3. monta um encontro NOVO no formato do contrato MQTT (IDs rotativos do dia), depois do
     último dado da base, entre dois usuários que não se seguem
  4. valida e traduz os IDs (como a ingestão), calcula compatibilidade e features só com o
     passado, e pontua com o modelo e com a regra simples
  5. grava a previsão em data/processed/previsoes.csv (no lugar do dashboard, que ainda não existe)

Pré-requisito: o pipeline do CC2 já rodou (data/staging e data/processed). Se não, roda
a partir dos arquivos do bootstrap.

    python -m opinion.checkpoint2
    python -m opinion.checkpoint2 --usuarios usr-00010 usr-00420 --local cinema --duracao 900
    python -m opinion.checkpoint2 --retreinar        # treina e registra uma nova versão antes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from opinion import contrato as C
from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.pontuar import carregar_modelo, matriz, pontuar, traduzir
from opinion.modelo.treinar import _imprimir, carregar, config, treinar
from opinion.pipeline import PASTA_STAGING, ingestao
from opinion.pipeline.executar import config as config_pipeline
from opinion.pipeline.executar import rodar_tudo


def _seguem(seg: pd.DataFrame, a: str, b: str, t) -> bool:
    s = seg[seg.desde < t]
    return bool(((s.seguidor == a) & (s.seguido == b) | (s.seguidor == b) & (s.seguido == a)).any())


def _escolher_par(f, t, semente: int) -> tuple[str, str]:
    """Dois usuários com avaliações e que não se seguem antes de t (o público da pulseira)."""
    rng = np.random.default_rng(semente)
    com_aval = np.sort(f["avaliacoes"][f["avaliacoes"].criado_em < t].usuario_id.unique())
    for _ in range(200):
        a, b = rng.choice(com_aval, 2, replace=False)
        if not _seguem(f["seguidores"], a, b, t):
            return str(a), str(b)
    raise RuntimeError("não achei um par de desconhecidos")


def main():
    ap = argparse.ArgumentParser(description="Checkpoint 2: encontro novo -> pipeline -> probabilidade do baseline")
    ap.add_argument("--usuarios", nargs=2, metavar=("A", "B"))
    ap.add_argument("--local", default="cinema", choices=C.LOCAIS)
    ap.add_argument("--duracao", type=int, default=600, help="segundos")
    ap.add_argument("--rssi", type=float, default=-58.0)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--retreinar", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg, cfg_p = config(), config_pipeline()

    if not (Path(PASTA_STAGING) / "encontros.parquet").exists() or \
            not (Path(PASTA_PROCESSADOS) / "dataset_modelo.parquet").exists():
        print("0) Sem dados do pipeline: rodando o pipeline do CC2 a partir do bootstrap")
        rodar_tudo("arquivos")
    f = ingestao.ler_staging(PASTA_STAGING)
    dm, meta = carregar()
    cols = meta["colunas_features"]

    print("1) Baseline registrado no MLflow")
    try:
        if a.retreinar:
            raise LookupError
        modelo, info = carregar_modelo(cfg)
    except Exception:
        print("   treinando e registrando (uma vez) ...")
        _imprimir(treinar(registrar=True))
        modelo, info = carregar_modelo(cfg)
    m = info["metricas"]
    print(f"   {info['uri']}  v{info['versao']}  ({info['algoritmo']})  run {info['run_id']}")
    print(f"   validação, PR-AUC desconhecidos: modelo {m['validacao_pr_auc_desconhecidos']:.3f} x "
          f"regra {m['validacao_regra_pr_auc_desconhecidos']:.3f}   (critério de aprovação)")
    print(f"   teste,     PR-AUC desconhecidos: modelo {m['teste_pr_auc_desconhecidos']:.3f} x "
          f"regra {m['teste_regra_pr_auc_desconhecidos']:.3f}   (registrado no treino, não recalculado)")
    print(f"   limiar do sinal {info['limiar']:.3f} (modelo) · {info['limiar_regra']:.3f} (regra, compat_score)")

    print("2) Pontuação ao vivo usa o mesmo pipeline do treino?")
    ref = dm[dm.split == "teste"].sample(1, random_state=a.semente).iloc[0]
    enc_ref = f["encontros"][f["encontros"].encontro_id == ref.encontro_id].iloc[0].to_dict()
    X_ref, _ = matriz(enc_ref, f, cfg_p, cols)
    igual = np.allclose(X_ref.iloc[0].to_numpy(dtype=float), ref[cols].to_numpy(dtype=float))
    print(f"   {ref.encontro_id}: features recalculadas {'==' if igual else '!='} dataset_modelo")
    if not igual:
        sys.exit("   FALHOU: a pontuação ao vivo não reproduz as features do treino")

    print("3) Encontro NOVO publicado pela pulseira (formato do contrato MQTT)")
    corte = max(f["interacoes"].criado_em.max(), f["encontros"].fim.max())
    inicio = (corte + pd.Timedelta(hours=1)).floor("s")
    ua, ub = a.usuarios or _escolher_par(f, inicio, a.semente)
    pulseira = f["usuarios"].set_index("usuario_id").pulseira_id.get(ua)
    payload = {"v": C.VERSAO, "encontro_id": f"e-checkpoint2-{inicio:%Y%m%d%H%M%S}",
               "usuario_a": C.token_do_dia(ua, inicio.date()), "usuario_b": C.token_do_dia(ub, inicio.date()),
               "inicio": inicio.isoformat(), "duracao_s": a.duracao, "rssi_medio": a.rssi, "local": a.local}
    print(f"   tópico : {C.topico(pulseira, 'encontro')}")
    print(f"   payload: {payload}")
    encontro = traduzir(payload, f["usuarios"].usuario_id.tolist())
    encontro["pulseira_id"] = pulseira
    print(f"   IDs rotativos traduzidos: {payload['usuario_a']} -> {encontro['usuario_a']}, "
          f"{payload['usuario_b']} -> {encontro['usuario_b']}")

    print("4) Pipeline (compatibilidade + features, só o passado) e baseline")
    r = pontuar(encontro, f, modelo, info, cfg_p, cols)
    X = r.pop("_X").iloc[0]
    for c in ("compat_score", "sim_generos", "generos_top3_comum", "n_obras_comuns", "historico_min",
              "algum_segue", "interacoes_anteriores_par", "n_encontros_anteriores_par", "log_duracao"):
        print(f"   {c:<28} {X[c]:.4g}")
    print(f"   PROBABILIDADE de conexão em 48 h: {r['probabilidade']:.3f}  "
          f"-> sinal: {'SIM' if r['sinal_modelo'] else 'não'} (limiar {r['limiar']:.3f})")
    print(f"   regra simples: compat_score {r['compat_score']:.3f} -> sinal: {'SIM' if r['sinal_regra'] else 'não'}")
    if not 0.0 <= r["probabilidade"] <= 1.0:
        sys.exit("   FALHOU: probabilidade fora de [0, 1]")

    saida = Path(PASTA_PROCESSADOS) / "previsoes.csv"
    linha = pd.DataFrame([{**r, "pontuado_em": pd.Timestamp.now().isoformat(timespec="seconds")}])
    linha.to_csv(saida, mode="a", header=not saida.exists(), index=False)
    print(f"5) Previsão gravada em {saida} (o dashboard ainda não existe; é ele que vai ler isto)")
    print("\nCHECKPOINT 2 OK ✔")


if __name__ == "__main__":
    main()
