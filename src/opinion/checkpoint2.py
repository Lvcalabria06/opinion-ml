"""Demonstra o aviso de afinidade no início do encontro e a previsão após o encontro.
Use --baseline para incluir a demonstração acadêmica do CC3."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from opinion import contrato as C
from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.pontuar import carregar_modelo, matriz, pontuar, traduzir
from opinion.modelo.treinar import _imprimir, carregar, config, treinar
from opinion.pipeline import PASTA_STAGING, ingestao
from opinion.pipeline.compatibilidade import motivos_do_sinal
from opinion.sinal_categoria import feedback
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
    ap = argparse.ArgumentParser(description="Checkpoint 2: aviso por categoria e estimativa de interação")
    ap.add_argument("--usuarios", nargs=2, metavar=("A", "B"))
    ap.add_argument("--local", default="cinema", choices=C.LOCAIS)
    ap.add_argument("--duracao", type=int, default=600, help="segundos")
    ap.add_argument("--rssi", type=float, default=-58.0)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--baseline", action="store_true", help="inclui a previsão de referência do CC3")
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

    print("1) Pontuação ao vivo usa o mesmo pipeline do treino?")
    ref = dm[dm.split == "teste"].sample(1, random_state=a.semente).iloc[0]
    enc_ref = f["encontros"][f["encontros"].encontro_id == ref.encontro_id].iloc[0].to_dict()
    X_ref, _ = matriz(enc_ref, f, cfg_p, cols)
    igual = np.allclose(X_ref.iloc[0].to_numpy(dtype=float), ref[cols].to_numpy(dtype=float))
    print(f"   {ref.encontro_id}: features recalculadas {'==' if igual else '!='} dataset_modelo")
    if not igual:
        sys.exit("   FALHOU: a pontuação ao vivo não reproduz as features do treino")

    print("2) Encontro simulado (formato do contrato MQTT)")
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

    X, dados = matriz(encontro, f, cfg_p, cols)
    linha = dados.iloc[0]
    midias = motivos_do_sinal(linha, cfg_p)
    print("3) Aviso na detecção da proximidade")
    if midias:
        print("  ", feedback(payload["encontro_id"], midias, inicio.isoformat())["mensagem"])
    else:
        print("   Sem categoria com afinidade alta.")

    estimativas = {}
    for midia in midias:
        try:
            modelo, info = carregar_modelo(cfg, midia)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST" and "not found" not in str(exc).lower():
                raise
            print(f"   {midia}: modelo indisponível. Treino: python -m opinion.modelo.comparar --midia {midia}")
            continue
        entrada = X.assign(compat_score=linha[f"compat_score_{midia}"])[list(modelo.feature_names_in_)]
        p = float(modelo.predict_proba(entrada)[0, 1])
        estimativas[midia] = p
        print(f"   {midia}: afinidade {linha[f'compat_score_{midia}']:.3f}; "
              f"estimativa de interação em 48 h após o encontro {p:.3f} ({info['algoritmo']})")

    saida = Path(PASTA_PROCESSADOS) / "avisos_categoria.csv"
    registro = {"encontro_id": payload["encontro_id"], "usuario_a": ua, "usuario_b": ub,
                "inicio": inicio.isoformat(), "midias": "|".join(midias),
                "estimativas_48h": json.dumps(estimativas)}
    pd.DataFrame([registro]).to_csv(saida, mode="a", header=not saida.exists(), index=False)

    if a.baseline or a.retreinar:
        if a.retreinar:
            _imprimir(treinar(registrar=True))
        modelo, info = carregar_modelo(cfg)
        r = pontuar(encontro, f, modelo, info, cfg_p, cols)
        r.pop("_X")
        print(f"   Baseline acadêmico CC3, estimativa de interação: {r['probabilidade']:.3f}")
        previsoes = Path(PASTA_PROCESSADOS) / "previsoes.csv"
        pd.DataFrame([{**r, "pontuado_em": pd.Timestamp.now().isoformat(timespec="seconds")}]).to_csv(
            previsoes, mode="a", header=not previsoes.exists(), index=False)
    print("\nCHECKPOINT 2 OK ✔")


if __name__ == "__main__":
    main()
