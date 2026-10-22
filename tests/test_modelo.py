"""Baseline CC3: split temporal, padronização só no treino, supera a regra simples."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from opinion.modelo.treinar import config, treinar


def _dataset(pasta, n=600, semente=0):
    rng = np.random.default_rng(semente)
    compat = rng.random(n)
    extra = rng.random(n)                       # o modelo vê; a regra simples não
    segue = (rng.random(n) < 0.3).astype(int)
    z = -2.2 + 1.2 * compat + 2.4 * extra + 1.1 * segue
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    split = np.array(["treino"] * int(n * 0.6) + ["validacao"] * int(n * 0.2)
                     + ["teste"] * (n - int(n * 0.6) - int(n * 0.2)))
    cols = ["compat_score", "extra", "algum_segue"]
    df = pd.DataFrame({"encontro_id": [f"e-{i}" for i in range(n)], "par_id": [f"p-{i}" for i in range(n)],
                       "compat_score": compat, "extra": extra, "algum_segue": segue,
                       "split": split, "conectou": y})
    df.to_parquet(pasta / "dataset_modelo.parquet", index=False)
    meta = {"colunas_features": cols, "por_split": {s: {"encontros": int((df.split == s).sum()),
                                                        "taxa_conexao": float(df[df.split == s].conectou.mean()),
                                                        "pares": int((df.split == s).sum())}
                                                    for s in ("treino", "validacao", "teste")},
            "taxa_conexao": float(df.conectou.mean())}
    (pasta / "metadados.json").write_text(json.dumps(meta))
    return df


def test_baseline_supera_regra_e_usa_pipeline(tmp_path):
    _dataset(tmp_path)
    cfg = config()
    cfg["mlflow"]["experimento"] = "teste-cc3"
    r = treinar(tmp_path, cfg, registrar=False)
    assert r["supera_regra_teste"]
    assert r["vencedor"]["teste"]["pr_auc_desconhecidos"] > r["regra_simples"]["teste"]["pr_auc_desconhecidos"]
    assert r["tipo"] in ("logreg", "arvore")
    # o StandardScaler (quando logreg) foi ajustado só no treino: é um passo do Pipeline
    from opinion.modelo.treinar import carregar, _pipeline
    dm, meta = carregar(tmp_path)
    m = _pipeline(r["tipo"], r["params"])
    m.fit(dm[dm.split == "treino"][meta["colunas_features"]], dm[dm.split == "treino"].conectou)
    assert isinstance(m, Pipeline)
    if r["tipo"] == "logreg":
        assert "escala" in m.named_steps


def test_split_vazio_falha(tmp_path):
    df = _dataset(tmp_path)
    df.loc[df.split == "teste", "split"] = "treino"
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    with pytest.raises(ValueError, match="teste"):
        treinar(tmp_path, registrar=False)
