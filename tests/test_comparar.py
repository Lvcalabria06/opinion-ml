"""CC4: 3+ modelos no mesmo protocolo; a escolha não depende do teste."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from opinion.modelo.comparar import comparar, ic_bootstrap
from opinion.modelo.treinar import config


def _dataset(pasta, n=900, semente=0):
    rng = np.random.default_rng(semente)
    compat = rng.random(n)
    extra = rng.random(n)
    segue = (rng.random(n) < 0.3).astype(int)
    z = -2.0 + 1.0 * compat + 2.5 * (extra > 0.6) + 1.0 * segue
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    split = np.array(["treino"] * int(n * 0.6) + ["validacao"] * int(n * 0.2)
                     + ["teste"] * (n - int(n * 0.6) - int(n * 0.2)))
    df = pd.DataFrame({"encontro_id": [f"x-{i}" for i in range(n)], "par_id": [f"p-{i}" for i in range(n)],
                       "local_original": rng.choice(["cinema", "campus"], n),
                       "compat_score": compat, "extra": extra, "algum_segue": segue, "split": split, "conectou": y})
    df.to_parquet(pasta / "dataset_modelo.parquet", index=False)
    (pasta / "metadados.json").write_text(json.dumps({"colunas_features": ["compat_score", "extra", "algum_segue"]}))
    return df


def _cfg():
    cfg = copy.deepcopy(config())
    cfg["comparacao"]["bootstrap"]["n"] = 50
    cfg["comparacao"]["importancia"]["repeticoes"] = 2
    cfg["comparacao"]["modelos"]["random_forest"]["grade"]["n_estimators"] = [30]
    return cfg


def _rodar(pasta):
    return comparar(pasta, _cfg(), registrar=False, doc=None, figuras=None)


def test_compara_tres_ou_mais_modelos_com_todas_as_metricas(tmp_path):
    _dataset(tmp_path)
    r = _rodar(tmp_path)
    assert len(r["familias"]) >= 3
    for f in r["familias"].values():
        te = f["teste"]
        for k in ("pr_auc", "pr_auc_desconhecidos", "roc_auc", "roc_auc_desconhecidos", "brier"):
            assert te[k] is not None
        nl = te["no_limiar_desconhecidos"]
        assert nl["vp"] + nl["fp"] + nl["fn"] + nl["vn"] == te["n_desconhecidos"]
    ic = r["bootstrap_teste_desconhecidos"]
    assert "regra" in ic and all(v["ic"][0] <= v["pr_auc"] <= v["ic"][1] + 1e-9 for v in ic.values())
    assert set(r["_contexto"].local) == {"cinema", "campus"}


def test_vencedor_e_escolhido_na_validacao_e_nao_muda_com_o_teste(tmp_path):
    df = _dataset(tmp_path)
    r1 = _rodar(tmp_path)
    melhor_va = max(r1["familias"], key=lambda k: r1["familias"][k]["validacao"]["pr_auc_desconhecidos"])
    assert r1["vencedor"] == melhor_va
    # embaralha só o label do teste: escolha, limiares e métricas de validação não podem mudar
    te = df.split == "teste"
    df.loc[te, "conectou"] = np.random.default_rng(9).permutation(df.loc[te, "conectou"].to_numpy())
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    r2 = _rodar(tmp_path)
    assert r2["vencedor"] == r1["vencedor"]
    for k in r1["familias"]:                  # tolerância só para o último bit do ponto flutuante
        assert r2["familias"][k]["params"] == r1["familias"][k]["params"]
        assert r2["familias"][k]["limiar"] == pytest.approx(r1["familias"][k]["limiar"], rel=1e-9)
        v1, v2 = r1["familias"][k]["validacao"], r2["familias"][k]["validacao"]
        for m in ("pr_auc", "pr_auc_desconhecidos", "roc_auc", "brier"):
            assert v2[m] == pytest.approx(v1[m], rel=1e-9)
        assert v2["no_limiar_desconhecidos"]["vp"] == v1["no_limiar_desconhecidos"]["vp"]


def test_bootstrap_pareado_detecta_diferenca_e_igualdade():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    bom = y + rng.normal(0, 0.5, len(y))
    ruim = rng.random(len(y))
    r = ic_bootstrap(y, {"bom": bom, "ruim": ruim, "copia": bom.copy()}, ["ruim", "bom"], 200, 0)
    assert r["bom"]["delta_ruim_ic"][0] > 0                  # claramente melhor: IC acima de zero
    assert r["copia"]["delta_bom_ic"] == [0.0, 0.0]          # idêntico: diferença exatamente zero
