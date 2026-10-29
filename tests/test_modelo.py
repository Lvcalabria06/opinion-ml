"""Baseline CC3: padronização só no treino, métricas no limiar, aprovação na validação."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from opinion.modelo.treinar import config, escolher_limiar, no_limiar, treinar


def _dataset(pasta, n=600, semente=0, y_da_regra=False):
    rng = np.random.default_rng(semente)
    compat = rng.random(n)
    extra = rng.random(n) * 50 + 100              # o modelo vê; a regra simples não (escala longe de 0/1)
    segue = (rng.random(n) < 0.3).astype(int)
    z = -2.2 + 1.2 * compat + 2.4 * (extra - 100) / 50 + 1.1 * segue
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    if y_da_regra:                                # a regra simples já acerta tudo: nada pode superá-la
        y = (compat > 0.7).astype(int)
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


def _so_logreg():
    cfg = copy.deepcopy(config())
    cfg["candidatos"] = {"regressao_logistica": cfg["candidatos"]["regressao_logistica"]}
    return cfg


def test_baseline_supera_regra_na_validacao_e_relata_metricas(tmp_path):
    _dataset(tmp_path)
    r = treinar(tmp_path, config(), registrar=False)
    assert r["supera_regra_validacao"]
    assert r["tipo"] in ("logreg", "arvore")
    te = r["vencedor"]["teste"]
    for k in ("pr_auc", "pr_auc_desconhecidos", "roc_auc", "brier", "prevalencia"):
        assert te[k] is not None
    nl = te["no_limiar_desconhecidos"]
    assert nl["vp"] + nl["fp"] + nl["fn"] + nl["vn"] == te["n_desconhecidos"]
    assert {"precisao", "recall", "f05", "taxa_sinal"} <= set(nl)
    # a regra simples é avaliada no mesmo protocolo (limiar próprio, escolhido na validação)
    assert "no_limiar_desconhecidos" in r["regra_simples"]["teste"]
    assert "brier" not in r["regra_simples"]["teste"]            # compat_score não é probabilidade


def test_padronizacao_ajustada_so_no_treino(tmp_path):
    df = _dataset(tmp_path)
    r = treinar(tmp_path, _so_logreg(), registrar=False)
    escala = r["_modelo"].named_steps["escala"]
    cols = ["compat_score", "extra", "algum_segue"]
    assert np.allclose(escala.mean_, df[df.split == "treino"][cols].mean().to_numpy())
    assert not np.allclose(escala.mean_, df[cols].mean().to_numpy())


def test_reprova_se_nao_supera_regra_na_validacao(tmp_path):
    _dataset(tmp_path, y_da_regra=True)
    with pytest.raises(RuntimeError, match="validação"):
        treinar(tmp_path, config(), registrar=False)


def test_limiar_f05_prefere_precisao():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    p = np.array([.95, .9, .8, .7, .6, .5, .4, .3, .2, .1])
    t_f05, _ = escolher_limiar(y, p, 0.5)
    t_f2, _ = escolher_limiar(y, p, 2.0)
    assert t_f05 > t_f2                              # F0,5 sinaliza menos, com mais precisão
    m = no_limiar(y, p, t_f05, 0.5)
    assert m["precisao"] == 1.0 and (m["vp"], m["fp"]) == (2, 0)


def test_split_vazio_falha(tmp_path):
    df = _dataset(tmp_path)
    df.loc[df.split == "teste", "split"] = "treino"
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    with pytest.raises(ValueError, match="teste"):
        treinar(tmp_path, registrar=False)
