"""CC6: validação cruzada agrupada por par; nunca usa o teste; resumo com estabilidade e gap."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from opinion.modelo.treinar import config
from opinion.modelo.validar import folds, validar


def _dataset(pasta, n_pares=120, enc=6, semente=0):
    """Pares repetidos (vários encontros por par); o rótulo é do par. Split por par, como no pipeline."""
    rng = np.random.default_rng(semente)
    par = np.repeat([f"p-{i}" for i in range(n_pares)], enc)
    amigo = rng.random(n_pares) < 0.15
    amigo[:8] = True                                   # garante positivos
    split_par = np.array(["treino"] * 70 + ["validacao"] * 25 + ["teste"] * (n_pares - 95))
    n = n_pares * enc
    compat = np.clip(rng.normal(0.4, 0.2, n) + 0.3 * np.repeat(amigo, enc), 0, 1)
    df = pd.DataFrame({"encontro_id": [f"e-{i}" for i in range(n)], "par_id": par,
                       "compat_score": compat, "extra": rng.random(n),
                       "split": np.repeat(split_par, enc), "conectou": np.repeat(amigo.astype(int), enc)})
    df.to_parquet(pasta / "dataset_modelo.parquet", index=False)
    (pasta / "metadados.json").write_text(json.dumps({"colunas_features": ["compat_score", "extra"]}))
    (pasta / "otimizacao.json").write_text(json.dumps({"vencedor": "random_forest", "familias": {
        "regressao_logistica": {"tipo": "logreg", "params": {"C": 0.1, "class_weight": None}},
        "random_forest": {"tipo": "random_forest", "params": {
            "n_estimators": 15, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": None}}}}))
    return df


def _cfg():
    cfg = copy.deepcopy(config())
    cfg["validacao_cruzada"].update({
        "k": 3, "sementes": [0, 1], "sementes_curvas": [0], "fracoes_aprendizado": [0.5, 1.0],
        "complexidade": {"regressao_logistica": {"parametro": "C", "valores": [0.01, 1]},
                         "random_forest": {"parametro": "max_depth", "valores": [2, 4]}}})
    return cfg


def test_folds_nao_misturam_pares_e_tem_positivos(tmp_path):
    df = _dataset(tmp_path)
    cv = df[df.split != "teste"].reset_index(drop=True)
    fs = folds(cv, 3, 0)
    assert len(fs) == 3
    todos = np.concatenate([va for _, va in fs])
    assert sorted(todos) == list(range(len(cv)))            # cada linha em um único fold de validação
    pares_va = [set(cv.par_id.iloc[va]) for _, va in fs]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not pares_va[i] & pares_va[j]
    for tr, va in fs:
        assert cv.conectou.iloc[va].sum() > 0 and cv.conectou.iloc[tr].sum() > 0
        assert not set(cv.par_id.iloc[tr]) & set(cv.par_id.iloc[va])


def test_folds_falham_alto_sem_positivos_suficientes(tmp_path):
    df = _dataset(tmp_path)
    cv = df[df.split != "teste"].reset_index(drop=True)
    cv["conectou"] = 0
    cv.loc[cv.par_id == "p-0", "conectou"] = 1               # um único par positivo, k = 3
    with pytest.raises(ValueError, match="positivo"):
        folds(cv, 3, 0)


def test_cv_nao_usa_linhas_do_teste(tmp_path):
    df = _dataset(tmp_path)
    r = validar(tmp_path, _cfg(), registrar=False, doc=None, figuras=None)
    n_cv = int((df.split != "teste").sum())
    assert r["protocolo"]["n_cv"] == n_cv
    assert set(r["protocolo"]["splits_usados"]) == {"treino", "validacao"}
    for m in r["modelos"].values():
        assert all(f["n_val"] + f["n_treino"] == n_cv for f in m["folds"])
    # nenhum par do teste participa da CV
    assert not set(df[df.split == "teste"].par_id) & set(r["_pares_cv"])


def test_resumo_tem_estabilidade_e_gap(tmp_path):
    _dataset(tmp_path)
    r = validar(tmp_path, _cfg(), registrar=False, doc=None, figuras=None)
    assert set(r["modelos"]) == {"regressao_logistica", "random_forest", "regra"}   # a regra simples entra
    for nome, m in r["modelos"].items():
        s = m["resumo"]
        for k in ("pr_auc_media", "pr_auc_desvio", "pr_auc_cv", "gap_pr_auc", "roc_auc_media"):
            assert k in s and s[k] is not None
        assert s["pr_auc_cv"] == pytest.approx(s["pr_auc_desvio"] / s["pr_auc_media"])
        assert len(m["folds"]) == 3 * 2                      # k folds × sementes
        assert set(s["por_semente"]) == {"0", "1"}
    assert r["modelos"]["regra"]["resumo"]["gap_pr_auc"] == pytest.approx(0, abs=0.2)
    assert len(r["curva_aprendizado"]["random_forest"]) == 2
    assert len(r["curva_complexidade"]["regressao_logistica"]["pontos"]) == 2
    assert "referencia_cc5" in r


def test_analise_gap_nao_positivo_e_sem_sobreajuste(tmp_path):
    from opinion.modelo.validar import analise, _nivel_sobreajuste
    assert _nivel_sobreajuste(-0.002, -0.03) == "sem sobreajuste"
    assert _nivel_sobreajuste(0.018, 0.30) == "sobreajuste moderado"
    _dataset(tmp_path)
    r = validar(tmp_path, _cfg(), registrar=False, doc=None, figuras=None)
    for n in ("regressao_logistica", "random_forest"):
        r["modelos"][n]["resumo"]["gap_pr_auc"] = -0.002
    txt = "\n".join(analise(r))
    assert "sem sobreajuste" in txt and "sobreajuste pequeno" not in txt
