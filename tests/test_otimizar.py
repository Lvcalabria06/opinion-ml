"""CC5: otimização com Optuna; escolha só na validação, parâmetros dentro do espaço do config."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from opinion.modelo.otimizar import otimizar
from opinion.modelo.treinar import config
from tests.test_comparar import _dataset


def _cfg():
    cfg = copy.deepcopy(config())
    cfg["comparacao"]["modelos"]["random_forest"]["grade"]["n_estimators"] = [20]
    for f in cfg["otimizacao"]["familias"].values():
        f["tentativas"] = 4
    esp = cfg["otimizacao"]["familias"]["random_forest"]["espaco"]
    esp["n_estimators"] = {"tipo": "int", "min": 10, "max": 30}
    return cfg


def _rodar(pasta):
    return otimizar(pasta, _cfg(), registrar=False, doc=None, figuras=None)


def _no_espaco(nome, v, e):
    t = e["tipo"]
    if t == "categorico":
        return v in e["valores"]
    return e["min"] <= v <= e["max"]


def test_parametros_dentro_do_espaco(tmp_path):
    _dataset(tmp_path)
    cfg = _cfg()
    r = otimizar(tmp_path, cfg, registrar=False, doc=None, figuras=None)
    for fam, f in r["familias"].items():
        espaco = cfg["otimizacao"]["familias"][fam]["espaco"]
        assert f["n_tentativas"] == 4
        assert set(f["params"]) == set(espaco)
        for k, v in f["params"].items():
            assert _no_espaco(k, v, espaco[k]), (fam, k, v)
        for t in f["tentativas"]:
            for k, v in t["params"].items():
                assert _no_espaco(k, v, espaco[k]), (fam, k, v)


def test_vencedor_escolhido_na_validacao_e_teste_nao_influencia(tmp_path):
    df = _dataset(tmp_path)
    r1 = _rodar(tmp_path)
    te = df.split == "teste"
    df.loc[te, "conectou"] = np.random.default_rng(9).permutation(df.loc[te, "conectou"].to_numpy())
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    r2 = _rodar(tmp_path)
    assert r2["vencedor"] == r1["vencedor"]
    melhor = max(r1["familias"], key=lambda k: r1["familias"][k]["validacao"]["pr_auc"])
    assert r1["vencedor"] == melhor
    for k in r1["familias"]:
        a, b = r1["familias"][k], r2["familias"][k]
        assert set(a["params"]) == set(b["params"])
        for p, v in a["params"].items():
            assert b["params"][p] == (pytest.approx(v, rel=1e-9) if isinstance(v, float) else v)
        assert b["limiar"] == pytest.approx(a["limiar"], rel=1e-9)
        assert b["validacao"]["pr_auc"] == pytest.approx(a["validacao"]["pr_auc"], rel=1e-9)


def test_grava_otimizacao_json(tmp_path):
    _dataset(tmp_path)
    r = _rodar(tmp_path)
    j = json.loads((tmp_path / "otimizacao.json").read_text(encoding="utf-8"))
    assert j["vencedor"] == r["vencedor"]
    assert set(j["familias"]) == {"regressao_logistica", "random_forest"}
    for k, f in j["familias"].items():
        assert f["params"] == r["familias"][k]["params"] and f["tipo"]


def test_detecta_parametro_no_limite_do_espaco():
    from opinion.modelo.otimizar import no_limite
    esp = {"C": {"tipo": "float_log", "min": 0.001, "max": 100},
           "max_depth": {"tipo": "int", "min": 2, "max": 24},
           "leaf": {"tipo": "int_log", "min": 5, "max": 200},
           "cw": {"tipo": "categorico", "valores": [None, "balanced"]}}
    assert no_limite({"C": 0.001, "max_depth": 12, "leaf": 30, "cw": None}, esp) == ["C"]
    assert no_limite({"C": 0.5, "max_depth": 2, "leaf": 200, "cw": "balanced"}, esp) == ["max_depth", "leaf"]
    assert no_limite({"C": 0.5, "max_depth": 12, "leaf": 30, "cw": None}, esp) == []


def test_nota_de_borda_interpreta_regularizacao_sem_pedir_ampliacao():
    from opinion.modelo.otimizar import analise
    esp_lr = {"C": {"tipo": "float_log", "min": 0.001, "max": 100, "regulariza": "menor"},
              "class_weight": {"tipo": "categorico", "valores": [None, "balanced"]}}
    esp_rf = {"n_estimators": {"tipo": "int", "min": 100, "max": 400},
              "max_depth": {"tipo": "int", "min": 2, "max": 24, "regulariza": "menor"}}

    def met(pr, teste=True):
        m = {"pr_auc": pr, "prevalencia": 0.0075, "roc_auc": 0.8, "brier": 0.007,
             "no_limiar_desconhecidos": {"precisao": 0.1, "recall": 0.1, "taxa_sinal": 0.01}}
        return m

    def fam(params, esp, vl, te, gv, gt):
        return {"params": params, "espaco": esp, "validacao": met(vl), "teste": met(te),
                "melhor_grade_cc4": {"validacao": met(gv), "teste": met(gt)}}
    r = {"vencedor": "lr", "regra_simples": {"validacao": met(0.027), "teste": met(0.027)},
         "protocolo": {"metrica_escolha": "pr_auc"},
         "_familias": {"lr": fam({"C": 0.001, "class_weight": None}, esp_lr, 0.070, 0.061, 0.061, 0.063),
                       "random_forest": fam({"n_estimators": 100, "max_depth": 12}, esp_rf, 0.065, 0.055, 0.061, 0.055)}}
    texto = "\n".join(analise(r))
    assert "borda do espaço" in texto and "regularização máxima" in texto
    assert "não se transferiu" in texto and "sem ampliá-lo" in texto
    assert "pode estar fora das faixas" not in texto
    # n_estimators na borda mas sem `regulariza`: nota neutra, sem leitura de regularização para ele
    assert "nota neutra" in texto and "`n_estimators` na borda: nota neutra" in texto
