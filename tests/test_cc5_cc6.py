"""CC5 (otimização) e CC6 (validação cruzada): protocolo sem uso do teste e agrupamento por par."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from opinion.modelo.otimizar import _como_sugestao, otimizar, sugerir
from opinion.modelo.treinar import config
from opinion.modelo.validacao_cruzada import kfold, resumir, temporal


def _dataset(pasta, n=1500, semente=0):
    rng = np.random.default_rng(semente)
    compat, extra = rng.random(n), rng.random(n)
    segue = (rng.random(n) < 0.3).astype(int)
    z = -2.2 + 1.0 * compat + 2.5 * (extra > 0.6) + 1.0 * segue
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    semana = np.sort(rng.integers(1, 13, n))
    split = np.where(semana <= 8, "treino", np.where(semana <= 10, "validacao", "teste"))
    inicio = pd.Timestamp("2026-07-06") + pd.to_timedelta((semana - 1) * 7 * 24 + rng.integers(0, 150, n), unit="h")
    par = [f"p-{i}" for i in rng.integers(0, n // 3, n)]          # pares repetidos entre encontros
    df = pd.DataFrame({"encontro_id": [f"x-{i}" for i in range(n)], "par_id": par, "local_original": "cinema",
                       "compat_score": compat, "extra": extra, "algum_segue": segue, "split": split,
                       "semana": semana, "inicio": inicio, "fim": inicio + pd.Timedelta(minutes=10), "conectou": y})
    df.to_parquet(pasta / "dataset_modelo.parquet", index=False)
    (pasta / "metadados.json").write_text(json.dumps({"colunas_features": ["compat_score", "extra", "algum_segue"]}))
    return df


def _cfg():
    cfg = copy.deepcopy(config())
    cfg["comparacao"]["bootstrap"]["n"] = 50
    cfg["otimizacao"]["amostradores"] = ["tpe", "aleatorio"]
    return cfg


def test_sugerir_respeita_o_espaco_e_listas_viram_categorias():
    import optuna
    espaco = {"a": {"min": 1, "max": 5, "inteiro": True}, "b": {"min": 0.01, "max": 1.0, "log": True},
              "c": [[64, 32], [16]], "d": [None, "balanced"]}
    estudo = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    estudo.enqueue_trial(_como_sugestao({"a": 3, "c": (64, 32), "d": "balanced"}, espaco))
    vistos = []
    estudo.optimize(lambda t: vistos.append(sugerir(t, espaco)) or 0.0, n_trials=6)
    assert vistos[0]["a"] == 3 and vistos[0]["c"] == [64, 32] and vistos[0]["d"] == "balanced"
    for p in vistos:
        assert 1 <= p["a"] <= 5 and 0.01 <= p["b"] <= 1.0 and p["c"] in ([64, 32], [16])
        assert p.get("d") in (None, "balanced")


def test_otimizacao_nao_usa_o_teste_e_parte_da_grade(tmp_path):
    df = _dataset(tmp_path)
    r1 = otimizar(tmp_path, _cfg(), familia="arvore_decisao", trials=4, registrar=False, doc=None, figuras=None)
    # a busca começa pela melhor configuração da grade: nunca termina pior que ela na validação
    assert max(b["melhor"] for b in r1["buscas"].values()) >= r1["grade_cc4"]["validacao"]["pr_auc_desconhecidos"] - 1e-12
    assert 0 < r1["teste"]["brier"] < 0.5 and r1["calibracao"] == "sigmoid"
    te = df.split == "teste"
    df.loc[te, "conectou"] = np.random.default_rng(3).permutation(df.loc[te, "conectou"].to_numpy())
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    r2 = otimizar(tmp_path, _cfg(), familia="arvore_decisao", trials=4, registrar=False, doc=None, figuras=None)
    assert r2["params"] == r1["params"]
    assert r2["limiar"] == pytest.approx(r1["limiar"], rel=1e-9)


def test_kfold_agrupa_por_par_e_resume_estabilidade(tmp_path):
    df = _dataset(tmp_path)
    base = df[df.split.isin(["treino", "validacao"])].reset_index(drop=True)
    modelos = {"otimizado": {"tipo": "arvore", "params": {"max_depth": 3}},
               "referencia": {"tipo": "logreg", "params": {"C": 1.0}}, "regra": None}
    kf = kfold(base, ["compat_score", "extra", "algum_segue"], modelos, k=4, semente=0)   # assert interno: sem par nos dois lados
    assert set(kf.modelo) == set(modelos) and kf.fold.nunique() == 4
    assert kf.groupby("fold").n_avaliacao.first().sum() == len(base)
    res = resumir(kf)
    assert {"media", "desvio", "cv", "gap"} <= set(res.columns)
    assert (res.loc[["otimizado", "referencia"], "treino_media"] > 0).all()


def test_janela_temporal_so_treina_com_o_passado_e_purga(tmp_path):
    df = _dataset(tmp_path)
    base = df[df.split.isin(["treino", "validacao"])].reset_index(drop=True)
    tp = temporal(base, ["compat_score", "extra", "algum_segue"], {"regra": None}, primeira=5, horas_label=48)
    assert list(tp.semana_avaliada) == list(range(5, 11))
    for _, r in tp.iterrows():
        k = r.semana_avaliada
        inicio_k = base[base.semana == k].inicio.min().normalize()
        esperado = ((base.semana < k) & (base.fim + pd.Timedelta(hours=48) < inicio_k)).sum()
        assert r.n_treino == esperado
