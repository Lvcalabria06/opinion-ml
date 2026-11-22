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


def test_publico_todos_usa_todos_os_encontros(tmp_path):
    """Modo Last.fm: não existe 'desconhecidos'; sem algum_segue, o recorte é o conjunto inteiro."""
    from opinion.modelo.treinar import mascara_desconhecidos
    df = _dataset(tmp_path)
    cols = ["compat_score", "extra"]
    df = df.drop(columns="algum_segue")
    df.to_parquet(tmp_path / "dataset_modelo.parquet", index=False)
    meta = json.loads((tmp_path / "metadados.json").read_text())
    meta["colunas_features"] = cols
    (tmp_path / "metadados.json").write_text(json.dumps(meta))
    cfg = copy.deepcopy(config())
    cfg["publico"] = "todos"
    assert mascara_desconhecidos(df, cfg).all()
    assert mascara_desconhecidos(df).all()                 # sem algum_segue: tudo True
    r = treinar(tmp_path, cfg, registrar=False)
    te = r["vencedor"]["teste"]
    assert te["n_desconhecidos"] == te["n"] == int((df.split == "teste").sum())
    assert te["pr_auc_desconhecidos"] == pytest.approx(te["pr_auc"])


def _dataset_todos(pasta):
    df = _dataset(pasta, n=1500)
    df = df.drop(columns="algum_segue")
    df["local_original"] = "campus"
    df.to_parquet(pasta / "dataset_modelo.parquet", index=False)
    meta = json.loads((pasta / "metadados.json").read_text())
    meta["colunas_features"] = ["compat_score", "extra"]
    (pasta / "metadados.json").write_text(json.dumps(meta))
    return df


def test_doc_do_cc4_em_modo_todos_nao_fala_de_desconhecidos(tmp_path):
    """No experimento Last.fm o texto não pode citar o recorte 'desconhecidos', 'interação em 48 h' nem o simulador."""
    from opinion.modelo.comparar import comparar, escrever_doc
    _dataset_todos(tmp_path)
    cfg = copy.deepcopy(config())
    cfg["publico"] = "todos"
    cfg["comparacao"]["bootstrap"]["n"] = 20
    cfg["comparacao"]["modelos"] = {k: cfg["comparacao"]["modelos"][k]
                                    for k in ("regressao_logistica", "arvore_decisao")}
    r = comparar(tmp_path, cfg, registrar=False, doc=None, figuras=None)
    for com_figuras in (False, True):
        texto = escrever_doc(r, com_figuras)
        for proibido in ("desconhecid", "48 h", "premissas do simulador", "desc."):
            assert proibido not in texto, proibido
    assert r["protocolo"]["publico"] == "todos"


def test_publico_desconhecidos_restaura_a_metrica_da_linha_principal():
    from opinion.modelo.treinar import chave_metrica
    cfg = copy.deepcopy(config())
    cfg["publico"] = "todos"
    assert chave_metrica(cfg) == "pr_auc"
    cfg["publico"] = "desconhecidos"
    assert chave_metrica(cfg) == "pr_auc_desconhecidos"


def test_calibracao_so_culpa_class_weight_quando_a_familia_usa(tmp_path):
    from opinion.modelo.comparar import analise, comparar
    _dataset_todos(tmp_path)
    cfg = copy.deepcopy(config())
    cfg["publico"] = "todos"
    cfg["comparacao"]["bootstrap"]["n"] = 20
    cfg["comparacao"]["modelos"] = {k: cfg["comparacao"]["modelos"][k]
                                    for k in ("regressao_logistica", "arvore_decisao")}
    r = comparar(tmp_path, cfg, registrar=False, doc=None, figuras=None)
    fam, b0 = r["_familias"], r["brier_constante_prevalencia"]
    for k in fam:  # força as duas famílias a ficarem mal calibradas
        fam[k]["teste"]["brier"] = b0 + 0.1
    fam["arvore_decisao"]["params"] = {"max_depth": 3, "class_weight": None}
    fam["regressao_logistica"]["params"] = {"C": 1.0, "class_weight": "balanced"}
    cal = next(t for t in analise(r) if t.startswith("**Calibração:**"))
    assert "class_weight=balanced" in cal and "Árvore" not in cal.split("class_weight")[0].split("(em")[-1]
    fam["regressao_logistica"]["params"] = {"C": 1.0, "class_weight": None}
    cal = next(t for t in analise(r) if t.startswith("**Calibração:**"))
    assert "class_weight" not in cal
