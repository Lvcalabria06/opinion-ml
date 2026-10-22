"""
CC3 — baseline explicável (Regressão Logística ou Árvore de Decisão).

Lê data/processed/dataset_modelo.parquet (saída do CC2). Treina no split temporal
"treino", escolhe o candidato e o limiar no "validacao", mede uma vez no "teste".
Compara com a regra simples (compat_score). O vencedor vai para o MLflow.

    python -m opinion.modelo.treinar
    python -m opinion.modelo.treinar --sem-registro     # só avalia, não grava no MLflow
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from opinion.modelo import CONFIG_MODELO, PASTA_PROCESSADOS


def config(caminho=CONFIG_MODELO) -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def carregar(pasta=PASTA_PROCESSADOS) -> tuple[pd.DataFrame, dict]:
    pasta = Path(pasta)
    dm = pd.read_parquet(pasta / "dataset_modelo.parquet")
    meta = json.loads((pasta / "metadados.json").read_text())
    return dm, meta


def mascara_desconhecidos(df: pd.DataFrame) -> pd.Series:
    return df.algum_segue == 0


def metricas(y, p, desc) -> dict:
    y, p, desc = np.asarray(y), np.asarray(p), np.asarray(desc, dtype=bool)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_desconhecidos": float(average_precision_score(y[desc], p[desc])) if desc.any() else None,
        "roc_auc": float(roc_auc_score(y, p)),
        "prevalencia": float(y.mean()),
        "prevalencia_desconhecidos": float(y[desc].mean()) if desc.any() else None,
        "n": int(len(y)),
        "n_desconhecidos": int(desc.sum()),
    }
    return out


def limiar_f1(y, p) -> tuple[float, float]:
    """Limiar em [0.05, 0.95] que maximiza F1. Empate: o menor limiar."""
    y, p = np.asarray(y), np.asarray(p)
    melhor, t_melhor = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, p >= t, zero_division=0)
        if f > melhor:
            melhor, t_melhor = f, float(t)
    return t_melhor, float(melhor)


def _pipeline(tipo: str, params: dict) -> Pipeline:
    if tipo == "logreg":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=0, **params)
        return Pipeline([("escala", StandardScaler()), ("clf", clf)])
    if tipo == "arvore":
        return Pipeline([("clf", DecisionTreeClassifier(random_state=0, **params))])
    raise ValueError(tipo)


def _grade(d: dict) -> list[dict]:
    chaves, valores = list(d), [v if isinstance(v, list) else [v] for v in d.values()]
    out = []
    for combo in itertools.product(*valores):
        p = {k: v for k, v in zip(chaves, combo) if v is not None}
        out.append(p)
    return out or [{}]


def candidatos(cfg: dict) -> list[tuple[str, str, dict]]:
    """(nome, tipo, params) para cada ponto da grade."""
    lista = []
    for nome, c in cfg["candidatos"].items():
        for p in _grade(c.get("grade") or {}):
            lista.append((nome, c["tipo"], p))
    return lista


def explicar(modelo: Pipeline, colunas: list[str], tipo: str) -> str:
    clf = modelo.named_steps["clf"]
    if tipo == "logreg":
        coef = pd.Series(clf.coef_.ravel(), index=colunas).sort_values(key=abs, ascending=False)
        linhas = [f"intercepto {clf.intercept_[0]:+.3f}", ""]
        linhas += [f"{n:>32}  {v:+.3f}" for n, v in coef.items()]
        return "\n".join(linhas) + "\n"
    return export_text(clf, feature_names=list(colunas), max_depth=8)


def treinar(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True) -> dict:
    cfg = cfg or config()
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    for nome, parte in (("treino", tr), ("validacao", va), ("teste", te)):
        if parte.empty:
            raise ValueError(f"split {nome} vazio")

    regra_va = metricas(va.conectou, va.compat_score, mascara_desconhecidos(va))
    regra_te = metricas(te.conectou, te.compat_score, mascara_desconhecidos(te))
    chave = cfg["metrica"]

    resultados = []
    for nome, tipo, params in candidatos(cfg):
        m = _pipeline(tipo, params)
        m.fit(tr[cols], tr.conectou)
        p_va = m.predict_proba(va[cols])[:, 1]
        met_va = metricas(va.conectou, p_va, mascara_desconhecidos(va))
        desc_va = mascara_desconhecidos(va)
        t, f1 = limiar_f1(va.conectou[desc_va], p_va[desc_va.to_numpy()])
        resultados.append({"nome": nome, "tipo": tipo, "params": params, "modelo": m,
                           "validacao": met_va, "limiar": t, "f1_validacao_desconhecidos": f1})

    vencedor = max(resultados, key=lambda r: r["validacao"][chave] or 0)
    p_te = vencedor["modelo"].predict_proba(te[cols])[:, 1]
    met_te = metricas(te.conectou, p_te, mascara_desconhecidos(te))
    desc_te = mascara_desconhecidos(te)
    pred = p_te >= vencedor["limiar"]
    met_te["f1_desconhecidos"] = float(f1_score(te.conectou[desc_te], pred[desc_te.to_numpy()], zero_division=0))
    texto = explicar(vencedor["modelo"], cols, vencedor["tipo"])

    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "algoritmo": vencedor["nome"],
        "tipo": vencedor["tipo"],
        "params": vencedor["params"],
        "limiar": vencedor["limiar"],
        "features": cols,
        "n_features": len(cols),
        "regra_simples": {"validacao": regra_va, "teste": regra_te},
        "vencedor": {"validacao": vencedor["validacao"], "teste": met_te,
                     "f1_validacao_desconhecidos": vencedor["f1_validacao_desconhecidos"]},
        "todos_candidatos": [{"nome": r["nome"], "params": r["params"], "validacao": r["validacao"]}
                             for r in resultados],
        "supera_regra_teste": met_te[chave] > regra_te[chave],
        "supera_regra_teste_geral": met_te["pr_auc"] > regra_te["pr_auc"],
        "explicacao": texto,
        "dataset": {"por_split": meta.get("por_split"), "taxa_conexao": meta.get("taxa_conexao")},
    }
    if not resumo["supera_regra_teste"]:
        raise RuntimeError(
            f"o baseline NÃO superou a regra simples no teste ({chave}: "
            f"{met_te[chave]:.3f} x {regra_te[chave]:.3f}). Checkpoint 2 falhou."
        )

    if registrar:
        resumo.update(_registrar(vencedor["modelo"], resumo, pasta, cfg))
    return resumo


def _registrar(modelo, resumo: dict, pasta: Path, cfg: dict) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    exemplo = dm[dm.split == "treino"][cols].head(5)

    with mlflow.start_run(run_name=f"{resumo['algoritmo']}-{resumo['tipo']}") as run:
        mlflow.log_params({"algoritmo": resumo["algoritmo"], "tipo": resumo["tipo"],
                           "limiar": resumo["limiar"], **{f"hp_{k}": v for k, v in resumo["params"].items()}})
        for split, bloco in (("validacao", resumo["vencedor"]["validacao"]),
                             ("teste", resumo["vencedor"]["teste"])):
            mlflow.log_metrics({f"{split}_{k}": v for k, v in bloco.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({
            "teste_regra_pr_auc": resumo["regra_simples"]["teste"]["pr_auc"],
            "teste_regra_pr_auc_desconhecidos": resumo["regra_simples"]["teste"]["pr_auc_desconhecidos"],
            "delta_pr_auc_desconhecidos": (resumo["vencedor"]["teste"]["pr_auc_desconhecidos"]
                                           - resumo["regra_simples"]["teste"]["pr_auc_desconhecidos"]),
        })
        mlflow.log_dict({k: v for k, v in resumo.items() if k != "explicacao"}, "resumo.json")
        mlflow.log_text(resumo["explicacao"], "explicacao.txt")
        assinatura = infer_signature(exemplo, modelo.predict_proba(exemplo)[:, 1])
        info = mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            signature=assinatura,
            input_example=exemplo,
            registered_model_name=cfg["mlflow"]["modelo_registrado"],
            serialization_format="cloudpickle",
        )
        versao = str(info.registered_model_version)
        MlflowClient().set_registered_model_alias(
            cfg["mlflow"]["modelo_registrado"], cfg["mlflow"]["alias"], versao
        )
        return {"mlflow_run_id": run.info.run_id, "mlflow_modelo": cfg["mlflow"]["modelo_registrado"],
                "mlflow_versao": versao, "mlflow_alias": cfg["mlflow"]["alias"]}


def _imprimir(r: dict) -> None:
    te, regra = r["vencedor"]["teste"], r["regra_simples"]["teste"]
    print(f"Algoritmo: {r['algoritmo']} {r['params']}  limiar={r['limiar']:.2f}")
    print(f"{'':22} {'modelo':>8} {'regra':>8} {'acaso':>8}")
    print(f"{'PR-AUC geral':22} {te['pr_auc']:8.3f} {regra['pr_auc']:8.3f} {te['prevalencia']:8.3f}")
    print(f"{'PR-AUC desconhecidos':22} {te['pr_auc_desconhecidos']:8.3f} "
          f"{regra['pr_auc_desconhecidos']:8.3f} {te['prevalencia_desconhecidos']:8.3f}")
    print("Supera a regra (desconhecidos, teste):", "sim" if r["supera_regra_teste"] else "NÃO")
    if r.get("mlflow_run_id"):
        print(f"MLflow: {r['mlflow_modelo']} v{r['mlflow_versao']} @{r['mlflow_alias']}  run {r['mlflow_run_id']}")


def main():
    ap = argparse.ArgumentParser(description="Treina o baseline CC3 e registra no MLflow")
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    r = treinar(a.pasta, registrar=not a.sem_registro)
    _imprimir(r)


if __name__ == "__main__":
    main()
