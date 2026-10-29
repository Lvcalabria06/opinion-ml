"""
CC3 — baseline explicável (Regressão Logística ou Árvore de Decisão).

Lê data/processed/dataset_modelo.parquet (saída do CC2). Treina no split temporal
"treino", escolhe o candidato e o limiar no "validacao" e só ali decide se o modelo
supera a regra simples (compat_score). O "teste" é medido no fim, só para relatar:
nada é escolhido nem aprovado com ele. O vencedor vai para o MLflow.

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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
                             roc_auc_score)
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


# ======================================================================== métricas
def metricas(y, p, desc, probabilidade: bool = True) -> dict:
    """Métricas de ranking e, se p é probabilidade, de calibração (não dependem do limiar)."""
    y, p, desc = np.asarray(y), np.asarray(p, dtype=float), np.asarray(desc, dtype=bool)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_desconhecidos": float(average_precision_score(y[desc], p[desc])) if desc.any() else None,
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_auc_desconhecidos": float(roc_auc_score(y[desc], p[desc])) if desc.any() else None,
        "prevalencia": float(y.mean()),
        "prevalencia_desconhecidos": float(y[desc].mean()) if desc.any() else None,
        "n": int(len(y)),
        "n_desconhecidos": int(desc.sum()),
    }
    if probabilidade:   # o compat_score da regra é uma pontuação, não probabilidade: Brier não se aplica
        out["brier"] = float(brier_score_loss(y, p))
        out["brier_desconhecidos"] = float(brier_score_loss(y[desc], p[desc])) if desc.any() else None
    return out


def fbeta(precisao, recall, beta: float):
    b2 = beta ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1 + b2) * precisao * recall / (b2 * precisao + recall)
    return np.nan_to_num(f)


def no_limiar(y, p, t: float, beta: float) -> dict:
    """O que acontece quando a pulseira sinaliza p >= t: matriz de confusão e derivadas."""
    y, sinal = np.asarray(y), np.asarray(p) >= t
    vn, fp, fn, vp = (int(v) for v in confusion_matrix(y, sinal, labels=[0, 1]).ravel())
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    return {"limiar": float(t), "precisao": precisao, "recall": recall,
            f"f{beta:g}".replace(".", ""): float(fbeta(np.array(precisao), np.array(recall), beta)),
            "taxa_sinal": float(sinal.mean()), "vp": vp, "fp": fp, "fn": fn, "vn": vn}


def escolher_limiar(y, p, beta: float) -> tuple[float, float]:
    """Limiar que maximiza F-beta (beta < 1 pesa mais a precisão: sinal errado cansa o usuário)."""
    precisao, recall, limiares = precision_recall_curve(np.asarray(y), np.asarray(p))
    f = fbeta(precisao[:-1], recall[:-1], beta)
    i = int(np.argmax(f))
    return float(limiares[i]), float(f[i])


# ======================================================================== candidatos
def _pipeline(tipo: str, params: dict) -> Pipeline:
    if tipo == "logreg":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=0, **params)
        return Pipeline([("escala", StandardScaler()), ("clf", clf)])
    if tipo == "arvore":
        return Pipeline([("clf", DecisionTreeClassifier(random_state=0, **params))])
    if tipo == "random_forest":
        return Pipeline([("clf", RandomForestClassifier(n_jobs=-1, random_state=0, **params))])
    if tipo == "hist_gb":
        return Pipeline([("clf", HistGradientBoostingClassifier(random_state=0, **params))])
    raise ValueError(tipo)


def _grade(d: dict) -> list[dict]:
    chaves, valores = list(d), [v if isinstance(v, list) else [v] for v in d.values()]
    out = []
    for combo in itertools.product(*valores):
        p = {k: v for k, v in zip(chaves, combo) if v is not None}
        out.append(p)
    return out or [{}]


def candidatos(cfg: dict, secao: dict | None = None) -> list[tuple[str, str, dict]]:
    """(nome, tipo, params) para cada ponto da grade."""
    lista = []
    for nome, c in (secao or cfg["candidatos"]).items():
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
    if tipo == "arvore":
        return export_text(clf, feature_names=list(colunas), max_depth=8)
    return ""   # floresta e boosting: a explicação vem da importância por permutação (opinion.modelo.comparar)


# ======================================================================== protocolo
def avaliar_regra(va, te, desc_va, desc_te, beta: float) -> dict:
    """Regra simples no mesmo protocolo: limiar do compat_score escolhido na validação, entre desconhecidos."""
    limiar, _ = escolher_limiar(va.conectou[desc_va], va.compat_score[desc_va], beta)
    regra = {"validacao": metricas(va.conectou, va.compat_score, desc_va, probabilidade=False),
             "teste": metricas(te.conectou, te.compat_score, desc_te, probabilidade=False), "limiar": limiar}
    for split, df, d in (("validacao", va, desc_va), ("teste", te, desc_te)):
        regra[split]["no_limiar_desconhecidos"] = no_limiar(df.conectou[d], df.compat_score[d], limiar, beta)
    return regra


def avaliar_candidato(nome, tipo, params, tr, va, cols, desc_va, beta: float) -> dict:
    """Ajusta no treino; métricas e limiar (F-beta entre desconhecidos) na validação."""
    m = _pipeline(tipo, params)
    m.fit(tr[cols], tr.conectou)
    p_va = m.predict_proba(va[cols])[:, 1]
    t, _ = escolher_limiar(va.conectou[desc_va], p_va[desc_va], beta)
    met_va = metricas(va.conectou, p_va, desc_va)
    met_va["no_limiar_desconhecidos"] = no_limiar(va.conectou[desc_va], p_va[desc_va], t, beta)
    return {"nome": nome, "tipo": tipo, "params": params, "modelo": m, "validacao": met_va, "limiar": t}


def medir_teste(r: dict, te, cols, desc_te, beta: float):
    """A medição única no teste — só para relatar, nada é escolhido com ela."""
    p_te = r["modelo"].predict_proba(te[cols])[:, 1]
    met = metricas(te.conectou, p_te, desc_te)
    met["no_limiar_desconhecidos"] = no_limiar(te.conectou[desc_te], p_te[desc_te], r["limiar"], beta)
    met["no_limiar_todos"] = no_limiar(te.conectou, p_te, r["limiar"], beta)
    return p_te, met


# ======================================================================== treino
def treinar(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True) -> dict:
    cfg = cfg or config()
    beta = float(cfg["limiar"]["beta"])
    nome_f = f"f{beta:g}".replace(".", "")
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    for nome, parte in (("treino", tr), ("validacao", va), ("teste", te)):
        if parte.empty:
            raise ValueError(f"split {nome} vazio")
    desc_va, desc_te = mascara_desconhecidos(va).to_numpy(), mascara_desconhecidos(te).to_numpy()
    chave = cfg["metrica"]

    regra = avaliar_regra(va, te, desc_va, desc_te, beta)
    resultados = [avaliar_candidato(nome, tipo, params, tr, va, cols, desc_va, beta)
                  for nome, tipo, params in candidatos(cfg)]

    vencedor = max(resultados, key=lambda r: r["validacao"][chave] or 0)
    v, rv = vencedor["validacao"][chave], regra["validacao"][chave]
    if not (v > rv and not np.isclose(v, rv)):          # empate (ou diferença de arredondamento) não conta
        raise RuntimeError(
            f"o melhor candidato NÃO superou a regra simples na validação ({chave}: "
            f"{vencedor['validacao'][chave]:.3f} x {regra['validacao'][chave]:.3f}). Baseline não se justifica."
        )

    p_te, met_te = medir_teste(vencedor, te, cols, desc_te, beta)

    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "algoritmo": vencedor["nome"],
        "tipo": vencedor["tipo"],
        "params": vencedor["params"],
        "limiar": vencedor["limiar"],
        "criterio_limiar": f"max {nome_f} entre desconhecidos na validação",
        "features": cols,
        "n_features": len(cols),
        "regra_simples": regra,
        "vencedor": {"validacao": vencedor["validacao"], "teste": met_te},
        "todos_candidatos": [{"nome": r["nome"], "params": r["params"], "limiar": r["limiar"],
                              "validacao": r["validacao"]} for r in resultados],
        "supera_regra_validacao": True,
        "supera_regra_teste": met_te[chave] > regra["teste"][chave],
        "supera_regra_teste_geral": met_te["pr_auc"] > regra["teste"]["pr_auc"],
        "explicacao": explicar(vencedor["modelo"], cols, vencedor["tipo"]),
        "dataset": {"por_split": meta.get("por_split"), "taxa_conexao": meta.get("taxa_conexao")},
        "_modelo": vencedor["modelo"],
    }
    if registrar:
        preds = {"y": te.conectou.to_numpy(), "p": p_te, "regra": te.compat_score.to_numpy(), "desc": desc_te}
        resumo.update(_registrar(vencedor["modelo"], resumo, preds, pasta, cfg))
    return resumo


# ======================================================================== gráficos
def _graficos(preds: dict, limiar: float, limiar_regra: float) -> dict:
    """Figuras do teste: curvas PR e ROC (modelo x regra), calibração e matriz de confusão."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    y, p, r, d = preds["y"], preds["p"], preds["regra"], preds["desc"]
    grupos = [("todos", np.ones_like(d, dtype=bool)), ("desconhecidos", d)]
    figs = {}

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (nome, m) in zip(axs, grupos):
        for rot, s in (("modelo", p), ("regra simples", r)):
            pr, rc, _ = precision_recall_curve(y[m], s[m])
            ax.plot(rc, pr, label=f"{rot} (PR-AUC {average_precision_score(y[m], s[m]):.3f})")
        ax.axhline(y[m].mean(), ls="--", c="gray", label=f"acaso ({y[m].mean():.3f})")
        ax.set(title=f"Curva PR — teste, {nome}", xlabel="recall", ylabel="precisão", ylim=(0, 1))
        ax.legend(fontsize=8)
    figs["graficos/curvas_pr.png"] = fig

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (nome, m) in zip(axs, grupos):
        for rot, s in (("modelo", p), ("regra simples", r)):
            fpr, tpr, _ = roc_curve(y[m], s[m])
            ax.plot(fpr, tpr, label=f"{rot} (ROC-AUC {roc_auc_score(y[m], s[m]):.3f})")
        ax.plot([0, 1], [0, 1], ls="--", c="gray")
        ax.set(title=f"Curva ROC — teste, {nome}", xlabel="taxa de falso positivo", ylabel="recall")
        ax.legend(fontsize=8)
    figs["graficos/curvas_roc.png"] = fig

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for nome, m in grupos:
        fp_, mp_ = calibration_curve(y[m], p[m], n_bins=10, strategy="quantile")
        ax.plot(mp_, fp_, marker="o", label=f"{nome} (Brier {brier_score_loss(y[m], p[m]):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="calibração perfeita")
    ax.set(title="Calibração — teste", xlabel="probabilidade prevista", ylabel="taxa real de conexão")
    ax.legend(fontsize=8)
    figs["graficos/calibracao.png"] = fig

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (rot, s, t) in zip(axs, (("modelo", p, limiar), ("regra simples", r, limiar_regra))):
        cm = confusion_matrix(y[d], s[d] >= t, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
        ax.set(title=f"{rot} — desconhecidos, limiar {t:.3f}", xticks=[0, 1], yticks=[0, 1],
               xticklabels=["sem sinal", "sinal"], yticklabels=["não conectou", "conectou"])
    figs["graficos/matriz_confusao_desconhecidos.png"] = fig
    return figs


def _metricas_planas(prefixo: str, bloco: dict) -> dict:
    out = {}
    for k, v in bloco.items():
        if isinstance(v, dict):
            out.update(_metricas_planas(f"{prefixo}_{k}", v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[f"{prefixo}_{k}"] = v
    return out


def _registrar(modelo, resumo: dict, preds: dict, pasta: Path, cfg: dict) -> dict:
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    dm, meta = carregar(pasta)
    exemplo = dm[dm.split == "treino"][meta["colunas_features"]].head(5)
    regra = resumo["regra_simples"]

    with mlflow.start_run(run_name=f"{resumo['algoritmo']}-{resumo['tipo']}") as run:
        mlflow.log_params({"algoritmo": resumo["algoritmo"], "tipo": resumo["tipo"],
                           "limiar": resumo["limiar"], "limiar_regra": regra["limiar"],
                           "beta_limiar": cfg["limiar"]["beta"], "metrica_escolha": cfg["metrica"],
                           "n_features": resumo["n_features"],
                           **{f"hp_{k}": v for k, v in resumo["params"].items()}})
        for split in ("validacao", "teste"):
            mlflow.log_metrics(_metricas_planas(split, resumo["vencedor"][split]))
            mlflow.log_metrics(_metricas_planas(f"{split}_regra", regra[split]))
        mlflow.log_metric("delta_pr_auc_desconhecidos_teste",
                          resumo["vencedor"]["teste"]["pr_auc_desconhecidos"] - regra["teste"]["pr_auc_desconhecidos"])
        mlflow.log_dict({k: v for k, v in resumo.items() if k != "explicacao" and not k.startswith("_")},
                        "resumo.json")
        mlflow.log_text(resumo["explicacao"], "explicacao.txt")
        for caminho, fig in _graficos(preds, resumo["limiar"], regra["limiar"]).items():
            mlflow.log_figure(fig, caminho)
            plt.close(fig)
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
    va, te, regra = r["vencedor"]["validacao"], r["vencedor"]["teste"], r["regra_simples"]
    print(f"Algoritmo: {r['algoritmo']} {r['params']}  limiar={r['limiar']:.3f}  ({r['criterio_limiar']})")
    print(f"{'':28} {'validação':>10} {'teste':>8} {'regra(te)':>10} {'acaso(te)':>10}")
    print(f"{'PR-AUC geral':28} {va['pr_auc']:10.3f} {te['pr_auc']:8.3f} {regra['teste']['pr_auc']:10.3f} "
          f"{te['prevalencia']:10.3f}")
    print(f"{'PR-AUC desconhecidos':28} {va['pr_auc_desconhecidos']:10.3f} {te['pr_auc_desconhecidos']:8.3f} "
          f"{regra['teste']['pr_auc_desconhecidos']:10.3f} {te['prevalencia_desconhecidos']:10.3f}")
    print(f"{'ROC-AUC desconhecidos':28} {va['roc_auc_desconhecidos']:10.3f} {te['roc_auc_desconhecidos']:8.3f} "
          f"{regra['teste']['roc_auc_desconhecidos']:10.3f} {0.5:10.3f}")
    print(f"{'Brier (todos)':28} {va['brier']:10.3f} {te['brier']:8.3f}")
    nl, nr = te["no_limiar_desconhecidos"], regra["teste"]["no_limiar_desconhecidos"]
    f = next(k for k in nl if k.startswith("f") and k[1:].isdigit())
    print(f"No limiar, desconhecidos (teste):  modelo  precisão {nl['precisao']:.3f}  recall {nl['recall']:.3f}  "
          f"{f} {nl[f]:.3f}  sinal em {nl['taxa_sinal']:.1%}")
    print(f"{'':35}regra   precisão {nr['precisao']:.3f}  recall {nr['recall']:.3f}  "
          f"{f} {nr[f]:.3f}  sinal em {nr['taxa_sinal']:.1%}")
    print(f"Matriz de confusão do modelo (desconhecidos, teste): VP {nl['vp']}  FP {nl['fp']}  "
          f"FN {nl['fn']}  VN {nl['vn']}")
    print("Supera a regra na validação (critério):", "sim" if r["supera_regra_validacao"] else "NÃO",
          "| no teste (só relato):", "sim" if r["supera_regra_teste"] else "NÃO")
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
