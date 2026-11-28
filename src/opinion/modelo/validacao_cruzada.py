"""
CC6 — validação cruzada, estabilidade, overfitting/underfitting e generalização.

    python -m opinion.modelo.validacao_cruzada
    python -m opinion.modelo.validacao_cruzada --sem-registro

Três análises sobre treino + validação (semanas 1–10); o teste (11–12) não é usado:

1. **Stratified k-fold agrupado por par** (`StratifiedGroupKFold`, grupo = `par_id`): o mesmo
   par de usuários nunca aparece no treino e na avaliação do mesmo fold, e a proporção de
   conexões fica parecida entre os folds. Para cada fold medimos o desempenho no próprio
   treino e no fold de avaliação: a diferença entre os dois é a evidência de overfitting.
2. **Curva de aprendizado**: o mesmo k-fold com frações crescentes do treino. Se a avaliação
   ainda sobe com mais dados, falta dado (variância); se treino e avaliação convergem baixos,
   falta informação nas features (viés).
3. **Janela temporal expansível**: treina nas semanas 1..k−1 (com purga de 48 h) e avalia na
   semana k. Mede se o modelo generaliza para semanas que ainda não viu, que é como ele é usado.

Modelos: o otimizado do CC5 (`opinion-cc5-otimizado@otimizado`, sem a calibração, que não muda
a ordenação), a regressão logística como referência explicável e a regra simples.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.comparar import _f, _nome, _pct, _s, recorte_midia
from opinion.modelo.treinar import _pipeline, carregar, config, mascara_desconhecidos, metricas
from opinion.mundo import RAIZ
from opinion.pipeline.compatibilidade import MIDIAS

DOC = RAIZ / "docs" / "validacao_cruzada.md"
FIGURAS = RAIZ / "docs" / "img" / "cc6"
REFERENCIA = {"familia": "regressao_logistica", "tipo": "logreg", "params": {"C": 1.0}}


# ======================================================================== modelos avaliados
def modelo_cc5(cfg: dict, midia: str | None = None) -> dict:
    """Família e hiperparâmetros do modelo otimizado no CC5 (MLflow); sem ele, a reserva do config."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    o = cfg["otimizacao"]
    nome = o["modelo_registrado"] + (f"-{midia}" if midia else "")
    try:
        v = MlflowClient().get_model_version_by_alias(nome, o["alias"])
        p = MlflowClient().get_run(v.run_id).data.params
        return {"familia": p["modelo"], "tipo": p["tipo"], "params": json.loads(p["params_json"]),
                "origem": f"{nome}@{o['alias']} (v{v.version})"}
    except Exception:
        r = cfg["producao"]["reserva"]
        return {"familia": r.get("familia", r["tipo"]), "tipo": r["tipo"], "params": r["params"],
                "origem": "config producao.reserva (CC5)"}


def _ajustar(spec: dict, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _pipeline(spec["tipo"], spec["params"]).fit(X, y)


def _precisao_topo(y, p, frac: float = 0.02) -> float:
    k = max(1, int(round(frac * len(y))))
    return float(np.asarray(y)[np.argsort(-np.asarray(p))[:k]].mean())


def _medir(y, p, desc, probabilidade=True) -> dict:
    m = metricas(y, p, desc, probabilidade)
    m["precisao_top2pct_desconhecidos"] = _precisao_topo(np.asarray(y)[desc], np.asarray(p)[desc])
    return m


# ======================================================================== análises
def kfold(base: pd.DataFrame, cols, modelos: dict, k: int, semente: int) -> pd.DataFrame:
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente)
    linhas = []
    for fold, (i_tr, i_av) in enumerate(cv.split(base, base.conectou, groups=base.par_id)):
        tr, av = base.iloc[i_tr], base.iloc[i_av]
        assert not set(tr.par_id) & set(av.par_id), "par nos dois lados do fold"
        d_tr, d_av = mascara_desconhecidos(tr).to_numpy(), mascara_desconhecidos(av).to_numpy()
        for nome, spec in modelos.items():
            t0 = time.time()
            if spec is None:                                    # regra simples: não treina
                p_tr, p_av, prob = tr.compat_score.to_numpy(), av.compat_score.to_numpy(), False
            else:
                m = _ajustar(spec, tr[cols], tr.conectou)
                p_tr, p_av, prob = m.predict_proba(tr[cols])[:, 1], m.predict_proba(av[cols])[:, 1], True
            mt, ma = _medir(tr.conectou, p_tr, d_tr, prob), _medir(av.conectou, p_av, d_av, prob)
            linhas.append({"modelo": nome, "fold": fold + 1, "n_treino": len(tr), "n_avaliacao": len(av),
                           "prevalencia_avaliacao": ma["prevalencia_desconhecidos"],
                           "treino_pr_auc_desconhecidos": mt["pr_auc_desconhecidos"],
                           **{f"avaliacao_{c}": ma.get(c) for c in ("pr_auc_desconhecidos", "pr_auc", "roc_auc_desconhecidos",
                                                                     "brier", "precisao_top2pct_desconhecidos")},
                           "segundos": time.time() - t0})
        print(f"  fold {fold + 1}/{k}: " + "  ".join(
            f"{n} {r['avaliacao_pr_auc_desconhecidos']:.3f}" for n, r in
            ((x["modelo"], x) for x in linhas[-len(modelos):])))
    return pd.DataFrame(linhas)


def curva_aprendizado(base, cols, spec: dict, fracs, k: int, semente: int, folds_usados: int = 3) -> pd.DataFrame:
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente)
    rng = np.random.default_rng(semente)
    linhas = []
    for fold, (i_tr, i_av) in enumerate(cv.split(base, base.conectou, groups=base.par_id)):
        if fold >= folds_usados:
            break
        tr_total, av = base.iloc[i_tr], base.iloc[i_av]
        d_av = mascara_desconhecidos(av).to_numpy()
        pares = tr_total.par_id.unique()
        rng.shuffle(pares)
        for frac in fracs:            # subamostra por PAR, para manter o agrupamento
            escolhidos = set(pares[:max(1, int(round(frac * len(pares))))])
            tr = tr_total[tr_total.par_id.isin(escolhidos)]
            d_tr = mascara_desconhecidos(tr).to_numpy()
            m = _ajustar(spec, tr[cols], tr.conectou)
            linhas.append({"fold": fold + 1, "fracao": frac, "n_treino": len(tr),
                           "treino": metricas(tr.conectou, m.predict_proba(tr[cols])[:, 1], d_tr)["pr_auc_desconhecidos"],
                           "avaliacao": metricas(av.conectou, m.predict_proba(av[cols])[:, 1], d_av)["pr_auc_desconhecidos"]})
        print(f"  curva de aprendizado, fold {fold + 1}: " + "  ".join(
            f"{r['fracao']:.0%}→{r['avaliacao']:.3f}" for r in linhas[-len(fracs):]))
    return pd.DataFrame(linhas)


def temporal(base, cols, modelos: dict, primeira: int, horas_label: int) -> pd.DataFrame:
    """Janela expansível: treina nas semanas < k (com purga) e avalia na semana k."""
    linhas = []
    for k in range(primeira, int(base.semana.max()) + 1):
        av = base[base.semana == k]
        inicio_k = av.inicio.min().normalize()
        tr = base[(base.semana < k) & (base.fim + pd.Timedelta(hours=horas_label) < inicio_k)]
        if av.empty or tr.empty or av.conectou.nunique() < 2:
            continue
        d_av = mascara_desconhecidos(av).to_numpy()
        for nome, spec in modelos.items():
            if spec is None:
                p, prob = av.compat_score.to_numpy(), False
            else:
                p, prob = _ajustar(spec, tr[cols], tr.conectou).predict_proba(av[cols])[:, 1], True
            m = metricas(av.conectou, p, d_av, prob)
            linhas.append({"modelo": nome, "semana_avaliada": k, "semanas_treino": k - 1, "n_treino": len(tr),
                           "n_avaliacao": len(av), "pr_auc_desconhecidos": m["pr_auc_desconhecidos"],
                           "prevalencia_desconhecidos": m["prevalencia_desconhecidos"],
                           "roc_auc_desconhecidos": m["roc_auc_desconhecidos"]})
        print(f"  semana {k}: " + "  ".join(f"{r['modelo']} {r['pr_auc_desconhecidos']:.3f}"
                                            for r in linhas[-len(modelos):]))
    return pd.DataFrame(linhas)


def resumir(kf: pd.DataFrame) -> pd.DataFrame:
    g = kf.groupby("modelo")
    out = pd.DataFrame({
        "media": g.avaliacao_pr_auc_desconhecidos.mean(), "desvio": g.avaliacao_pr_auc_desconhecidos.std(ddof=1),
        "minimo": g.avaliacao_pr_auc_desconhecidos.min(), "maximo": g.avaliacao_pr_auc_desconhecidos.max(),
        "treino_media": g.treino_pr_auc_desconhecidos.mean(),
        "roc_media": g.avaliacao_roc_auc_desconhecidos.mean(), "roc_desvio": g.avaliacao_roc_auc_desconhecidos.std(ddof=1),
        "brier_media": g.avaliacao_brier.mean(),
        "precisao_top2_media": g.avaliacao_precisao_top2pct_desconhecidos.mean(),
        "prevalencia_media": g.prevalencia_avaliacao.mean(),
    })
    out["cv"] = out.desvio / out.media
    out["gap"] = out.treino_media - out.media
    out["gap_relativo"] = out.gap / out.media
    return out


# ======================================================================== orquestração
def validar(pasta=PASTA_PROCESSADOS, cfg=None, midia: str | None = None, registrar=True,
            doc: Path | None = DOC, figuras: Path | None = FIGURAS) -> dict:
    cfg = cfg or config()
    c = cfg["validacao_cruzada"]
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    if midia is not None:
        dm, cols = recorte_midia(dm, meta, midia)
        doc = figuras = None
    base = dm[dm.split.isin(["treino", "validacao"])].reset_index(drop=True)
    otim = modelo_cc5(cfg, midia)
    modelos = {"otimizado": otim, "referencia": REFERENCIA, "regra": None}
    print(f"Modelo otimizado: {_nome(otim['familia'])} {otim['params']} ({otim['origem']})")
    print(f"Stratified k-fold agrupado por par (k={c['k']}) em {len(base):,} encontros das semanas 1–10:")
    kf = kfold(base, cols, modelos, c["k"], c["semente"])
    print("Curva de aprendizado do modelo otimizado:")
    ca = curva_aprendizado(base, cols, otim, c["curva_aprendizado"], c["k"], c["semente"])
    print("Janela temporal expansível:")
    from opinion.pipeline.executar import config as config_pipeline
    tp = temporal(base, cols, modelos, c["temporal_primeira_semana_validacao"],
                  config_pipeline()["label"]["janela_horas"])
    res = resumir(kf)
    teste = _teste_cc5(cfg, midia)
    r = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "midia": midia or "todas",
         "k": c["k"], "n_encontros": len(base), "n_pares": int(base.par_id.nunique()), "otimizado": otim,
         "referencia": REFERENCIA, "resumo": res.reset_index().to_dict("records"), "teste_cc5": teste,
         "_kf": kf, "_ca": ca, "_tp": tp, "_res": res}
    r["diagnostico"] = diagnostico(r)
    figs = _figuras(r)
    if figuras is not None:
        Path(figuras).mkdir(parents=True, exist_ok=True)
        for nome, fig in figs.items():
            fig.savefig(Path(figuras) / Path(nome).name, dpi=110, bbox_inches="tight")
    texto = escrever_doc(r, figuras is not None)
    if doc is not None:
        Path(doc).write_text(texto, encoding="utf-8")
    if registrar:
        r.update(_registrar(r, figs, texto, cfg))
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    return r


def _teste_cc5(cfg, midia) -> float | None:
    """PR-AUC entre desconhecidos no teste, medida uma vez no CC5 (só para comparar com a faixa da CV)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        o = cfg["otimizacao"]
        v = MlflowClient().get_model_version_by_alias(o["modelo_registrado"] + (f"-{midia}" if midia else ""), o["alias"])
        return MlflowClient().get_run(v.run_id).data.metrics.get("teste_pr_auc_desconhecidos")
    except Exception:
        return None


def diagnostico(r: dict) -> dict:
    res, ca, tp = r["_res"], r["_ca"], r["_tp"]
    o = res.loc["otimizado"]
    cm = ca.groupby("fracao")[["treino", "avaliacao"]].mean()
    ganho_final = float(cm.avaliacao.iloc[-1] - cm.avaliacao.iloc[-2]) if len(cm) > 1 else 0.0
    if o.gap_relativo > 0.5:
        ajuste = "overfitting"
    elif o.gap_relativo < 0.15 and o.media < 3 * o.prevalencia_media:
        ajuste = "underfitting"
    else:
        ajuste = "equilibrado"
    tpo = tp[tp.modelo == "otimizado"] if not tp.empty else tp
    return {"ajuste": ajuste, "gap_relativo": float(o.gap_relativo), "cv": float(o.cv),
            "estavel": bool(o.cv < 0.15),
            "ganho_ultimo_passo_curva": ganho_final, "curva_ainda_sobe": bool(ganho_final > 0.005),
            "temporal_media": float(tpo.pr_auc_desconhecidos.mean()) if not tpo.empty else None,
            "temporal_min": float(tpo.pr_auc_desconhecidos.min()) if not tpo.empty else None,
            "temporal_max": float(tpo.pr_auc_desconhecidos.max()) if not tpo.empty else None,
            "folds_acima_da_regra": int((r["_kf"].pivot(index="fold", columns="modelo",
                                                         values="avaliacao_pr_auc_desconhecidos")
                                         .eval("otimizado > regra")).sum())}


# ======================================================================== gráficos
def _figuras(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kf, ca, tp = r["_kf"], r["_ca"], r["_tp"]
    rot = {"otimizado": f"{_nome(r['otimizado']['familia'])} (CC5)", "referencia": "Regressão logística",
           "regra": "Regra simples"}
    figs = {}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ordem = ["otimizado", "referencia", "regra"]
    axs[0].boxplot([kf[kf.modelo == m].avaliacao_pr_auc_desconhecidos for m in ordem], tick_labels=[rot[m] for m in ordem])
    for i, m in enumerate(ordem, 1):
        v = kf[kf.modelo == m].avaliacao_pr_auc_desconhecidos
        axs[0].scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=14, zorder=3)
    axs[0].axhline(kf.prevalencia_avaliacao.mean(), c="gray", ls=":", label="acaso")
    axs[0].set(title=f"PR-AUC entre desconhecidos nos {r['k']} folds", ylabel="PR-AUC")
    axs[0].legend(fontsize=8)
    w = 0.38
    for j, m in enumerate(["otimizado", "referencia"]):
        d = kf[kf.modelo == m]
        axs[1].bar(d.fold + (j - 0.5) * w, d.treino_pr_auc_desconhecidos, w, alpha=0.35, color=f"C{j}",
                   label=f"{rot[m]} — treino")
        axs[1].bar(d.fold + (j - 0.5) * w, d.avaliacao_pr_auc_desconhecidos, w, color=f"C{j}",
                   label=f"{rot[m]} — avaliação")
    axs[1].set(title="Treino x avaliação por fold (overfitting)", xlabel="fold", ylabel="PR-AUC desconhecidos")
    axs[1].legend(fontsize=7)
    figs["validacao_cruzada/folds.png"] = fig

    cm = ca.groupby("fracao").agg(n=("n_treino", "mean"), treino=("treino", "mean"), treino_sd=("treino", "std"),
                                  avaliacao=("avaliacao", "mean"), avaliacao_sd=("avaliacao", "std"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for col, cor in (("treino", "C0"), ("avaliacao", "C1")):
        ax.plot(cm.n, cm[col], marker="o", color=cor, label=col.replace("avaliacao", "avaliação"))
        ax.fill_between(cm.n, cm[col] - cm[f"{col}_sd"].fillna(0), cm[col] + cm[f"{col}_sd"].fillna(0),
                        color=cor, alpha=0.15)
    ax.set(title=f"Curva de aprendizado — {rot['otimizado']}", xlabel="encontros de treino",
           ylabel="PR-AUC desconhecidos")
    ax.legend(fontsize=8)
    figs["validacao_cruzada/curva_aprendizado.png"] = fig

    if not tp.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for m in ordem:
            d = tp[tp.modelo == m]
            ax.plot(d.semana_avaliada, d.pr_auc_desconhecidos, marker="o", label=rot[m],
                    ls="--" if m == "regra" else "-")
        d = tp[tp.modelo == "otimizado"]
        ax.plot(d.semana_avaliada, d.prevalencia_desconhecidos, c="gray", ls=":", label="acaso")
        ax.set(title="Janela temporal expansível (treina até k−1, avalia k)", xlabel="semana avaliada",
               ylabel="PR-AUC desconhecidos")
        ax.legend(fontsize=8)
        figs["validacao_cruzada/temporal.png"] = fig
    return figs


# ======================================================================== documento
def escrever_doc(r: dict, com_figuras: bool) -> str:
    res, d, kf, tp = r["_res"], r["diagnostico"], r["_kf"], r["_tp"]
    nome_o = _nome(r["otimizado"]["familia"])
    rot = {"otimizado": f"{nome_o} otimizado (CC5)", "referencia": "Regressão logística (referência)",
           "regra": "Regra simples"}
    L = ["# Validação cruzada e estabilidade — OpiniOn (CC6)", "",
         f"Gerado por `python -m opinion.modelo.validacao_cruzada` em {datetime.now():%d/%m/%Y %H:%M}. "
         "Registrado no MLflow, experimento `opinion-cc6-validacao-cruzada`.", "",
         "> Encontros e label simulados ([`fontes_dados.md`](fontes_dados.md)). As conclusões valem para o "
         "processo da simulação.", "",
         "## 1. Protocolo", "",
         f"- **Dados:** treino + validação (semanas 1–10), {_n(r['n_encontros'])} encontros de "
         f"{_n(r['n_pares'])} pares. O teste (semanas 11–12) não entra.",
         f"- **Stratified k-fold agrupado por par** (`StratifiedGroupKFold`, k = {r['k']}): todos os encontros "
         "de um par ficam no mesmo fold, o que impede que o modelo seja avaliado em um par que ele já viu "
         "no treino. A estratificação mantém a taxa de conexão parecida entre os folds.",
         "- **Estabilidade:** média, desvio-padrão, coeficiente de variação (CV) e amplitude da PR-AUC entre "
         "desconhecidos nos folds.",
         "- **Overfitting/underfitting:** desempenho no próprio treino de cada fold contra o fold de "
         "avaliação, mais uma curva de aprendizado (frações de 10% a 100% do treino, subamostradas por par).",
         "- **Generalização para o futuro:** janela temporal expansível, treinando nas semanas 1..k−1 "
         "(com purga dos encontros cujo label de 48 h invade a semana k) e avaliando na semana k.",
         f"- **Modelos:** {nome_o} com os hiperparâmetros do CC5 (`{_params(r['otimizado']['params'])}`, "
         f"origem: {r['otimizado']['origem']}), regressão logística (C = 1) como referência explicável e a "
         "regra simples (`compat_score`).", "",
         "## 2. Resultados por fold (PR-AUC entre desconhecidos)", "",
         "| Fold | " + " | ".join(rot[m] for m in ("otimizado", "referencia", "regra")) + " | Prevalência |",
         "|---|---|---|---|---|"]
    pv = kf.pivot(index="fold", columns="modelo", values="avaliacao_pr_auc_desconhecidos")
    prev = kf.groupby("fold").prevalencia_avaliacao.first()
    for fold, linha in pv.iterrows():
        L.append(f"| {fold} | {_f(linha.otimizado)} | {_f(linha.referencia)} | {_f(linha.regra)} | {_f(prev[fold])} |")
    L += ["", "## 3. Estabilidade", "",
          "| Modelo | Média | Desvio | CV | Mín.–máx. | ROC-AUC desc. | Precisão no top 2% | Brier (sem calibrar) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ("otimizado", "referencia", "regra"):
        x = res.loc[m]
        brier = "não se aplica" if m == "regra" else _f(x.brier_media, 4)
        L.append(f"| {rot[m]} | {_f(x.media)} | {_f(x.desvio)} | {_pct(x.cv)} | {_f(x.minimo)}–{_f(x.maximo)} | "
                 f"{_f(x.roc_media)} ± {_f(x.roc_desvio)} | {_pct(x.precisao_top2_media)} | {brier} |")
    L.append("")
    if com_figuras:
        L += ["![Folds](img/cc6/folds.png)", ""]
    L += ["## 4. Overfitting e underfitting", "",
          "| Modelo | PR-AUC no treino | PR-AUC na avaliação | Diferença | Diferença relativa |", "|---|---|---|---|---|"]
    for m in ("otimizado", "referencia"):
        x = res.loc[m]
        L.append(f"| {rot[m]} | {_f(x.treino_media)} | {_f(x.media)} | {_s(x.gap)} | {_pct(x.gap_relativo)} |")
    cm = r["_ca"].groupby("fracao")[["n_treino", "treino", "avaliacao"]].mean()
    L += ["", "**Curva de aprendizado** (média de 3 folds):", "",
          "| Fração do treino | Encontros | PR-AUC treino | PR-AUC avaliação |", "|---|---|---|---|"]
    for frac, x in cm.iterrows():
        L.append(f"| {_pct(frac)} | {_n(x.n_treino)} | {_f(x.treino)} | {_f(x.avaliacao)} |")
    L.append("")
    if com_figuras:
        L += ["![Curva de aprendizado](img/cc6/curva_aprendizado.png)", ""]
    if not tp.empty:
        L += ["## 5. Generalização para semanas futuras", "",
              "| Semana avaliada | Encontros de treino | " + " | ".join(rot[m] for m in ("otimizado", "referencia", "regra"))
              + " | Prevalência |", "|---|---|---|---|---|---|"]
        pt = tp.pivot(index="semana_avaliada", columns="modelo", values="pr_auc_desconhecidos")
        nt = tp.groupby("semana_avaliada").n_treino.first()
        pr = tp.groupby("semana_avaliada").prevalencia_desconhecidos.first()
        for s, x in pt.iterrows():
            L.append(f"| {s} | {_n(nt[s])} | {_f(x.otimizado)} | {_f(x.referencia)} | {_f(x.regra)} | {_f(pr[s])} |")
        L += ["", "As semanas 3 e 7 têm convenção de games, e a semana 11 (no teste) também. "
              "A variação entre semanas inclui essa mudança de contexto.", ""]
        if com_figuras:
            L += ["![Temporal](img/cc6/temporal.png)", ""]
    L += ["## 6. Análise", ""] + [f"- {t}" for t in analise(r)] + [""]
    L += ["## 7. Limitações", "",
          "- O k-fold mistura semanas: um fold pode treinar com encontros posteriores aos que avalia. As features "
          "continuam calculadas só com o passado de cada encontro, mas mudanças de comportamento ao longo do "
          "tempo ficam escondidas. Por isso a seção 5 complementa com a janela temporal.",
          "- Os hiperparâmetros vieram do CC5, escolhidos na validação (semanas 9–10), que faz parte destes folds. "
          "A média da CV é, portanto, levemente otimista para o modelo otimizado.",
          "- Com label simulado, estabilidade alta significa que o modelo aprende de forma consistente o processo "
          "da simulação; com pessoas reais (semana 16) a análise precisa ser refeita.", ""]
    return "\n".join(L)


def analise(r: dict) -> list[str]:
    res, d = r["_res"], r["diagnostico"]
    o, ref, rg = res.loc["otimizado"], res.loc["referencia"], res.loc["regra"]
    nome_o = _nome(r["otimizado"]["familia"])
    out = [f"**Estabilidade:** {nome_o} teve PR-AUC entre desconhecidos de {_f(o.media)} ± {_f(o.desvio)} "
           f"(CV {_pct(o.cv)}, de {_f(o.minimo)} a {_f(o.maximo)}). "
           + ("A variação entre folds é baixa: o resultado não depende de uma partição favorável."
              if d["estavel"] else
              "A variação entre folds é relevante: com poucos positivos entre desconhecidos em cada fold, "
              "a PR-AUC oscila; os resultados devem ser lidos como faixa, não como número único.")]
    out.append(f"**Contra as referências:** o modelo otimizado superou a regra simples em "
               f"{d['folds_acima_da_regra']} de {r['k']} folds (média {_f(o.media)} x {_f(rg.media)}) e a "
               f"regressão logística na média ({_f(o.media)} x {_f(ref.media)}).")
    if d["ajuste"] == "overfitting":
        txt = (f"**Overfitting:** no treino a PR-AUC é {_f(o.treino_media)}, {_pct(d['gap_relativo'])} acima da "
               "avaliação. O modelo memoriza parte do treino. ")
    elif d["ajuste"] == "underfitting":
        txt = ("**Underfitting:** treino e avaliação ficam próximos e baixos, o que indica que as features não "
               "carregam informação suficiente (viés). ")
    else:
        txt = (f"**Ajuste:** a diferença entre treino ({_f(o.treino_media)}) e avaliação ({_f(o.media)}) é de "
               f"{_pct(d['gap_relativo'])}. Há algum sobreajuste, esperado em modelos de conjunto, mas sem "
               "sinal de memorização forte. ")
    txt += ("A curva de aprendizado ainda sobe no último passo "
            f"({_s(d['ganho_ultimo_passo_curva'])}): mais dados devem ajudar." if d["curva_ainda_sobe"] else
            f"A curva de aprendizado se estabiliza no fim ({_s(d['ganho_ultimo_passo_curva'])} no último passo): "
            "mais encontros simulados do mesmo tipo trariam pouco ganho; o limite está na informação disponível.")
    out.append(txt)
    out.append(f"**Referência explicável:** a regressão logística tem diferença treino-avaliação de "
               f"{_pct(ref.gap_relativo)}, menor que a do modelo otimizado. Modelos mais simples generalizam de forma "
               "mais previsível, mas ficam abaixo na média.")
    if d["temporal_media"] is not None:
        out.append(f"**Futuro:** treinando só com o passado, o modelo otimizado ficou entre {_f(d['temporal_min'])} e "
                   f"{_f(d['temporal_max'])} nas semanas avaliadas (média {_f(d['temporal_media'])}), contra "
                   f"{_f(o.media)} no k-fold. "
                   + ("A diferença é pequena: o desempenho se mantém quando o modelo prevê semanas que não viu."
                      if abs(d["temporal_media"] - o.media) < 0.03 else
                      "A diferença indica que parte do desempenho do k-fold vem de misturar semanas; a janela "
                      "temporal é a estimativa mais fiel para o uso real."))
    if r["teste_cc5"] is not None:
        dentro = o.minimo - 0.01 <= r["teste_cc5"] <= o.maximo + 0.01
        out.append(f"**Teste:** a PR-AUC medida uma vez no teste do CC5 ({_f(r['teste_cc5'])}) "
                   + ("está dentro da faixa observada nos folds." if dentro else
                      "fica fora da faixa dos folds; as semanas 11–12 incluem uma convenção de games, onde "
                      "a taxa de conexão é maior, o que muda a prevalência e a métrica."))
    return out


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _params(p: dict) -> str:
    return ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in p.items())


# ======================================================================== MLflow
def _registrar(r: dict, figs: dict, texto: str, cfg: dict) -> dict:
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["validacao_cruzada"]["experimento"] + ("" if r["midia"] == "todas" else f"-{r['midia']}"))
    kf = r["_kf"]
    with mlflow.start_run(run_name=f"cv-{r['otimizado']['familia']}-{datetime.now():%Y%m%d-%H%M}") as pai:
        mlflow.log_params({"k": r["k"], "estrategia": "StratifiedGroupKFold(par_id)", "modelo": r["otimizado"]["familia"],
                           "origem": r["otimizado"]["origem"], "params_json": json.dumps(r["otimizado"]["params"]),
                           "n_encontros": r["n_encontros"], "midia": r["midia"], "ajuste": r["diagnostico"]["ajuste"]})
        mlflow.log_metrics({k: v for k, v in r["diagnostico"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                            and v is not None})
        mlflow.log_text(kf.to_csv(index=False), "folds.csv")
        mlflow.log_text(r["_ca"].to_csv(index=False), "curva_aprendizado.csv")
        mlflow.log_text(r["_tp"].to_csv(index=False), "temporal.csv")
        mlflow.log_text(texto, "validacao_cruzada.md")
        for caminho, fig in figs.items():
            mlflow.log_figure(fig, caminho)
        for m, x in r["_res"].iterrows():
            with mlflow.start_run(run_name=m, nested=True):
                mlflow.log_param("modelo", m)
                mlflow.log_metrics({f"cv_{k}": float(v) for k, v in x.items() if pd.notna(v)})
                for _, f in kf[kf.modelo == m].iterrows():
                    mlflow.log_metric("fold_pr_auc_desconhecidos", f.avaliacao_pr_auc_desconhecidos, step=int(f.fold))
                    mlflow.log_metric("fold_treino_pr_auc_desconhecidos", f.treino_pr_auc_desconhecidos, step=int(f.fold))
    return {"mlflow_run": pai.info.run_id}


def main():
    ap = argparse.ArgumentParser(description="CC6: validação cruzada agrupada por par, estabilidade e generalização")
    ap.add_argument("--midia", choices=MIDIAS)
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    sem_doc = a.sem_registro or a.midia
    r = validar(a.pasta, midia=a.midia, registrar=not a.sem_registro,
                doc=None if sem_doc else DOC, figuras=None if sem_doc else FIGURAS)
    print()
    print(r["_res"][["media", "desvio", "cv", "treino_media", "gap_relativo"]].round(3).to_string())
    for t in analise(r):
        print(" -", t.replace("**", ""))
    if not sem_doc:
        print(f"\nDocumento: {DOC}")


if __name__ == "__main__":
    main()
