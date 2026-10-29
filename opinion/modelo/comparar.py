"""
CC4 — comparação de 3+ modelos no MLflow, com análise crítica.

Mesmo protocolo do baseline (CC3, opinion.modelo.treinar): cada família de modelo é
ajustada no "treino" com uma grade pequena e fixa; a configuração e o limiar do sinal
(F0,5 entre desconhecidos) são escolhidos na "validacao"; o "teste" é medido uma vez
por família, só para relatar. A família vencedora também é escolhida na validação.

Para cada família: um run filho no MLflow com parâmetros, métricas, curvas PR e ROC,
calibração, matriz de confusão, explicação e o modelo. No run pai: a tabela comparativa,
os gráficos lado a lado e o intervalo de confiança (bootstrap no teste) das diferenças.
A análise crítica é escrita em docs/comparacao_modelos.md.

    python -m opinion.modelo.comparar
    python -m opinion.modelo.comparar --sem-registro     # sem MLflow e sem reescrever o doc
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.treinar import (_graficos, _metricas_planas, avaliar_candidato, avaliar_regra, candidatos,
                                    carregar, config, explicar, mascara_desconhecidos, medir_teste, no_limiar)
from opinion.mundo import PASTA_BOOTSTRAP, RAIZ

DOC = RAIZ / "docs" / "comparacao_modelos.md"
FIGURAS = RAIZ / "docs" / "img" / "cc4"
VERDADE = PASTA_BOOTSTRAP / "verdade_simulacao_NAO_USAR_COMO_FEATURE" / "encontros_ocultos.csv"
NOMES = {"regressao_logistica": "Regressão logística", "arvore_decisao": "Árvore de decisão",
         "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "regra": "Regra simples"}


def _nome(k: str) -> str:
    return NOMES.get(k, k)


# ======================================================================== análises
def ic_bootstrap(y, pontuacoes: dict[str, np.ndarray], referencias: list[str], n: int, semente: int) -> dict:
    """PR-AUC de cada pontuação e diferença contra as referências, com IC 95% (bootstrap pareado)."""
    y = np.asarray(y)
    rng = np.random.default_rng(semente)
    amostras = {k: np.empty(n) for k in pontuacoes}
    for b in range(n):
        i = rng.integers(0, len(y), len(y))
        for k, p in pontuacoes.items():
            amostras[k][b] = average_precision_score(y[i], p[i])
    out = {}
    for k, p in pontuacoes.items():
        r = {"pr_auc": float(average_precision_score(y, p)),
             "ic": [float(np.percentile(amostras[k], 2.5)), float(np.percentile(amostras[k], 97.5))]}
        for ref in referencias:
            d = amostras[k] - amostras[ref]
            r[f"delta_{ref}"] = float(r["pr_auc"] - average_precision_score(y, pontuacoes[ref]))
            r[f"delta_{ref}_ic"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            r[f"p_nao_melhor_que_{ref}"] = float((d <= 0).mean())
        out[k] = r
    return out


def por_contexto(te: pd.DataFrame, desc, pontuacoes: dict, limiares: dict, beta: float) -> pd.DataFrame:
    """Precisão e taxa de sinal no limiar, por local, entre desconhecidos do teste."""
    if "local_original" not in te:
        return pd.DataFrame()
    linhas = []
    d = te[desc]
    for k, p in pontuacoes.items():
        pd_ = p[desc]
        for local, idx in d.groupby("local_original").indices.items():
            m = no_limiar(d.conectou.to_numpy()[idx], pd_[idx], limiares[k], beta)
            linhas.append({"modelo": k, "local": local, "encontros": len(idx),
                           "taxa_real": float(d.conectou.to_numpy()[idx].mean()),
                           "precisao": m["precisao"], "taxa_sinal": m["taxa_sinal"]})
    return pd.DataFrame(linhas)


def teto(te: pd.DataFrame, desc, taxa_sinal: float) -> dict | None:
    """Referência que nenhum modelo alcança: a probabilidade escondida do simulador (nunca é feature)."""
    if not VERDADE.exists():
        return None
    oc = pd.read_csv(VERDADE, usecols=["encontro_id", "p_conexao_latente"])
    t = te[desc].merge(oc, on="encontro_id")
    if t.empty:
        return None
    k = max(1, int(round(taxa_sinal * len(t))))
    return {"pr_auc_desconhecidos": float(average_precision_score(t.conectou, t.p_conexao_latente)),
            "precisao_mesma_taxa_de_sinal": float(t.nlargest(k, "p_conexao_latente").conectou.mean()),
            "taxa_sinal": float(taxa_sinal)}


# ======================================================================== comparação
def comparar(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True, doc: Path | None = DOC,
             figuras: Path | None = FIGURAS) -> dict:
    cfg = cfg or config()
    c = cfg["comparacao"]
    beta, chave = float(cfg["limiar"]["beta"]), cfg["metrica"]
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    for nome, parte in (("treino", tr), ("validacao", va), ("teste", te)):
        if parte.empty:
            raise ValueError(f"split {nome} vazio")
    desc_va, desc_te = mascara_desconhecidos(va).to_numpy(), mascara_desconhecidos(te).to_numpy()
    regra = avaliar_regra(va, te, desc_va, desc_te, beta)

    familias = {}
    for fam, spec in c["modelos"].items():
        t0 = time.time()
        grade = [avaliar_candidato(n, tipo, p, tr, va, cols, desc_va, beta)
                 for n, tipo, p in candidatos(cfg, {fam: spec})]
        tempo = (time.time() - t0) / len(grade)
        melhor = max(grade, key=lambda r: r["validacao"][chave] or 0)
        t1 = time.time()
        p_te, met_te = medir_teste(melhor, te, cols, desc_te, beta)
        latencia_ms = 1000 * (time.time() - t1) / len(te)
        imp = permutation_importance(melhor["modelo"], va[cols][desc_va], va.conectou[desc_va],
                                     scoring="average_precision", n_repeats=c["importancia"]["repeticoes"],
                                     random_state=c["importancia"]["semente"], n_jobs=1)
        familias[fam] = {**melhor, "teste": met_te, "p_teste": p_te, "tempo_treino_s": tempo,
                         "latencia_ms": latencia_ms, "n_configuracoes": len(grade),
                         "grade": [{"params": g["params"], chave: g["validacao"][chave]} for g in grade],
                         "importancia": pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)}
        print(f"  {_nome(fam):<20} {len(grade)} configs  validação {chave} {melhor['validacao'][chave]:.3f}  "
              f"teste {met_te[chave]:.3f}  ({tempo:.1f} s/config)")

    vencedor = max(familias, key=lambda k: familias[k]["validacao"][chave] or 0)
    v, rv = familias[vencedor]["validacao"][chave], regra["validacao"][chave]
    if not (v > rv and not np.isclose(v, rv)):
        raise RuntimeError(f"nenhuma família superou a regra simples na validação ({v:.3f} x {rv:.3f})")

    y_d = te.conectou.to_numpy()[desc_te]
    pont = {k: f["p_teste"][desc_te] for k, f in familias.items()} | {"regra": te.compat_score.to_numpy()[desc_te]}
    ref = c["referencia"] if c["referencia"] in familias else vencedor
    ic = ic_bootstrap(y_d, pont, [ref, "regra"], c["bootstrap"]["n"], c["bootstrap"]["semente"])
    limiares = {k: f["limiar"] for k, f in familias.items()} | {"regra": regra["limiar"]}
    pont_todos = {k: f["p_teste"] for k, f in familias.items()} | {"regra": te.compat_score.to_numpy()}
    ctx = por_contexto(te, desc_te, pont_todos, limiares, beta)
    prev_treino = float(tr.conectou.mean())

    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "protocolo": {"metrica_escolha": chave, "beta_limiar": beta, "referencia": ref,
                      "por_split": meta.get("por_split"), "n_features": len(cols)},
        "vencedor": vencedor,
        "familias": {k: {"tipo": f["tipo"], "params": f["params"], "limiar": f["limiar"],
                         "validacao": f["validacao"], "teste": f["teste"], "tempo_treino_s": f["tempo_treino_s"],
                         "latencia_ms": f["latencia_ms"], "n_configuracoes": f["n_configuracoes"],
                         "grade": f["grade"], "importancia_top10": f["importancia"].head(10).round(5).to_dict()}
                     for k, f in familias.items()},
        "regra_simples": regra,
        "bootstrap_teste_desconhecidos": ic,
        "brier_constante_prevalencia": float(brier_score_loss(te.conectou, np.full(len(te), prev_treino))),
        "teto": teto(te, desc_te, familias[vencedor]["teste"]["no_limiar_desconhecidos"]["taxa_sinal"]),
        "_familias": familias, "_te": te, "_desc_te": desc_te, "_contexto": ctx,
    }
    figs = _figuras(resumo)
    if figuras is not None:
        Path(figuras).mkdir(parents=True, exist_ok=True)
        for nome, fig in figs.items():
            fig.savefig(Path(figuras) / Path(nome).name, dpi=110, bbox_inches="tight")
    texto = escrever_doc(resumo, figuras is not None)
    if doc is not None:
        Path(doc).write_text(texto, encoding="utf-8")
    if registrar:
        resumo.update(_registrar(resumo, figs, texto, cfg, pasta))
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    return resumo


# ======================================================================== gráficos
def _figuras(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

    fam, te, d = r["_familias"], r["_te"], r["_desc_te"]
    y = te.conectou.to_numpy()
    pont = {k: f["p_teste"] for k, f in fam.items()} | {"regra": te.compat_score.to_numpy()}
    lim = {k: f["limiar"] for k, f in fam.items()} | {"regra": r["regra_simples"]["limiar"]}
    figs = {}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (titulo, m) in zip(axs, (("todos", np.ones_like(d)), ("desconhecidos", d))):
        for k, p in pont.items():
            pr, rc, _ = precision_recall_curve(y[m], p[m])
            ax.plot(rc, pr, ls="--" if k == "regra" else "-",
                    label=f"{_nome(k)} ({average_precision_score(y[m], p[m]):.3f})")
        ax.axhline(y[m].mean(), c="gray", ls=":", label=f"acaso ({y[m].mean():.3f})")
        ax.set(title=f"Precisão × recall — teste, {titulo}", xlabel="recall", ylabel="precisão", ylim=(0, 0.6))
        ax.legend(fontsize=7)
    figs["comparacao/curvas_pr.png"] = fig

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (titulo, m) in zip(axs, (("todos", np.ones_like(d)), ("desconhecidos", d))):
        for k, p in pont.items():
            fpr, tpr, _ = roc_curve(y[m], p[m])
            ax.plot(fpr, tpr, ls="--" if k == "regra" else "-", label=f"{_nome(k)} ({roc_auc_score(y[m], p[m]):.3f})")
        ax.plot([0, 1], [0, 1], c="gray", ls=":")
        ax.set(title=f"ROC — teste, {titulo}", xlabel="taxa de falso positivo", ylabel="recall")
        ax.legend(fontsize=7)
    figs["comparacao/curvas_roc.png"] = fig

    ic = r["bootstrap_teste_desconhecidos"]
    ordem = sorted(ic, key=lambda k: ic[k]["pr_auc"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, k in enumerate(ordem):
        lo, hi = ic[k]["ic"]
        ax.errorbar(ic[k]["pr_auc"], i, xerr=[[ic[k]["pr_auc"] - lo], [hi - ic[k]["pr_auc"]]], fmt="o", capsize=4,
                    color="tab:orange" if k == "regra" else "tab:blue")
    ax.axvline(y[d].mean(), c="gray", ls=":", label="acaso")
    if r.get("teto"):
        ax.axvline(r["teto"]["pr_auc_desconhecidos"], c="tab:green", ls="--", label="teto do simulador")
    ax.set(yticks=range(len(ordem)), yticklabels=[_nome(k) for k in ordem], xlabel="PR-AUC (IC 95%)",
           title="PR-AUC entre desconhecidos — teste")
    ax.legend(fontsize=8)
    figs["comparacao/pr_auc_ic.png"] = fig

    n = len(pont)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n + 1.2, 3.4), sharey=True)
    for col, (ax, k) in enumerate(zip(axs, pont)):
        cm = confusion_matrix(y[d], pont[k][d] >= lim[k], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, _n(v), ha="center", va="center", fontsize=10,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=f"{_nome(k)}\nlimiar {_f(lim[k])}", xticks=[0, 1], yticks=[0, 1],
               xticklabels=["sem sinal", "sinal"], yticklabels=["não conectou", "conectou"])
        if col:
            ax.tick_params(labelleft=False)
    fig.tight_layout()
    fig.suptitle("Matriz de confusão — desconhecidos, teste", y=1.08)
    figs["comparacao/matrizes_confusao.png"] = fig

    imp = pd.DataFrame({_nome(k): f["importancia"] for k, f in fam.items()})
    top = imp.max(axis=1).sort_values(ascending=False).index[:12]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    imp.loc[top[::-1]].plot.barh(ax=ax, width=0.8)
    ax.set(title="Importância por permutação (queda de PR-AUC, desconhecidos, validação)", xlabel="queda de PR-AUC")
    ax.legend(fontsize=7)
    figs["comparacao/importancia.png"] = fig
    return figs


# ======================================================================== documento
def _f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}".replace(".", ",")


def _s(x) -> str:
    return f"{x:+.3f}".replace(".", ",")


def _lista(nomes: list[str]) -> str:
    return nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " e " + nomes[-1]


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def escrever_doc(r: dict, com_figuras: bool) -> str:
    fam, regra, ic, venc = r["_familias"], r["regra_simples"], r["bootstrap_teste_desconhecidos"], r["vencedor"]
    ref = r["protocolo"]["referencia"]
    ps = r["protocolo"]["por_split"] or {}
    ordem = sorted(fam, key=lambda k: -(fam[k]["validacao"]["pr_auc_desconhecidos"] or 0))
    brier0 = r["brier_constante_prevalencia"]
    L = [
        "# Comparação de modelos — OpiniOn (CC4)", "",
        f"Gerado por `python -m opinion.modelo.comparar` em {datetime.now():%d/%m/%Y %H:%M}. "
        "Números do teste medidos **uma vez** por modelo; nada foi escolhido com eles.", "",
        "> Usuários, obras e notas são reais (Amazon Reviews 2023); encontros e o label `conectou` são simulados "
        "([`fontes_dados.md`](fontes_dados.md)). As métricas medem quão bem cada modelo recupera o processo da "
        "simulação, não o comportamento de usuários reais da pulseira.", "",
        "## 1. Protocolo", "",
        "- **Mesmos dados e split do CC3:** treino semanas 1–8"
        + (f" ({_n(ps.get('treino', {}).get('encontros', 0))} encontros)" if ps else "")
        + ", validação 9–10, teste 11–12; purga na fronteira treino/validação.",
        f"- **Cada família** tem uma grade pequena e fixa (`config/modelo.yaml → comparacao`). A configuração e o "
        f"limiar do sinal (F{_f(r['protocolo']['beta_limiar'], 1)} entre desconhecidos) são escolhidos na validação. "
        "Otimização de hiperparâmetros fica para o CC5 (Optuna) e validação cruzada para o CC6.",
        f"- **Escolha do melhor modelo:** {r['protocolo']['metrica_escolha']} na validação. "
        "A regra simples (`compat_score`) recebe o mesmo protocolo, com limiar próprio.",
        f"- **Incerteza:** intervalo de confiança de 95% por bootstrap pareado ({len(ic)} pontuações, "
        "mesmas reamostras para todas) no teste, entre desconhecidos.", "",
        "## 2. Resultado", "",
        "| Modelo | PR-AUC desc. (valid.) | PR-AUC desc. (teste, IC 95%) | PR-AUC geral (teste) | ROC-AUC desc. (teste) | Brier (teste) |",
        "|---|---|---|---|---|---|",
    ]
    for k in ordem:
        f, i = fam[k], ic[k]
        marca = " **(escolhido)**" if k == venc else ""
        L.append(f"| {_nome(k)}{marca} | {_f(f['validacao']['pr_auc_desconhecidos'])} | {_f(i['pr_auc'])} "
                 f"({_f(i['ic'][0])}–{_f(i['ic'][1])}) | {_f(f['teste']['pr_auc'])} | "
                 f"{_f(f['teste']['roc_auc_desconhecidos'])} | {_f(f['teste']['brier'])} |")
    L.append(f"| Regra simples | {_f(regra['validacao']['pr_auc_desconhecidos'])} | {_f(ic['regra']['pr_auc'])} "
             f"({_f(ic['regra']['ic'][0])}–{_f(ic['regra']['ic'][1])}) | {_f(regra['teste']['pr_auc'])} | "
             f"{_f(regra['teste']['roc_auc_desconhecidos'])} | não se aplica |")
    prev_d = fam[venc]["teste"]["prevalencia_desconhecidos"]
    L += [f"| *acaso* | | *{_f(prev_d)}* | *{_f(fam[venc]['teste']['prevalencia'])}* | *0,500* | "
          f"*{_f(brier0)}* (constante) |", ""]
    if com_figuras:
        L += ["![PR-AUC com IC](img/cc4/pr_auc_ic.png)", "", "![Curvas PR](img/cc4/curvas_pr.png)", "",
              "![Curvas ROC](img/cc4/curvas_roc.png)", ""]

    L += ["## 3. No ponto de operação (desconhecidos, teste)", "",
          "O que a pulseira faria: sinaliza quando a pontuação passa do limiar escolhido na validação.", "",
          "| Modelo | Limiar | Precisão | Recall | F0,5 | Sinal em | VP | FP | FN | VN |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ordem + ["regra"]:
        nl = (fam[k]["teste"] if k != "regra" else regra["teste"])["no_limiar_desconhecidos"]
        lim = fam[k]["limiar"] if k != "regra" else regra["limiar"]
        L.append(f"| {_nome(k)} | {_f(lim)} | {_pct(nl['precisao'])} | {_pct(nl['recall'])} | {_f(nl['f05'])} | "
                 f"{_pct(nl['taxa_sinal'])} | {_n(nl['vp'])} | {_n(nl['fp'])} | {_n(nl['fn'])} | {_n(nl['vn'])} |")
    L.append("")
    if com_figuras:
        L += ["![Matrizes de confusão](img/cc4/matrizes_confusao.png)", ""]

    L += ["## 4. Diferenças e incerteza (PR-AUC entre desconhecidos, teste)", "",
          f"| Modelo | vs {_nome(ref)} (IC 95%) | vs regra simples (IC 95%) |", "|---|---|---|"]
    for k in ordem:
        i = ic[k]
        a = "—" if k == ref else f"{_s(i[f'delta_{ref}'])} ({_s(i[f'delta_{ref}_ic'][0])} a {_s(i[f'delta_{ref}_ic'][1])})"
        b = f"{_s(i['delta_regra'])} ({_s(i['delta_regra_ic'][0])} a {_s(i['delta_regra_ic'][1])})"
        L.append(f"| {_nome(k)} | {a} | {b} |")
    L.append("")

    ctx = r["_contexto"]
    if not ctx.empty:
        L += ["## 5. Precisão do sinal por contexto (desconhecidos, teste)", "",
              "| Local | Encontros | Taxa real | " + " | ".join(_nome(k) for k in ordem + ["regra"]) + " |",
              "|---|---|---|" + "---|" * (len(ordem) + 1)]
        for local, g in ctx.groupby("local"):
            g = g.set_index("modelo")
            celulas = [f"{_pct(g.loc[k, 'precisao'])} ({_pct(g.loc[k, 'taxa_sinal'])})" for k in ordem + ["regra"]]
            L.append(f"| {local} | {_n(g.encontros.iloc[0])} | {_pct(g.taxa_real.iloc[0])} | " + " | ".join(celulas) + " |")
        L += ["", "Cada célula: precisão (fração de encontros com sinal).", ""]

    L += ["## 6. O que cada modelo usa (importância por permutação, validação)", "",
          "Queda de PR-AUC entre desconhecidos quando a feature é embaralhada. Top 5 de cada modelo:", "",
          "| Modelo | 1º | 2º | 3º | 4º | 5º |", "|---|---|---|---|---|---|"]
    for k in ordem:
        top = fam[k]["importancia"].head(5)
        L.append(f"| {_nome(k)} | " + " | ".join(f"`{n}` ({v:+.3f})".replace(".", ",") for n, v in top.items()) + " |")
    L.append("")
    if com_figuras:
        L += ["![Importância](img/cc4/importancia.png)", ""]

    L += ["## 7. Custo", "", "| Modelo | Configurações testadas | Treino por configuração | Pontuação por encontro |",
          "|---|---|---|---|"]
    for k in ordem:
        f = fam[k]
        L.append(f"| {_nome(k)} | {f['n_configuracoes']} | {_f(f['tempo_treino_s'], 1)} s | "
                 f"{_f(f['latencia_ms'], 3)} ms |")
    L.append("")

    L += ["## 8. Análise crítica", ""] + [f"- {t}" for t in analise(r)] + [""]
    L += ["## 9. Limitações", "",
          "- **Label simulado.** O ranking dos modelos vale para o processo da simulação. Com outras premissas "
          "(ver os cenários em [`validacao_simulacao.md`](validacao_simulacao.md)) a ordem entre modelos próximos "
          "pode mudar.",
          "- **Um único split temporal.** O IC do bootstrap cobre a variação de amostragem do teste, não a de "
          "treinar em outras semanas. Estabilidade entre folds é o CC6.",
          "- **Grades fixas e pequenas.** Um modelo pode ter perdido só por falta de ajuste; o CC5 otimiza o "
          "vencedor com Optuna.",
          "- **Teto do simulador.** Só existe porque os dados são simulados; serve para dimensionar a margem, "
          "não como meta realista.", ""]
    return "\n".join(L)


def analise(r: dict) -> list[str]:
    """Conclusões escritas a partir dos números (mudam se os números mudarem)."""
    fam, regra, ic, venc = r["_familias"], r["regra_simples"], r["bootstrap_teste_desconhecidos"], r["vencedor"]
    ref, chave = r["protocolo"]["referencia"], "pr_auc_desconhecidos"
    out = []
    ordem_va = sorted(fam, key=lambda k: -fam[k]["validacao"][chave])
    ordem_te = sorted(fam, key=lambda k: -fam[k]["teste"][chave])
    out.append(f"**Escolhido na validação: {_nome(venc)}** ({_f(fam[venc]['validacao'][chave])}). "
               + ("A ordem dos modelos no teste é a mesma da validação."
                  if ordem_va == ordem_te else
                  f"No teste a ordem muda (1º no teste: {_nome(ordem_te[0])}): as diferenças entre eles são "
                  "pequenas o bastante para mudar de posição com outra amostra."))
    if venc != ref:
        lo, hi = ic[venc][f"delta_{ref}_ic"]
        sig = lo > 0
        out.append(f"**Contra o baseline ({_nome(ref)}):** {_s(ic[venc][f'delta_{ref}'])} de PR-AUC "
                   f"(IC 95% {_s(lo)} a {_s(hi)}). "
                   + ("O ganho é estatisticamente consistente no teste."
                      if sig else "O intervalo inclui zero: **não dá para afirmar que o modelo mais complexo é "
                                  "melhor que o baseline** com estes dados."))
    else:
        out.append("**O baseline do CC3 continua o melhor na validação**: os modelos mais complexos não "
                   "justificaram a perda de explicabilidade.")
    lo, hi = ic[venc]["delta_regra_ic"]
    out.append(f"**Contra a regra simples:** {_s(ic[venc]['delta_regra'])} de PR-AUC (IC 95% {_s(lo)} a "
               f"{_s(hi)}) — " + ("o modelo supera a regra de forma consistente." if lo > 0 else
                                   "a vantagem sobre a regra não é consistente no teste."))
    nl, nr = fam[venc]["teste"]["no_limiar_desconhecidos"], regra["teste"]["no_limiar_desconhecidos"]
    out.append(f"**No ponto de operação**, {_nome(venc)} acerta {_pct(nl['precisao'])} dos sinais "
               f"(recall {_pct(nl['recall'])}, sinal em {_pct(nl['taxa_sinal'])} dos encontros); a regra, "
               f"{_pct(nr['precisao'])} (recall {_pct(nr['recall'])}, sinal em {_pct(nr['taxa_sinal'])}). "
               + ("Ganhar em PR-AUC não se traduziu em precisão bem maior no limiar escolhido: a diferença está em "
                  "avisar mais vezes com a mesma precisão."
                  if nl["precisao"] - nr["precisao"] < 0.02 else
                  "A vantagem aparece também na precisão do sinal, que é o que o usuário sente."))
    b0 = r["brier_constante_prevalencia"]
    ruins = [k for k in fam if fam[k]["teste"]["brier"] > b0]
    bons = [k for k in fam if fam[k]["teste"]["brier"] <= b0]
    out.append(f"**Calibração:** prever sempre a prevalência dá Brier {_f(b0)}. "
               + (f"{_lista([_nome(k) for k in ruins])} {'fica' if len(ruins) == 1 else 'ficam'} pior que isso "
                  "(probabilidades infladas, efeito de `class_weight=balanced`); " if ruins else "")
               + (f"{_lista([_nome(k) for k in bons])} {'fica' if len(bons) == 1 else 'ficam'} melhor. "
                  if bons else "")
               + "A ordenação e o limiar não dependem disso, mas a probabilidade só pode ser mostrada como chance "
                 "real (dashboard, faixas de sinal) depois de calibrada.")
    tops = {k: set(fam[k]["importancia"].head(3).index) for k in fam}
    comum = set.intersection(*tops.values())
    out.append("**O que os modelos aprendem:** "
               + (f"todos têm {', '.join(f'`{c}`' for c in sorted(comum))} entre as 3 features mais importantes"
                  if comum else "as 3 features mais importantes variam entre os modelos")
               + ". Lembrete: a importância reflete as premissas do simulador, não uma descoberta sobre usuários.")
    if r.get("teto"):
        t = r["teto"]
        out.append(f"**Margem:** o teto do simulador (a probabilidade escondida) chega a PR-AUC "
                   f"{_f(t['pr_auc_desconhecidos'])} e precisão {_pct(t['precisao_mesma_taxa_de_sinal'])} com a "
                   f"mesma taxa de sinal; {_nome(venc)} está em {_f(ic[venc]['pr_auc'])} e "
                   f"{_pct(nl['precisao'])}. A parte que falta vem do que nenhum dado observa (a abertura social "
                   "de cada pessoa) e do sorteio do resultado.")
    lat = fam[venc]["latencia_ms"]
    out.append(f"**Custo:** {_nome(venc)} pontua um encontro em {_f(lat, 3)} ms, sem impedimento para o sinal em "
               "tempo real (CC7).")
    simples = venc in ("regressao_logistica", "arvore_decisao")
    out.append("**Recomendação para o CC5:** " + (
        f"otimizar {_nome(venc)} com Optuna, mantendo o baseline como referência explicável."
        if not simples else
        f"{_nome(venc)} venceu sendo explicável; o CC5 pode otimizá-lo ou testar se um modelo de conjunto, "
        "ajustado, passa a superá-lo com folga."))
    return out


# ======================================================================== MLflow
def _registrar(r: dict, figs: dict, texto: str, cfg: dict, pasta) -> dict:
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    import warnings
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    c = cfg["comparacao"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(c["experimento"])
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    exemplo = dm[dm.split == "treino"][cols].head(5)
    fam, te, d, regra, ic = r["_familias"], r["_te"], r["_desc_te"], r["regra_simples"], r["bootstrap_teste_desconhecidos"]
    ref = r["protocolo"]["referencia"]
    info_venc = None

    with mlflow.start_run(run_name=f"comparacao-{datetime.now():%Y%m%d-%H%M}") as pai:
        mlflow.log_params({"vencedor": r["vencedor"], "metrica_escolha": r["protocolo"]["metrica_escolha"],
                           "referencia": ref, "beta_limiar": r["protocolo"]["beta_limiar"],
                           "n_modelos": len(fam), "bootstrap_n": c["bootstrap"]["n"]})
        for k, i in ic.items():
            mlflow.log_metric(f"teste_pr_auc_desconhecidos_{k}", i["pr_auc"])
        mlflow.log_metrics(_metricas_planas("regra_validacao", regra["validacao"]))
        mlflow.log_metrics(_metricas_planas("regra_teste", regra["teste"]))
        tabela = pd.DataFrame([{"modelo": k, "params": str(f["params"]), "limiar": f["limiar"],
                                "validacao_pr_auc_desconhecidos": f["validacao"]["pr_auc_desconhecidos"],
                                "teste_pr_auc_desconhecidos": f["teste"]["pr_auc_desconhecidos"],
                                "teste_ic_inf": ic[k]["ic"][0], "teste_ic_sup": ic[k]["ic"][1],
                                "teste_roc_auc_desconhecidos": f["teste"]["roc_auc_desconhecidos"],
                                "teste_brier": f["teste"]["brier"],
                                **{f"teste_{m}": f["teste"]["no_limiar_desconhecidos"][m]
                                   for m in ("precisao", "recall", "f05", "taxa_sinal", "vp", "fp", "fn", "vn")}}
                               for k, f in fam.items()])
        mlflow.log_text(tabela.to_csv(index=False), "comparacao.csv")
        mlflow.log_text(texto, "comparacao_modelos.md")
        mlflow.log_dict({k: v for k, v in r.items() if not k.startswith("_")}, "resumo.json")
        if not r["_contexto"].empty:
            mlflow.log_text(r["_contexto"].to_csv(index=False), "por_contexto.csv")
        for caminho, fig in figs.items():
            mlflow.log_figure(fig, caminho)

        for k, f in fam.items():
            with mlflow.start_run(run_name=k, nested=True) as filho:
                mlflow.log_params({"modelo": k, "tipo": f["tipo"], "limiar": f["limiar"],
                                   "limiar_regra": regra["limiar"], "n_configuracoes": f["n_configuracoes"],
                                   **{f"hp_{p}": v for p, v in f["params"].items()}})
                mlflow.log_metrics(_metricas_planas("validacao", f["validacao"]))
                mlflow.log_metrics(_metricas_planas("teste", f["teste"]))
                mlflow.log_metrics({"tempo_treino_s": f["tempo_treino_s"], "latencia_ms": f["latencia_ms"],
                                    "teste_ic_inf": ic[k]["ic"][0], "teste_ic_sup": ic[k]["ic"][1],
                                    "delta_regra": ic[k]["delta_regra"]}
                                   | ({f"delta_{ref}": ic[k][f"delta_{ref}"]} if k != ref else {}))
                mlflow.log_dict({"grade": f["grade"]}, "grade.json")
                mlflow.log_text(f["importancia"].to_frame("queda_pr_auc").to_csv(), "importancia.csv")
                exp = explicar(f["modelo"], cols, f["tipo"])
                if exp:
                    mlflow.log_text(exp, "explicacao.txt")
                preds = {"y": te.conectou.to_numpy(), "p": f["p_teste"], "regra": te.compat_score.to_numpy(), "desc": d}
                for caminho, fig in _graficos(preds, f["limiar"], regra["limiar"]).items():
                    mlflow.log_figure(fig, caminho)
                    plt.close(fig)
                info = mlflow.sklearn.log_model(
                    sk_model=f["modelo"], name="modelo", input_example=exemplo,
                    signature=infer_signature(exemplo, f["modelo"].predict_proba(exemplo)[:, 1]),
                    serialization_format="cloudpickle")
                if k == r["vencedor"]:
                    info_venc = (info, filho.info.run_id)

    mv = mlflow.register_model(info_venc[0].model_uri, c["modelo_registrado"])
    MlflowClient().set_registered_model_alias(c["modelo_registrado"], c["alias"], mv.version)
    return {"mlflow_run_pai": pai.info.run_id, "mlflow_run_vencedor": info_venc[1],
            "mlflow_modelo": f"{c['modelo_registrado']}@{c['alias']}", "mlflow_versao": str(mv.version)}


# ======================================================================== CLI
def _imprimir(r: dict) -> None:
    fam, ic, regra = r["_familias"], r["bootstrap_teste_desconhecidos"], r["regra_simples"]
    print(f"\n{'':22} {'valid.':>7} {'teste':>7} {'IC 95%':>15} {'precisão':>9} {'recall':>7} {'sinal':>7} {'Brier':>6}")
    for k in sorted(fam, key=lambda k: -fam[k]["validacao"]["pr_auc_desconhecidos"]):
        f, nl = fam[k], fam[k]["teste"]["no_limiar_desconhecidos"]
        print(f"{_nome(k) + (' *' if k == r['vencedor'] else ''):22} {f['validacao']['pr_auc_desconhecidos']:7.3f} "
              f"{ic[k]['pr_auc']:7.3f} {ic[k]['ic'][0]:7.3f}–{ic[k]['ic'][1]:.3f} {nl['precisao']:9.3f} "
              f"{nl['recall']:7.3f} {nl['taxa_sinal']:7.1%} {f['teste']['brier']:6.3f}")
    nr = regra["teste"]["no_limiar_desconhecidos"]
    print(f"{'Regra simples':22} {regra['validacao']['pr_auc_desconhecidos']:7.3f} {ic['regra']['pr_auc']:7.3f} "
          f"{ic['regra']['ic'][0]:7.3f}–{ic['regra']['ic'][1]:.3f} {nr['precisao']:9.3f} {nr['recall']:7.3f} "
          f"{nr['taxa_sinal']:7.1%}")
    print("(PR-AUC entre desconhecidos; * = escolhido na validação; precisão/recall/sinal no limiar, teste)\n")
    for t in analise(r):
        print(" -", t.replace("**", ""))
    if r.get("mlflow_run_pai"):
        print(f"\nMLflow: experimento opinion-cc4, run {r['mlflow_run_pai']}  ·  melhor: {r['mlflow_modelo']} "
              f"v{r['mlflow_versao']}")


def main():
    ap = argparse.ArgumentParser(description="CC4: compara 3+ modelos e registra no MLflow")
    ap.add_argument("--sem-registro", action="store_true", help="não grava no MLflow nem reescreve o doc")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    print("Comparando modelos (treino → escolha na validação → teste uma vez):")
    if a.sem_registro:
        r = comparar(a.pasta, registrar=False, doc=None, figuras=None)
    else:
        r = comparar(a.pasta)
        print(f"\nAnálise crítica em {DOC}")
    _imprimir(r)


if __name__ == "__main__":
    main()
