"""
CC6 — validação cruzada estratificada e agrupada, estabilidade, over/underfitting e generalização.

Usa só treino + validação (o teste fica fora). Folds por `StratifiedGroupKFold`: estratificados pelo
rótulo `conectou` (positivos raros) e agrupados por `par_id` (os encontros de um par nunca se dividem
entre treino e validação de um fold). Os hiperparâmetros das duas famílias vêm do CC5
(data/processed/otimizacao.json); a regra simples (`compat_score`) entra como referência, sem ajuste.
A padronização da logística fica dentro do pipeline, ajustada só no treino de cada fold.

Saídas: PR-AUC/ROC-AUC de treino e validação por fold e semente; média, desvio, coeficiente de variação
e gap treino−validação; curva de aprendizado e de complexidade; comparação com o teste do CC5;
MLflow `opinion-cc6-lastfm` e docs/validacao_cruzada.md.

    python -m opinion.modelo.validar
    python -m opinion.modelo.validar --sem-registro      # sem MLflow e sem reescrever o doc
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.otimizar import _f, _n, _nome, _params, _pct, _s, _seg
from opinion.modelo.treinar import _pipeline, carregar, config, metricas
from opinion.mundo import RAIZ

DOC = RAIZ / "docs" / "validacao_cruzada.md"
FIGURAS = RAIZ / "docs" / "img" / "cc6"
REGRA = "regra"


# ======================================================================== folds
def folds(dm_cv: pd.DataFrame, k: int, semente: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k folds (índices posicionais de treino e validação), estratificados por `conectou` e agrupados por
    `par_id`. Falha alto se algum fold (treino ou validação) ficar sem positivo."""
    y, g = dm_cv["conectou"].to_numpy(), dm_cv["par_id"].to_numpy()
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente)
    out = []
    for i, (tr, va) in enumerate(sgk.split(np.zeros(len(y)), y, g)):
        if y[va].sum() == 0 or y[tr].sum() == 0:
            raise ValueError(f"fold {i} sem positivo (k={k}, semente={semente}, positivos={int(y.sum())}): "
                             "reduza k ou junte mais dados")
        if set(g[tr]) & set(g[va]):
            raise ValueError(f"fold {i}: par_id em treino e validação")   # não deve ocorrer
        out.append((tr, va))
    return out


def _par(y, p) -> tuple[float, float]:
    """(PR-AUC, ROC-AUC) pelo mesmo `treinar.metricas` do restante do projeto."""
    m = metricas(y, p, np.ones(len(y), dtype=bool), probabilidade=False)
    return m["pr_auc"], m["roc_auc"]


def _modelos(pasta) -> dict[str, tuple[str, dict]]:
    """Famílias otimizadas (CC5): nome -> (tipo, parâmetros)."""
    arq = Path(pasta) / "otimizacao.json"
    if not arq.exists():
        raise FileNotFoundError(f"{arq} não existe: rode `python -m opinion.modelo.otimizar` (CC5) antes")
    fam = json.loads(arq.read_text(encoding="utf-8"))["familias"]
    return {k: (v["tipo"], v["params"]) for k, v in fam.items()}


def _bloco(pt, rt, pv, rv) -> dict:
    return {"pr_auc_treino": pt, "roc_auc_treino": rt, "pr_auc_val": pv, "roc_auc_val": rv}


def _ajustar(tipo, params, tr, va, cols) -> dict:
    m = _pipeline(tipo, params)
    m.fit(tr[cols], tr.conectou)
    return _bloco(*_par(tr.conectou, m.predict_proba(tr[cols])[:, 1]), *_par(va.conectou, m.predict_proba(va[cols])[:, 1]))


def _regra(tr, va) -> dict:
    return _bloco(*_par(tr.conectou, tr.compat_score), *_par(va.conectou, va.compat_score))


def _desvio(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def _resumir(fs: list[dict]) -> dict:
    d = pd.DataFrame(fs)
    por_semente = d.groupby("semente")
    media = por_semente.pr_auc_val.mean()
    desvio = float(np.mean([_desvio(g.pr_auc_val) for _, g in por_semente]))   # variação entre folds, por semente
    m = float(d.pr_auc_val.mean())
    return {
        "pr_auc_media": m, "pr_auc_desvio": desvio, "pr_auc_cv": desvio / m if m else None,
        "pr_auc_treino_media": float(d.pr_auc_treino.mean()),
        "gap_pr_auc": float((d.pr_auc_treino - d.pr_auc_val).mean()),
        "pr_auc_min": float(d.pr_auc_val.min()), "pr_auc_max": float(d.pr_auc_val.max()),
        "roc_auc_media": float(d.roc_auc_val.mean()),
        "roc_auc_desvio": float(np.mean([_desvio(g.roc_auc_val) for _, g in por_semente])),
        "roc_auc_treino_media": float(d.roc_auc_treino.mean()),
        "gap_roc_auc": float((d.roc_auc_treino - d.roc_auc_val).mean()),
        "prevalencia_media": float(d.prevalencia.mean()),
        "por_semente": {str(s): float(v) for s, v in media.items()},
        "desvio_entre_sementes": _desvio(media.to_numpy()),
    }


# ======================================================================== CV, curvas, referência
def _cv(dm_cv, cols, modelos, k, sementes) -> tuple[dict, dict]:
    res = {n: [] for n in [*modelos, REGRA]}
    tempos = {n: 0.0 for n in res}
    for s in sementes:
        for i, (itr, iva) in enumerate(folds(dm_cv, k, s)):
            tr, va = dm_cv.iloc[itr], dm_cv.iloc[iva]
            base = {"semente": int(s), "fold": i, "n_treino": len(tr), "n_val": len(va),
                    "positivos_val": int(va.conectou.sum()), "prevalencia": float(va.conectou.mean())}
            for nome, (tipo, params) in modelos.items():
                t0 = time.time()
                res[nome].append({**base, **_ajustar(tipo, params, tr, va, cols)})
                tempos[nome] += time.time() - t0
            res[REGRA].append({**base, **_regra(tr, va)})
    return res, tempos


def _subamostra(tr: pd.DataFrame, frac: float, semente: int) -> pd.DataFrame:
    """Fração do treino do fold, sorteada por par (não por linha), para não dividir pares."""
    if frac >= 1.0:
        return tr
    pares = tr.par_id.unique()
    sel = np.random.default_rng(semente).permutation(pares)[:max(1, int(round(frac * len(pares))))]
    sub = tr[tr.par_id.isin(sel)]
    if sub.conectou.sum() < 2 or sub.conectou.nunique() < 2:
        raise ValueError(f"fração {frac} do treino sem positivos suficientes")
    return sub


def _agregar(linhas: list[dict], chave: str) -> dict:
    d = pd.DataFrame(linhas)
    return {chave: linhas[0][chave], "n_treino": float(d.n_treino.mean()),
            "treino_media": float(d.pr_auc_treino.mean()), "treino_desvio": _desvio(d.pr_auc_treino),
            "val_media": float(d.pr_auc_val.mean()), "val_desvio": _desvio(d.pr_auc_val),
            "roc_treino_media": float(d.roc_auc_treino.mean()), "roc_val_media": float(d.roc_auc_val.mean())}


def _curva_aprendizado(dm_cv, cols, modelos, k, sementes, fracoes) -> dict:
    out = {n: [] for n in modelos}
    for nome, (tipo, params) in modelos.items():
        for fr in fracoes:
            linhas = []
            for s in sementes:
                for i, (itr, iva) in enumerate(folds(dm_cv, k, s)):
                    tr = _subamostra(dm_cv.iloc[itr], fr, 1000 * s + i)
                    linhas.append({"fracao": fr, "n_treino": len(tr), **_ajustar(tipo, params, tr, dm_cv.iloc[iva], cols)})
            out[nome].append(_agregar(linhas, "fracao"))
    return out


def _curva_complexidade(dm_cv, cols, modelos, k, sementes, spec) -> dict:
    out = {}
    for nome, (tipo, params) in modelos.items():
        par, valores = spec[nome]["parametro"], spec[nome]["valores"]
        pontos = []
        for v in valores:
            linhas = []
            for s in sementes:
                for itr, iva in folds(dm_cv, k, s):
                    linhas.append({"valor": v, "n_treino": len(itr),
                                   **_ajustar(tipo, {**params, par: v}, dm_cv.iloc[itr], dm_cv.iloc[iva], cols)})
            pontos.append(_agregar(linhas, "valor"))
        out[nome] = {"parametro": par, "pontos": pontos, "valor_otimizado": params.get(par)}
    return out


def _referencia_cc5(dm, cols, modelos) -> dict:
    """Teste do CC5 reproduzido: reajusta no treino (determinístico, mesmos parâmetros) e mede validação e
    teste. Só para comparar com a CV; nada é escolhido com isso."""
    tr, va, te = (dm[dm.split == s] for s in ("treino", "validacao", "teste"))
    out = {"prevalencia_teste": float(te.conectou.mean()), "prevalencia_validacao": float(va.conectou.mean()),
           "n_teste": len(te), "modelos": {}}
    for nome, (tipo, params) in modelos.items():
        m = _pipeline(tipo, params).fit(tr[cols], tr.conectou)
        out["modelos"][nome] = {"pr_auc_validacao": _par(va.conectou, m.predict_proba(va[cols])[:, 1])[0],
                                "pr_auc_teste": _par(te.conectou, m.predict_proba(te[cols])[:, 1])[0]}
    out["modelos"][REGRA] = {"pr_auc_validacao": _par(va.conectou, va.compat_score)[0],
                             "pr_auc_teste": _par(te.conectou, te.compat_score)[0]}
    return out


def _pareado(res: dict, a: str, b: str) -> dict:
    da = pd.DataFrame(res[a]).pr_auc_val.to_numpy() - pd.DataFrame(res[b]).pr_auc_val.to_numpy()
    return {"a": a, "b": b, "media": float(da.mean()), "desvio": _desvio(da), "vitorias_a": int((da > 0).sum()),
            "n": int(len(da)), "minimo": float(da.min()), "maximo": float(da.max())}


# ======================================================================== validação
def validar(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True, doc: Path | None = DOC,
            figuras: Path | None = FIGURAS) -> dict:
    cfg = cfg or config()
    c = cfg["validacao_cruzada"]
    k, sementes, sementes_curvas = int(c["k"]), [int(s) for s in c["sementes"]], [int(s) for s in c["sementes_curvas"]]
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    cv = dm[dm.split.isin(["treino", "validacao"])].reset_index(drop=True)   # o teste nunca entra
    if cv.empty or (dm.split == "teste").sum() == 0:
        raise ValueError("splits treino/validacao/teste precisam existir")
    modelos = _modelos(pasta)
    T0 = time.time()
    tempos = {}

    t0 = time.time()
    res, t_mod = _cv(cv, cols, modelos, k, sementes)
    tempos["cv_s"], tempos["cv_por_modelo_s"] = time.time() - t0, t_mod
    print(f"  CV {k} folds x {len(sementes)} sementes: {_seg(tempos['cv_s'])}", flush=True)

    t0 = time.time()
    fracoes = [float(x) for x in c["fracoes_aprendizado"]]
    aprend = _curva_aprendizado(cv, cols, modelos, k, sementes_curvas, fracoes)
    tempos["curva_aprendizado_s"] = time.time() - t0
    print(f"  curva de aprendizado: {_seg(tempos['curva_aprendizado_s'])}", flush=True)

    t0 = time.time()
    compl = _curva_complexidade(cv, cols, modelos, k, sementes_curvas, c["complexidade"])
    tempos["curva_complexidade_s"] = time.time() - t0
    print(f"  curva de complexidade: {_seg(tempos['curva_complexidade_s'])}", flush=True)

    t0 = time.time()
    ref = _referencia_cc5(dm, cols, modelos)
    tempos["referencia_cc5_s"] = time.time() - t0
    tempos["total_s"] = time.time() - T0

    nomes = list(modelos)
    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "protocolo": {"k": k, "sementes": sementes, "sementes_curvas": sementes_curvas,
                      "fracoes_aprendizado": fracoes, "splits_usados": ["treino", "validacao"], "n_cv": len(cv),
                      "n_pares_cv": int(cv.par_id.nunique()), "positivos_cv": int(cv.conectou.sum()),
                      "prevalencia_cv": float(cv.conectou.mean()), "n_features": len(cols),
                      "publico": cfg.get("publico", "desconhecidos"), "experimento": c["experimento"]},
        "modelos": {n: {"folds": res[n], "resumo": _resumir(res[n])} for n in res},
        "parametros": {n: {"tipo": t, "params": p} for n, (t, p) in modelos.items()},
        "pareado": _pareado(res, nomes[1], nomes[0]) if len(nomes) > 1 else None,
        "curva_aprendizado": aprend, "curva_complexidade": compl, "referencia_cc5": ref,
        "vencedor_cc5": json.loads((Path(pasta) / "otimizacao.json").read_text(encoding="utf-8")).get("vencedor"),
        "tempos": tempos, "_pares_cv": cv.par_id.unique().tolist(),
    }
    figs = _figuras(r)
    if figuras is not None:
        Path(figuras).mkdir(parents=True, exist_ok=True)
        for nome, fig in figs.items():
            fig.savefig(Path(figuras) / nome, dpi=110, bbox_inches="tight")
    texto = escrever_doc(r, figuras is not None)
    if doc is not None:
        Path(doc).write_text(texto, encoding="utf-8")
    if registrar:
        r.update(_registrar(r, figs, texto, cfg))
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    return r


# ======================================================================== gráficos
def _val_curto(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _figuras(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mods = r["modelos"]
    ordem = list(mods)
    prev = r["protocolo"]["prevalencia_cv"]
    figs = {}

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    rng = np.random.default_rng(0)
    for i, n in enumerate(ordem):
        v = [f["pr_auc_val"] for f in mods[n]["folds"]]
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, alpha=0.6, c="tab:blue", zorder=3)
    ax.axhline(prev, c="gray", ls=":", label=f"acaso (prevalência {_f(prev, 4)})")
    ax.set_xticks(range(len(ordem)), [_nome(n) for n in ordem])
    ax.set(ylabel="PR-AUC na validação do fold", title="Estabilidade: PR-AUC por fold e semente")
    ax.legend(fontsize=8)
    figs["estabilidade.png"] = fig

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x, larg = np.arange(len(ordem)), 0.35
    ax.bar(x - larg / 2, [mods[n]["resumo"]["pr_auc_treino_media"] for n in ordem], larg, label="treino")
    ax.bar(x + larg / 2, [mods[n]["resumo"]["pr_auc_media"] for n in ordem], larg,
           yerr=[mods[n]["resumo"]["pr_auc_desvio"] for n in ordem], capsize=3, label="validação (média ± desvio)")
    ax.axhline(prev, c="gray", ls=":", label="acaso")
    ax.set_xticks(x, [_nome(n) for n in ordem])
    ax.set(ylabel="PR-AUC", title="Treino × validação (gap)")
    ax.legend(fontsize=8)
    figs["treino_x_validacao.png"] = fig

    mod2 = [n for n in ordem if n != REGRA]
    regra_val = mods[REGRA]["resumo"]["pr_auc_media"]
    for arq, chave, eixo, titulo in (("curva_aprendizado.png", "curva_aprendizado", "fracao", "Curva de aprendizado"),
                                     ("curva_complexidade.png", "curva_complexidade", "valor", "Curva de complexidade")):
        fig, axs = plt.subplots(1, len(mod2), figsize=(5.2 * len(mod2), 3.8), squeeze=False)
        for ax, n in zip(axs[0], mod2):
            pts = r[chave][n] if chave == "curva_aprendizado" else r[chave][n]["pontos"]
            xs = [q[eixo] for q in pts]
            for lado, cor, rot in (("treino", "tab:orange", "treino"), ("val", "tab:blue", "validação")):
                m = np.array([q[f"{lado}_media"] for q in pts])
                d = np.array([q[f"{lado}_desvio"] for q in pts])
                ax.plot(xs, m, "o-", c=cor, label=rot)
                ax.fill_between(xs, m - d, m + d, color=cor, alpha=0.18)
            ax.axhline(prev, c="gray", ls=":", label="acaso")
            ax.axhline(regra_val, c="tab:green", ls="--", lw=1, label="regra simples (validação)")
            if chave == "curva_aprendizado":
                ax.set(xlabel="fração do treino de cada fold (por par)")
            else:
                par = r[chave][n]["parametro"]
                ax.set(xlabel=par)
                if par == "C":
                    ax.set_xscale("log")
                ot = r[chave][n]["valor_otimizado"]
                if ot is not None:
                    ax.axvline(ot, c="tab:red", ls=":", lw=1, label=f"otimizado ({_val_curto(ot)})")
            ax.set(ylabel="PR-AUC", title=f"{titulo} — {_nome(n)}")
            ax.legend(fontsize=7)
        figs[arq] = fig

    ref = r["referencia_cc5"]["modelos"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, n in enumerate(ordem):
        s = mods[n]["resumo"]
        ax.errorbar(i, s["pr_auc_media"], yerr=s["pr_auc_desvio"], fmt="o", c="tab:blue", capsize=4,
                    label="CV (média ± desvio)" if i == 0 else None)
        ax.scatter(i, ref[n]["pr_auc_teste"], marker="D", c="tab:red", zorder=4, label="teste do CC5" if i == 0 else None)
    ax.axhline(r["referencia_cc5"]["prevalencia_teste"], c="gray", ls=":", label="acaso (teste)")
    ax.set_xticks(range(len(ordem)), [_nome(n) for n in ordem])
    ax.set(ylabel="PR-AUC", title="Generalização: CV × teste do CC5")
    ax.legend(fontsize=8)
    figs["cv_x_teste.png"] = fig
    return figs


# ======================================================================== documento
def _mp(m: float, d: float, n=3) -> str:
    return f"{_f(m, n)} ± {_f(d, n)}"


def _leitura_aprendizado(pts: list[dict]) -> dict:
    a, ult = pts[0], pts[-1]
    return {"gap_ini": a["treino_media"] - a["val_media"], "gap_fim": ult["treino_media"] - ult["val_media"],
            "ganho_ult": ult["val_media"] - pts[-2]["val_media"] if len(pts) > 1 else 0.0,
            "desvio_fim": ult["val_desvio"]}


def escrever_doc(r: dict, com_figuras: bool) -> str:
    p, mods, ref, t = r["protocolo"], r["modelos"], r["referencia_cc5"], r["tempos"]
    nomes = [n for n in mods if n != REGRA]
    ordem = list(mods)
    prev = p["prevalencia_cv"]
    todos = p["publico"] == "todos"
    nval = p["k"] * len(p["sementes"])
    fonte = ("> Usuários, obras, notas e o rótulo `conectou` (amizade no Last.fm) são reais (HetRec 2011 Last.fm, "
             "[`fontes_dados.md`](fontes_dados.md)); só os encontros são simulados, **independentes da amizade**. "
             "O rótulo é estático (as amizades não têm data): as métricas mostram quanto o gosto musical acompanha "
             "a amizade, não que o encontro a causou."
             if todos else
             "> Usuários, obras e notas são reais (Amazon Reviews 2023); encontros e o label `conectou` são simulados "
             "([`fontes_dados.md`](fontes_dados.md)). As métricas medem a simulação, não usuários reais.")
    L = ["# Validação cruzada, estabilidade e generalização — OpiniOn (CC6)", "",
         f"Gerado por `python -m opinion.modelo.validar` em {datetime.now():%d/%m/%Y %H:%M}. A validação cruzada usa "
         "**só treino + validação**; o teste não entra. Os números do teste vêm do CC5, apenas para comparar.", "", fonte, "",
         "## 1. Estratégia de validação", "",
         f"- **Dados:** {_n(p['n_cv'])} encontros de {_n(p['n_pares_cv'])} pares (os grupos de treino e validação do split por par), "
         f"{_n(p['positivos_cv'])} positivos (prevalência {_f(prev, 4)}). "
         f"Público avaliado: {'todos os encontros' if todos else 'desconhecidos'}.",
         f"- **{p['k']} folds, estratificados e agrupados por `par_id`** (`StratifiedGroupKFold`), repetidos com "
         f"{len(p['sementes'])} sementes ({', '.join(map(str, p['sementes']))}) → {nval} validações por modelo.",
         "- **Por que estratificada:** os positivos são raros (menos de 1%). Um k-fold simples poderia deixar um fold com "
         "poucos ou nenhum positivo, e a PR-AUC dele seria ruído ou nem existiria. A estratificação mantém a prevalência "
         "parecida em todos os folds; o código falha alto se algum fold (treino ou validação) ficar sem positivo.",
         "- **Por que agrupada por par:** cada par tem vários encontros com o mesmo rótulo (e features muito parecidas). "
         "Um k-fold comum por linha colocaria encontros do mesmo par no treino e na validação: o modelo \"reconheceria\" o "
         "par e a métrica ficaria inflada. Agrupar por `par_id` garante que um par inteiro está ou no treino ou na validação "
         "(é o mesmo critério do split treino/validação/teste do projeto, feito por par).",
         f"- **Por que não Leave-One-Out:** com {_n(p['n_cv'])} encontros seriam {_n(p['n_cv'])} ajustes por modelo, "
         "um custo inviável sem ganho de informação; pior, LOO por linha deixaria os outros encontros do mesmo par no "
         "treino (vazamento entre pares). O análogo correto seria deixar um par de fora por vez: caro e de alta variância; "
         "k = 5 agrupado é o compromisso usual.",
         "- **Padronização dentro do pipeline** da logística, ajustada só no treino de cada fold.",
         "- **Hiperparâmetros do CC5** (escolhidos na validação do CC5): "
         + "; ".join(f"{_nome(n)}: {_params(r['parametros'][n]['params'])}" for n in nomes)
         + ". A regra simples (`compat_score`) é usada como pontuação, sem ajuste.", "",
         "## 2. Resultado da validação cruzada", "",
         f"PR-AUC na validação de cada fold ({nval} valores por modelo). Desvio = desvio entre os folds "
         "(média dos desvios de cada semente); CV% = desvio / média. Gap = PR-AUC de treino − validação.", "",
         "| Modelo | PR-AUC (média ± desvio) | CV% | Mín–máx | ROC-AUC (média ± desvio) | PR-AUC no treino | Gap PR-AUC | Gap ROC-AUC |",
         "|---|---|---|---|---|---|---|---|"]
    for n in ordem:
        s = mods[n]["resumo"]
        L.append(f"| {_nome(n)} | {_mp(s['pr_auc_media'], s['pr_auc_desvio'])} | {_pct(s['pr_auc_cv'])} | "
                 f"{_f(s['pr_auc_min'])}–{_f(s['pr_auc_max'])} | {_mp(s['roc_auc_media'], s['roc_auc_desvio'])} | "
                 f"{_f(s['pr_auc_treino_media'])} | {_s(s['gap_pr_auc'])} | {_s(s['gap_roc_auc'])} |")
    L += [f"| *acaso (prevalência)* | *{_f(prev, 4)}* | | | *0,500* | | | |", ""]
    L += ["Média por semente (PR-AUC na validação) e variação entre sementes:", "",
          "| Modelo | " + " | ".join(f"Semente {s}" for s in p["sementes"]) + " | Desvio entre sementes |",
          "|---|" + "---|" * (len(p["sementes"]) + 1)]
    for n in ordem:
        s = mods[n]["resumo"]
        L.append(f"| {_nome(n)} | " + " | ".join(_f(s["por_semente"][str(x)]) for x in p["sementes"])
                 + f" | {_f(s['desvio_entre_sementes'], 4)} |")
    L.append("")
    pa = r["pareado"]
    if pa:
        L += [f"Diferença pareada ({_nome(pa['a'])} − {_nome(pa['b'])}, mesmo fold e semente): média {_s(pa['media'])}, "
              f"desvio {_f(pa['desvio'])}, de {_s(pa['minimo'])} a {_s(pa['maximo'])}; {_nome(pa['a'])} é melhor em "
              f"{pa['vitorias_a']} de {pa['n']} validações.", ""]
    if com_figuras:
        L += ["![Estabilidade](img/cc6/estabilidade.png)", "", "![Treino × validação](img/cc6/treino_x_validacao.png)", ""]

    L += ["## 3. Curva de aprendizado", "",
          "Cada ponto usa uma fração dos pares do treino de cada fold (a validação do fold é sempre a inteira); "
          f"{p['k']} folds, {len(p['sementes_curvas'])} semente(s) ({', '.join(map(str, p['sementes_curvas']))}) "
          "nas curvas, para limitar o tempo.", ""]
    for n in nomes:
        L += [f"**{_nome(n)}**", "", "| Fração | Encontros no treino | PR-AUC treino | PR-AUC validação (± desvio) | Gap |",
              "|---|---|---|---|---|"]
        for q in r["curva_aprendizado"][n]:
            L.append(f"| {_f(q['fracao'], 2)} | {_n(q['n_treino'])} | {_f(q['treino_media'])} | "
                     f"{_mp(q['val_media'], q['val_desvio'])} | {_s(q['treino_media'] - q['val_media'])} |")
        L.append("")
    if com_figuras:
        L += ["![Curva de aprendizado](img/cc6/curva_aprendizado.png)", ""]

    L += ["## 4. Curva de complexidade", "",
          "Varia um hiperparâmetro de complexidade e mantém os demais nos valores do CC5 "
          f"({p['k']} folds, {len(p['sementes_curvas'])} semente(s)).", ""]
    for n in nomes:
        c = r["curva_complexidade"][n]
        L += [f"**{_nome(n)}** — `{c['parametro']}` (otimizado no CC5: `{_val_curto(c['valor_otimizado'])}`)", "",
              f"| `{c['parametro']}` | PR-AUC treino | PR-AUC validação (± desvio) | Gap |", "|---|---|---|---|"]
        for q in c["pontos"]:
            L.append(f"| {_val_curto(q['valor'])} | {_f(q['treino_media'])} | {_mp(q['val_media'], q['val_desvio'])} | "
                     f"{_s(q['treino_media'] - q['val_media'])} |")
        L.append("")
    if com_figuras:
        L += ["![Curva de complexidade](img/cc6/curva_complexidade.png)", ""]

    L += ["## 5. Generalização: CV × teste do CC5", "",
          "O teste do CC5 (medido uma vez; reproduzido aqui reajustando no treino com os mesmos parâmetros) tem "
          f"{_n(ref['n_teste'])} encontros e prevalência {_f(ref['prevalencia_teste'], 4)}.", "",
          "| Modelo | PR-AUC CV (média ± desvio) | Validação do CC5 | Teste do CC5 | Teste − média CV | Em desvios da CV |",
          "|---|---|---|---|---|---|"]
    for n in ordem:
        s, q = mods[n]["resumo"], ref["modelos"][n]
        d = q["pr_auc_teste"] - s["pr_auc_media"]
        z = d / s["pr_auc_desvio"] if s["pr_auc_desvio"] else float("nan")
        L.append(f"| {_nome(n)} | {_mp(s['pr_auc_media'], s['pr_auc_desvio'])} | {_f(q['pr_auc_validacao'])} | "
                 f"{_f(q['pr_auc_teste'])} | {_s(d)} | {_s(z)} |")
    L += [f"| *acaso (prevalência)* | *{_f(prev, 4)}* | *{_f(ref['prevalencia_validacao'], 4)}* | "
          f"*{_f(ref['prevalencia_teste'], 4)}* | | |", ""]
    if com_figuras:
        L += ["![CV × teste](img/cc6/cv_x_teste.png)", ""]

    L += ["## 6. Análise", ""] + [f"- {x}" for x in analise(r)] + [""]
    L += ["## 7. Custo", "", "| Etapa | Tempo |", "|---|---|"]
    for n in nomes:
        L.append(f"| CV — {_nome(n)} ({nval} ajustes) | {_seg(t['cv_por_modelo_s'][n])} |")
    L += [f"| CV — total (inclui a regra) | {_seg(t['cv_s'])} |",
          f"| Curva de aprendizado | {_seg(t['curva_aprendizado_s'])} |",
          f"| Curva de complexidade | {_seg(t['curva_complexidade_s'])} |",
          f"| Reprodução do teste do CC5 | {_seg(t['referencia_cc5_s'])} |",
          f"| **Total** | **{_seg(t['total_s'])}** |", ""]
    L += ["## 8. Limitações", "",
          "- **Os hiperparâmetros foram escolhidos no CC5 com a validação** (15% dos pares), que aqui volta como parte dos "
          "folds: a média da CV é um pouco otimista para esses valores; o teste é a referência fora da escolha.",
          "- **A CV não tem dimensão temporal.** O rótulo é estático (amizade sem data) e os encontros são simulados; "
          "a CV mede estabilidade da amostragem entre pares, não deriva no tempo.",
          f"- **As sementes repetem os mesmos dados**: os folds se sobrepõem, então os {nval} valores não são "
          "independentes e o desvio não é um erro-padrão.",
          "- **Cada curva varia um fator por vez** e usa 1 semente (a CV usa mais); ela indica tendência, não um ponto exato.",
          "- **Rótulo estático e encontros simulados** (aviso no topo): ranking de modelos e de hiperparâmetros não é uma "
          "descoberta sobre usuários.", ""]
    return "\n".join(L)


def _nivel_sobreajuste(gap: float, rel: float) -> str:
    if gap <= 0.005 or rel < 0.05:
        return "sem sobreajuste"
    return "sobreajuste claro" if rel > 0.5 else "sobreajuste moderado" if rel > 0.2 else "sobreajuste pequeno"


def analise(r: dict) -> list[str]:
    p, mods, ref = r["protocolo"], r["modelos"], r["referencia_cc5"]
    nomes = [n for n in mods if n != REGRA]
    prev = p["prevalencia_cv"]
    out = []
    for n in nomes:
        s = mods[n]["resumo"]
        nivel = ("alta variação entre folds" if s["pr_auc_cv"] >= 0.25 else
                 "variação moderada" if s["pr_auc_cv"] >= 0.10 else "estável")
        out.append(f"**Estabilidade de {_nome(n)}:** PR-AUC {_mp(s['pr_auc_media'], s['pr_auc_desvio'])} "
                   f"(CV% {_pct(s['pr_auc_cv'])}; de {_f(s['pr_auc_min'])} a {_f(s['pr_auc_max'])}) — {nivel}. "
                   f"Entre sementes a média varia {_f(s['desvio_entre_sementes'], 4)}.")
    pa = r["pareado"]
    mel = None
    if pa:
        a, b = mods[pa["a"]]["resumo"], mods[pa["b"]]["resumo"]
        if pa["vitorias_a"] in (0, pa["n"]):
            vv = f"{_nome(pa['a'] if pa['vitorias_a'] else pa['b'])} vence em todas as {pa['n']} validações"
        else:
            vv = f"{_nome(pa['a'])} vence em {pa['vitorias_a']} de {pa['n']}"
        mel = pa["a"] if pa["media"] > 0 else pa["b"]
        ruido = abs(pa["media"]) < max(a["pr_auc_desvio"], b["pr_auc_desvio"])
        consistente = max(pa["vitorias_a"], pa["n"] - pa["vitorias_a"]) >= 0.8 * pa["n"]
        texto = ("A diferença é maior que o desvio entre folds: o melhor é consistente." if not ruido else
                 "A diferença é menor que o desvio entre folds de cada modelo, mas pareada tem o mesmo sinal na maioria "
                 "das validações: há uma tendência, de tamanho pequeno." if consistente else
                 "A diferença é menor que o desvio entre folds de cada modelo e o sinal muda de fold para fold: na prática "
                 "os dois são indistinguíveis com estes dados.")
        out.append(f"**{_nome(pa['a'])} × {_nome(pa['b'])}:** diferença média de {_s(pa['media'])} "
                   f"(a favor de {_nome(mel)}); {vv}. {texto}")
        if r.get("vencedor_cc5") and r["vencedor_cc5"] != mel:
            out.append(f"**Escolha do CC5:** a validação única escolheu {_nome(r['vencedor_cc5'])}, mas a CV não distingue os "
                       "dois (diferença menor que o desvio); a validação única apontou "
                       f"{_nome(r['vencedor_cc5'])} por margem pequena.")
        elif r.get("vencedor_cc5"):
            out.append(f"**Escolha do CC5:** a validação cruzada aponta o mesmo modelo, {_nome(mel)}, mas a margem acima vale.")
    regra = mods[REGRA]["resumo"]
    for n in nomes:
        s = mods[n]["resumo"]
        out.append(f"**{_nome(n)} contra a regra e o acaso:** PR-AUC média {_f(s['pr_auc_media'])} = "
                   f"{_f(s['pr_auc_media'] / prev, 1)}x a prevalência ({_f(prev, 4)}) e "
                   f"{_f(s['pr_auc_media'] / regra['pr_auc_media'], 1)}x a regra simples "
                   f"({_mp(regra['pr_auc_media'], regra['pr_auc_desvio'])}).")
    for n in nomes:
        s, ap, cx = mods[n]["resumo"], _leitura_aprendizado(r["curva_aprendizado"][n]), r["curva_complexidade"][n]
        rel = s["gap_pr_auc"] / s["pr_auc_media"] if s["pr_auc_media"] else 0
        nivel = _nivel_sobreajuste(s["gap_pr_auc"], rel)
        txt = (f"**Over/underfitting de {_nome(n)}:** gap treino−validação de {_s(s['gap_pr_auc'])} PR-AUC "
               f"({_pct(rel)} da média da validação) — {nivel}. ")
        if s["pr_auc_treino_media"] < 3 * prev:
            txt += (f"O treino também é baixo ({_f(s['pr_auc_treino_media'])}, {_f(s['pr_auc_treino_media'] / prev, 1)}x a "
                    "prevalência): sinal de subajuste (modelo ou features limitados), não só de variância. ")
        sobe = ap["ganho_ult"] > ap["desvio_fim"]
        txt += (f"Na curva de aprendizado o gap vai de {_s(ap['gap_ini'])} (menor fração) a {_s(ap['gap_fim'])} (treino inteiro) "
                f"e a validação {'ainda sobe' if sobe else 'está num platô'} no último passo ({_s(ap['ganho_ult'])}; "
                f"desvio {_f(ap['desvio_fim'])}): "
                + ("mais dados provavelmente ajudariam." if sobe else
                   "mais dados do mesmo tipo trariam pouco; o limite está no modelo, nas features e no rótulo."))
        pts = cx["pontos"]
        melhor = max(pts, key=lambda q: q["val_media"])
        i = pts.index(melhor)
        pos = "no extremo menos complexo" if i == 0 else "no extremo mais complexo" if i == len(pts) - 1 else "no interior"
        outro = f"; o valor do CC5 é `{_val_curto(cx['valor_otimizado'])}`" if cx["valor_otimizado"] is not None else ""
        txt += (f" Na curva de complexidade a melhor validação está em `{cx['parametro']}={_val_curto(melhor['valor'])}` "
                f"({_f(melhor['val_media'])}), {pos} da faixa testada{outro}; o gap vai de "
                f"{_s(pts[0]['treino_media'] - pts[0]['val_media'])} no primeiro valor a "
                f"{_s(pts[-1]['treino_media'] - pts[-1]['val_media'])} no último.")
        plana = (max(q["val_media"] for q in pts) - min(q["val_media"] for q in pts)) <= max(q["val_desvio"] for q in pts)
        if nivel == "sem sobreajuste" and plana:
            txt += " Treino ≈ validação e curva de complexidade plana: perfil de subajuste / limite de informação das features."
        elif cx["parametro"] == "max_depth" and nivel != "sem sobreajuste":
            txt += " Mais complexidade só amplia o gap, sem ganho na validação."
        out.append(txt)
    for n in nomes:
        s, q = mods[n]["resumo"], ref["modelos"][n]
        d = q["pr_auc_teste"] - s["pr_auc_media"]
        z = d / s["pr_auc_desvio"] if s["pr_auc_desvio"] else 0
        quando = ("dentro de 1 desvio da CV" if abs(z) <= 1 else
                  "entre 1 e 2 desvios da CV" if abs(z) <= 2 else "além de 2 desvios da CV")
        out.append(f"**Generalização de {_nome(n)}:** teste do CC5 {_f(q['pr_auc_teste'])} contra CV "
                   f"{_mp(s['pr_auc_media'], s['pr_auc_desvio'])} ({_s(d)}; {quando}); validação do CC5 "
                   f"{_f(q['pr_auc_validacao'])}. "
                   + ("O teste cai dentro da variação esperada entre folds: a diferença entre validação e teste do CC5 é, "
                      "em boa parte, ruído de split." if abs(z) <= 1 else
                      "O teste fica fora da variação típica dos folds: além do ruído de split pode haver diferença "
                      "sistemática entre os pares do teste e os demais (ou o teste tem poucos positivos)."))
    teste_ordem = max(nomes, key=lambda n: ref["modelos"][n]["pr_auc_teste"])
    cv_ordem = max(nomes, key=lambda n: mods[n]["resumo"]["pr_auc_media"])
    out.append(f"**Ordem dos modelos:** na CV o melhor é {_nome(cv_ordem)}; no teste do CC5, {_nome(teste_ordem)}. "
               + ("A ordem coincide." if cv_ordem == teste_ordem else
                  "A ordem difere, e as diferenças entre os dois modelos são pequenas diante do ruído: "
                  "não há como declarar um vencedor firme."))
    return out


# ======================================================================== MLflow
def _registrar(r: dict, figs: dict, texto: str, cfg: dict) -> dict:
    import warnings

    import mlflow
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    c = cfg["validacao_cruzada"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(c["experimento"])
    p, ref = r["protocolo"], r["referencia_cc5"]
    with mlflow.start_run(run_name=f"validacao-cruzada-{datetime.now():%Y%m%d-%H%M}") as pai:
        mlflow.log_params({"k": p["k"], "sementes": ",".join(map(str, p["sementes"])),
                           "sementes_curvas": ",".join(map(str, p["sementes_curvas"])),
                           "n_cv": p["n_cv"], "n_pares_cv": p["n_pares_cv"], "publico": p["publico"],
                           "vencedor_cc5": r["vencedor_cc5"]})
        mlflow.log_metrics({"prevalencia_cv": p["prevalencia_cv"], "prevalencia_teste": ref["prevalencia_teste"],
                            **{f"tempo_{k}": v for k, v in r["tempos"].items() if isinstance(v, float)}})
        mlflow.log_text(texto, "validacao_cruzada.md")
        mlflow.log_dict({k: v for k, v in r.items() if not k.startswith("_")}, "resumo.json")
        for nome, fig in figs.items():
            mlflow.log_figure(fig, nome)
        for n, m in r["modelos"].items():
            with mlflow.start_run(run_name=n, nested=True):
                if n in r["parametros"]:
                    mlflow.log_params({"modelo": n, "tipo": r["parametros"][n]["tipo"],
                                       **{f"hp_{k}": v for k, v in r["parametros"][n]["params"].items()}})
                else:
                    mlflow.log_params({"modelo": n, "tipo": "regra compat_score"})
                mlflow.log_metrics({k: v for k, v in m["resumo"].items() if isinstance(v, float)})
                mlflow.log_metrics({"cc5_pr_auc_validacao": ref["modelos"][n]["pr_auc_validacao"],
                                    "cc5_pr_auc_teste": ref["modelos"][n]["pr_auc_teste"]})
                mlflow.log_text(pd.DataFrame(m["folds"]).to_csv(index=False), "folds.csv")
                for i, f in enumerate(m["folds"]):
                    mlflow.log_metric("pr_auc_val_fold", f["pr_auc_val"], step=i)
    return {"mlflow_run_pai": pai.info.run_id}


# ======================================================================== CLI
def _imprimir(r: dict) -> None:
    p, ref = r["protocolo"], r["referencia_cc5"]
    print(f"\nCV: {p['k']} folds x {len(p['sementes'])} sementes, {p['n_cv']} encontros, prevalência {p['prevalencia_cv']:.4f}")
    print(f"{'':20} {'PR-AUC':>16} {'CV%':>6} {'ROC':>6} {'treino':>7} {'gap':>7} {'teste CC5':>10}")
    for n, m in r["modelos"].items():
        s = m["resumo"]
        print(f"{_nome(n):20} {s['pr_auc_media']:8.3f} ± {s['pr_auc_desvio']:.3f} {100 * s['pr_auc_cv']:5.1f}% "
              f"{s['roc_auc_media']:6.3f} {s['pr_auc_treino_media']:7.3f} {s['gap_pr_auc']:+7.3f} "
              f"{ref['modelos'][n]['pr_auc_teste']:10.3f}")
    print(f"(prevalência no teste {ref['prevalencia_teste']:.4f})\n")
    for x in analise(r):
        print(" -", x.replace("**", ""))
    if r.get("mlflow_run_pai"):
        print(f"\nMLflow: run {r['mlflow_run_pai']}")
    print(f"Tempo total: {_seg(r['tempos']['total_s'])}")


def main():
    ap = argparse.ArgumentParser(description="CC6: validação cruzada agrupada, estabilidade e curvas")
    ap.add_argument("--sem-registro", action="store_true", help="não grava no MLflow nem reescreve o doc")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    print("Validando (treino + validação; teste fora):")
    if a.sem_registro:
        r = validar(a.pasta, registrar=False, doc=None, figuras=None)
    else:
        r = validar(a.pasta)
        print(f"\nRelatório em {DOC}")
    _imprimir(r)


if __name__ == "__main__":
    main()
