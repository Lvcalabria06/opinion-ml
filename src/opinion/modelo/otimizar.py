"""
CC5 — otimização de hiperparâmetros com Optuna (regressão logística e Random Forest).

Mesmo protocolo do CC3/CC4 (opinion.modelo.treinar): cada tentativa ajusta no "treino" e é
pontuada na "validacao" (PR-AUC, `metrica` do config); o limiar do sinal (F0,5) também sai da
validação. O melhor de cada família é reajustado uma vez e o "teste" é medido uma única vez,
só para relatar. A família vencedora é escolhida na validação.

No MLflow: um run pai (comparação) e um run filho por família (parâmetros, métricas de validação
e teste, tentativas.csv, histórico da otimização, importância dos hiperparâmetros, modelo).
O relatório é docs/otimizacao.md e os melhores parâmetros vão para data/processed/otimizacao.json
(lido pelo CC6).

    python -m opinion.modelo.otimizar
    python -m opinion.modelo.otimizar --sem-registro     # sem MLflow e sem reescrever o doc
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

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.treinar import (_graficos, _metricas_planas, avaliar_candidato, avaliar_regra, candidatos,
                                    carregar, chave_metrica, config, mascara_desconhecidos, medir_teste)
from opinion.mundo import RAIZ

DOC = RAIZ / "docs" / "otimizacao.md"
FIGURAS = RAIZ / "docs" / "img" / "cc5"
NOMES = {"regressao_logistica": "Regressão logística", "random_forest": "Random Forest", "regra": "Regra simples"}


def _nome(k: str) -> str:
    return NOMES.get(k, k)


# ======================================================================== espaço de busca
def sugerir(trial, espaco: dict) -> dict:
    """Traduz o espaço do config em chamadas `trial.suggest_*` e devolve os parâmetros do sklearn."""
    params = {}
    for nome, e in espaco.items():
        t = e["tipo"]
        if t == "float_log":
            params[nome] = trial.suggest_float(nome, float(e["min"]), float(e["max"]), log=True)
        elif t == "int":
            params[nome] = trial.suggest_int(nome, int(e["min"]), int(e["max"]))
        elif t == "int_log":
            params[nome] = trial.suggest_int(nome, int(e["min"]), int(e["max"]), log=True)
        elif t == "categorico":
            params[nome] = trial.suggest_categorical(nome, list(e["valores"]))
        else:
            raise ValueError(f"tipo de espaço desconhecido: {t} ({nome})")
    return params


def _otimizar_familia(tipo, spec, tr, va, cols, desc_va, beta, chave, semente) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tempos = []

    def objetivo(trial):
        t0 = time.time()
        params = sugerir(trial, spec["espaco"])
        r = avaliar_candidato("optuna", tipo, params, tr, va, cols, desc_va, beta)
        tempos.append(time.time() - t0)
        return r["validacao"][chave] or 0.0

    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semente))
    t0 = time.time()
    estudo.optimize(objetivo, n_trials=int(spec["tentativas"]))
    tempo_total = time.time() - t0
    return {"estudo": estudo, "tempo_total_s": tempo_total, "tempo_tentativa_s": float(np.mean(tempos)),
            "tempo_max_tentativa_s": float(np.max(tempos))}


# ======================================================================== otimização
def otimizar(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True, doc: Path | None = DOC,
             figuras: Path | None = FIGURAS) -> dict:
    cfg = cfg or config()
    o = cfg["otimizacao"]
    beta, chave = float(cfg["limiar"]["beta"]), chave_metrica(cfg)
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    for nome, parte in (("treino", tr), ("validacao", va), ("teste", te)):
        if parte.empty:
            raise ValueError(f"split {nome} vazio")
    desc_va, desc_te = mascara_desconhecidos(va, cfg).to_numpy(), mascara_desconhecidos(te, cfg).to_numpy()
    regra = avaliar_regra(va, te, desc_va, desc_te, beta)

    familias = {}
    for fam, spec in o["familias"].items():
        # Busca: só treino -> validação. O teste não entra aqui.
        b = _otimizar_familia(spec["tipo"], spec, tr, va, cols, desc_va, beta, chave, int(o["semente"]))
        estudo = b["estudo"]
        melhor = avaliar_candidato(fam, spec["tipo"], sugerir_fixo(estudo.best_trial, spec["espaco"]),
                                   tr, va, cols, desc_va, beta)
        p_te, met_te = medir_teste(melhor, te, cols, desc_te, beta)   # medição única do teste

        grade_cc4 = None
        if fam in cfg.get("comparacao", {}).get("modelos", {}):
            g = [avaliar_candidato(n, t, p, tr, va, cols, desc_va, beta)
                 for n, t, p in candidatos(cfg, {fam: cfg["comparacao"]["modelos"][fam]})]
            gm = max(g, key=lambda r: r["validacao"][chave] or 0)
            _, gt = medir_teste(gm, te, cols, desc_te, beta)
            grade_cc4 = {"params": gm["params"], "limiar": gm["limiar"], "validacao": gm["validacao"], "teste": gt,
                         "n_configuracoes": len(g)}
        familias[fam] = {
            **melhor, "teste": met_te, "p_teste": p_te, "n_tentativas": len(estudo.trials),
            "tentativas": [{"params": t.params, "valor": t.value} for t in estudo.trials],
            "melhor_grade_cc4": grade_cc4, "espaco": spec["espaco"], "_estudo": estudo,
            **{k: v for k, v in b.items() if k != "estudo"}}
        print(f"  {_nome(fam):<20} {len(estudo.trials)} tentativas  validação {chave} {melhor['validacao'][chave]:.3f}  "
              f"teste {met_te[chave]:.3f}  ({b['tempo_total_s']:.0f} s, {b['tempo_tentativa_s']:.1f} s/tentativa)")

    vencedor = max(familias, key=lambda k: familias[k]["validacao"][chave] or 0)
    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "protocolo": {"metrica_escolha": chave, "beta_limiar": beta, "publico": cfg.get("publico", "desconhecidos"),
                      "semente": int(o["semente"]), "n_features": len(cols),
                      "n_treino": len(tr), "n_validacao": len(va), "n_teste": len(te)},
        "vencedor": vencedor,
        "familias": {k: {x: f[x] for x in ("tipo", "params", "limiar", "validacao", "teste", "n_tentativas",
                                           "melhor_grade_cc4", "tempo_total_s", "tempo_tentativa_s",
                                           "tempo_max_tentativa_s", "espaco", "tentativas")}
                     for k, f in familias.items()},
        "regra_simples": regra,
        "_familias": familias, "_te": te, "_desc_te": desc_te,
    }
    # Contrato com o CC6: melhores parâmetros de cada família (escolhidos na validação).
    (Path(pasta) / "otimizacao.json").write_text(json.dumps(
        {"vencedor": vencedor, "familias": {k: {"tipo": f["tipo"], "params": f["params"]}
                                            for k, f in familias.items()}}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    figs = _figuras(resumo)
    if figuras is not None:
        Path(figuras).mkdir(parents=True, exist_ok=True)
        for nome, fig in figs.items():
            (Path(figuras) / nome).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(figuras) / nome, dpi=110, bbox_inches="tight")
    texto = escrever_doc(resumo, figuras is not None)
    if doc is not None:
        Path(doc).write_text(texto, encoding="utf-8")
    if registrar:
        resumo.update(_registrar(resumo, figs, texto, cfg, tr[cols].head(5)))
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    return resumo


def sugerir_fixo(trial, espaco: dict) -> dict:
    """Parâmetros de um trial já concluído (na ordem do espaço), no formato do sklearn."""
    return {nome: trial.params[nome] for nome in espaco}


# ======================================================================== gráficos
def _figuras(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import optuna

    chave = r["protocolo"]["metrica_escolha"]
    figs = {}
    for fam, f in r["_familias"].items():
        est = f["_estudo"]
        v = [t.value for t in est.trials]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.scatter(range(len(v)), v, s=18, alpha=0.6, label="tentativa")
        ax.plot(range(len(v)), np.maximum.accumulate(v), c="tab:red", label="melhor até agora")
        ax.set(title=f"{_nome(fam)} — histórico da otimização", xlabel="tentativa",
               ylabel=f"{chave} (validação)")
        ax.legend(fontsize=8)
        figs[f"{fam}/historico.png"] = fig
        try:
            imp = optuna.importance.get_param_importances(est)
        except Exception:    # poucas tentativas ou parâmetro constante
            imp = {}
        if imp:
            fig, ax = plt.subplots(figsize=(6.2, 3.2))
            s = pd.Series(imp).sort_values()
            ax.barh(s.index, s.values, color="tab:blue")
            ax.set(title=f"{_nome(fam)} — importância dos hiperparâmetros", xlabel="importância relativa")
            figs[f"{fam}/importancia_hiperparametros.png"] = fig
            f["importancia_hp"] = {k: float(x) for k, x in imp.items()}
        else:
            f["importancia_hp"] = {}

    fam, te, d = r["_familias"], r["_te"], r["_desc_te"]
    ordem = list(fam)
    x = np.arange(len(ordem))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    larg = 0.2
    series = [("Grade CC4 — validação", [fam[k]["melhor_grade_cc4"]["validacao"][chave] if fam[k]["melhor_grade_cc4"] else np.nan for k in ordem]),
              ("Optuna — validação", [fam[k]["validacao"][chave] for k in ordem]),
              ("Grade CC4 — teste", [fam[k]["melhor_grade_cc4"]["teste"][chave] if fam[k]["melhor_grade_cc4"] else np.nan for k in ordem]),
              ("Optuna — teste", [fam[k]["teste"][chave] for k in ordem])]
    for i, (rot, vals) in enumerate(series):
        ax.bar(x + (i - 1.5) * larg, vals, larg, label=rot)
    ax.axhline(te.conectou.to_numpy()[d].mean(), c="gray", ls=":", label="acaso (prevalência)")
    ax.set_xticks(x, [_nome(k) for k in ordem])
    ax.set(ylabel="PR-AUC", title="Melhor da grade do CC4 × melhor do Optuna")
    ax.legend(fontsize=7)
    figs["grade_x_optuna.png"] = fig
    return figs


# ======================================================================== documento
def _f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}".replace(".", ",")


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def _s(x) -> str:
    return f"{x:+.3f}".replace(".", ",")


def _lista(nomes: list[str]) -> str:
    n = [f"`{x}`" for x in nomes]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " e " + n[-1]


def _params(p: dict) -> str:
    return ", ".join(f"`{k}={v:.4g}`" if isinstance(v, float) else f"`{k}={v}`" for k, v in p.items())


def _espaco(e: dict) -> str:
    if e["tipo"] == "categorico":
        return "{" + ", ".join("nulo" if v is None else str(v) for v in e["valores"]) + "}"
    esc = " (escala log)" if e["tipo"].endswith("log") else ""
    return f"{e['min']} a {e['max']}{esc}"


def _seg(s: float) -> str:
    return f"{s / 60:.1f} min".replace(".", ",") if s >= 120 else f"{s:.1f} s".replace(".", ",")


def escrever_doc(r: dict, com_figuras: bool) -> str:
    fam, regra, venc, p = r["_familias"], r["regra_simples"], r["vencedor"], r["protocolo"]
    chave = p["metrica_escolha"]
    todos = p["publico"] == "todos"
    pub = "todos os encontros" if todos else "desconhecidos"
    prev_te = fam[venc]["teste"]["prevalencia"]
    fonte = ("> Usuários, obras, notas e o rótulo `conectou` (amizade no Last.fm) são reais (HetRec 2011 Last.fm, "
             "[`fontes_dados.md`](fontes_dados.md)); só os encontros são simulados, **independentes da amizade**. "
             "O rótulo é estático (as amizades não têm data): as métricas mostram quanto o gosto musical acompanha "
             "a amizade, não que o encontro a causou."
             if todos else
             "> Usuários, obras e notas são reais (Amazon Reviews 2023); encontros e o label `conectou` são simulados "
             "([`fontes_dados.md`](fontes_dados.md)). As métricas medem a simulação, não usuários reais.")
    L = ["# Otimização de hiperparâmetros — OpiniOn (CC5)", "",
         f"Gerado por `python -m opinion.modelo.otimizar` em {datetime.now():%d/%m/%Y %H:%M}. "
         "Números do teste medidos **uma vez** por família (e a melhor da grade do CC4, só para comparação); nada foi escolhido com eles.", "", fonte, "",
         "## 1. Estratégia", "",
         "- **Busca bayesiana (TPE, Optuna)** com semente fixa; o objetivo de cada tentativa é a "
         f"{chave} na validação ({pub}), com o modelo ajustado só no treino ({_n(p['n_treino'])} encontros; "
         f"validação {_n(p['n_validacao'])}, teste {_n(p['n_teste'])}).",
         "- **Padronização dentro do pipeline** (logística), ajustada só no treino de cada tentativa.",
         f"- **Limiar do sinal** (F{_f(p['beta_limiar'], 1)}) escolhido na validação, como no CC3/CC4.",
         "- **O melhor de cada família é reajustado e o teste é medido uma única vez.** A família vencedora "
         "também é escolhida na validação.",
         "- Espaços e número de tentativas ficam em `config/modelo.yaml → otimizacao`.", "",
         "## 2. Espaços de busca", ""]
    for k, f in fam.items():
        L += [f"**{_nome(k)}** — {f['n_tentativas']} tentativas", "", "| Hiperparâmetro | Espaço |", "|---|---|"]
        L += [f"| `{n}` | {_espaco(e)} |" for n, e in f["espaco"].items()]
        L.append("")
    L += ["## 3. Melhor da grade do CC4 × melhor do Optuna", "",
          f"PR-AUC {('em ' + pub) if todos else 'entre ' + pub}. Prevalência no teste: {_f(prev_te)}; "
          f"regra simples (`compat_score`): validação {_f(regra['validacao'][chave])}, teste {_f(regra['teste'][chave])}.", "",
          "| Família | Grade CC4 (valid.) | Optuna (valid.) | Grade CC4 (teste) | Optuna (teste) | Δ teste |",
          "|---|---|---|---|---|---|"]
    for k, f in fam.items():
        g = f["melhor_grade_cc4"]
        d = _s(_dif(f["teste"][chave], g["teste"][chave])) if g else "—"
        L.append(f"| {_nome(k)}{' **(escolhido)**' if k == venc else ''} | "
                 f"{_f(g['validacao'][chave]) if g else '—'} | {_f(f['validacao'][chave])} | "
                 f"{_f(g['teste'][chave]) if g else '—'} | {_f(f['teste'][chave])} | {d} |")
    L += [f"| Regra simples | {_f(regra['validacao'][chave])} | | {_f(regra['teste'][chave])} | | |",
          f"| *acaso (prevalência)* | | | *{_f(prev_te)}* | | |", ""]
    if com_figuras:
        L += ["![Grade × Optuna](img/cc5/grade_x_optuna.png)", ""]

    L += ["## 4. Parâmetros vencedores", "", "| Família | Parâmetros | Limiar | Valid. PR-AUC | Teste PR-AUC | ROC-AUC (teste) | Brier (teste) |",
          "|---|---|---|---|---|---|---|"]
    for k, f in fam.items():
        L.append(f"| {_nome(k)} | {_params(f['params'])} | {_f(f['limiar'])} | {_f(f['validacao'][chave])} | "
                 f"{_f(f['teste'][chave])} | {_f(f['teste']['roc_auc'])} | {_f(f['teste']['brier'])} |")
    L += ["", f"### No ponto de operação (teste, {pub})", "",
          "| Modelo | Limiar | Precisão | Recall | F0,5 | Sinal em | VP | FP | FN | VN |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in list(fam) + ["regra"]:
        nl = (fam[k]["teste"] if k != "regra" else regra["teste"])["no_limiar_desconhecidos"]
        lim = fam[k]["limiar"] if k != "regra" else regra["limiar"]
        L.append(f"| {_nome(k)} | {_f(lim)} | {_pct(nl['precisao'])} | {_pct(nl['recall'])} | {_f(nl['f05'])} | "
                 f"{_pct(nl['taxa_sinal'])} | {_n(nl['vp'])} | {_n(nl['fp'])} | {_n(nl['fn'])} | {_n(nl['vn'])} |")
    L.append("")
    if com_figuras:
        for k in fam:
            L += [f"![Histórico {_nome(k)}](img/cc5/{k}/historico.png)", ""]
            if fam[k].get("importancia_hp"):
                L += [f"![Importância dos hiperparâmetros {_nome(k)}](img/cc5/{k}/importancia_hiperparametros.png)", ""]

    L += ["## 5. Custo", "", "| Família | Tentativas | Tempo total | Por tentativa | Mais lenta |", "|---|---|---|---|---|"]
    for k, f in fam.items():
        L.append(f"| {_nome(k)} | {f['n_tentativas']} | {_seg(f['tempo_total_s'])} | {_seg(f['tempo_tentativa_s'])} | "
                 f"{_seg(f['tempo_max_tentativa_s'])} |")
    L += ["", "## 6. Análise", ""] + [f"- {t}" for t in analise(r)] + [""]
    L += ["## 7. Limitações", "",
          "- **Um único split por par.** A diferença entre grade e Optuna no teste pode estar dentro do ruído de "
          "amostragem; a estabilidade entre folds é o CC6.",
          "- **O valor da validação das tentativas é otimista**: o melhor de N tentativas foi escolhido por esse "
          "mesmo número. O teste (medido uma vez) é a referência honesta.",
          "- **Rótulo estático e encontros simulados** (ver aviso no topo): as métricas medem o quanto o gosto "
          "musical acompanha a amizade; ranking de modelos e de hiperparâmetros não é uma descoberta sobre usuários.", ""]
    return "\n".join(L)


def limites(params: dict, espaco: dict, margem: float = 0.05) -> dict[str, str]:
    """Parâmetros numéricos cujo melhor valor está a menos de `margem` da faixa de um limite ("min" ou "max");
    escala log se o espaço for log."""
    out = {}
    for nome, e in espaco.items():
        v = params.get(nome)
        if e["tipo"] == "categorico" or v is None:
            continue
        lo, hi, x = float(e["min"]), float(e["max"]), float(v)
        if e["tipo"].endswith("log"):
            lo, hi, x = np.log(lo), np.log(hi), np.log(x)
        if hi > lo and min(x - lo, hi - x) <= margem * (hi - lo):
            out[nome] = "min" if x - lo <= hi - x else "max"
    return out


def no_limite(params: dict, espaco: dict, margem: float = 0.05) -> list[str]:
    return list(limites(params, espaco, margem))


def _dif(a: float, b: float) -> float:
    """Diferença entre os valores já arredondados como são exibidos (3 casas)."""
    return round(a, 3) - round(b, 3)


def _val(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def nota_limites(nome_fam: str, f: dict, ganho_val: float | None, ganho_teste: float | None) -> str | None:
    """Interpretação dos parâmetros na borda do espaço (sem sugerir ampliar o espaço)."""
    lim = limites(f["params"], f["espaco"])
    if not lim:
        return None
    desc = ", ".join(f"`{n}={_val(f['params'][n])}` ({'mínimo' if lado == 'min' else 'máximo'} do espaço)"
                     for n, lado in lim.items())
    t = (f"**{nome_fam}: o melhor ponto está na borda do espaço de busca** ({desc}); não é um ótimo interior.")
    # regularizadores empurrados para "mais simples": direção declarada no config (`regulariza`)
    reg = [n for n, lado in lim.items()
           if f["espaco"][n].get("regulariza") == ("menor" if lado == "min" else "maior")]
    neutros = [n for n in lim if n not in reg]
    if reg:
        t += (f" Em {_lista(reg)} a busca empurra para a regularização máxima, isto é, para modelos mais simples.")
    if neutros:
        t += (f" {_lista(neutros)} na borda: nota neutra, sem leitura de regularização nessa direção.")
    if ganho_val is not None:
        if ganho_teste is not None and ganho_teste > 0:
            t += (f" O ganho do Optuna sobre a grade do CC4 na validação ({_s(ganho_val)}) também apareceu no teste "
                  f"({_s(ganho_teste)}).")
        else:
            t += (f" O ganho do Optuna sobre a grade do CC4 na validação ({_s(ganho_val)}) não se transferiu ao teste "
                  f"({_s(ganho_teste)}); por isso o espaço foi mantido, sem ampliá-lo mais.")
    return t


def analise(r: dict) -> list[str]:
    fam, regra, venc, chave = r["_familias"], r["regra_simples"], r["vencedor"], r["protocolo"]["metrica_escolha"]
    out = [f"**Escolhido na validação: {_nome(venc)}** (PR-AUC {_f(fam[venc]['validacao'][chave])}; "
           f"teste {_f(fam[venc]['teste'][chave])})."]
    for k, f in fam.items():
        g = f["melhor_grade_cc4"]
        if not g:
            continue
        dv, dt = _dif(f["validacao"][chave], g["validacao"][chave]), _dif(f["teste"][chave], g["teste"][chave])
        peq = abs(dv) < 0.005 and abs(dt) < 0.005
        out.append(f"**{_nome(k)}:** Optuna {_s(dv)} na validação e {_s(dt)} no teste em relação à melhor da grade do CC4 — "
                   + ("ganho pequeno: a grade pequena pode já estar perto do ótimo e o ajuste fino render pouco." if peq and dt >= 0
                      else "o ajuste fino não ajudou no teste (a diferença pode estar dentro do ruído de uma única divisão; só o CC6 estima esse ruído)." if dt <= 0
                      else "melhora perceptível, que a validação cruzada do CC6 deve confirmar."))
    melhor_te = max(fam, key=lambda k: fam[k]["teste"][chave])
    if melhor_te != venc:
        a, b = fam[venc], fam[melhor_te]
        out.append(f"**O vencedor na validação ({_nome(venc)} {_f(a['validacao'][chave])} × {_nome(melhor_te)} "
                   f"{_f(b['validacao'][chave])}) fica abaixo de {_nome(melhor_te)} no teste ({_f(a['teste'][chave])} × "
                   f"{_f(b['teste'][chave])}).** A diferença na validação é de {_f(_dif(a['validacao'][chave], b['validacao'][chave]))}; "
                   "a escolha seguiu a regra da validação, e o CC6 (validação cruzada) decide com mais dados.")
    for k, f in fam.items():
        g = f["melhor_grade_cc4"]
        nota = nota_limites(_nome(k), f,
                            _dif(f["validacao"][chave], g["validacao"][chave]) if g else None,
                            _dif(f["teste"][chave], g["teste"][chave]) if g else None)
        if nota:
            out.append(nota)
    prev = fam[venc]["teste"]["prevalencia"]
    out.append(f"**Contra a prevalência e a regra simples:** o vencedor tem PR-AUC {_f(fam[venc]['teste'][chave])} no teste, "
               f"{_f(fam[venc]['teste'][chave] / prev, 1)}x a prevalência ({_f(prev)}); a regra `compat_score` tem "
               f"{_f(regra['teste'][chave])} ({_s(_dif(fam[venc]['teste'][chave], regra['teste'][chave]))} de diferença).")
    nl = fam[venc]["teste"]["no_limiar_desconhecidos"]
    out.append(f"**No ponto de operação**, {_nome(venc)} acerta {_pct(nl['precisao'])} dos sinais (recall "
               f"{_pct(nl['recall'])}, sinal em {_pct(nl['taxa_sinal'])} dos encontros).")
    for k, f in fam.items():
        if f.get("importancia_hp"):
            top = max(f["importancia_hp"], key=f["importancia_hp"].get)
            out.append(f"**Hiperparâmetro mais influente em {_nome(k)}:** `{top}` "
                       f"({_pct(f['importancia_hp'][top])} da importância); a busca mostra o que importa neste conjunto, "
                       "não uma regra geral.")
    return out


# ======================================================================== MLflow
def _registrar(r: dict, figs: dict, texto: str, cfg: dict, exemplo) -> dict:
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    import warnings
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    o = cfg["otimizacao"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(o["experimento"])
    fam, te, d, regra = r["_familias"], r["_te"], r["_desc_te"], r["regra_simples"]
    info_venc = None

    with mlflow.start_run(run_name=f"otimizacao-{datetime.now():%Y%m%d-%H%M}") as pai:
        mlflow.log_params({"vencedor": r["vencedor"], "metrica_escolha": r["protocolo"]["metrica_escolha"],
                           "beta_limiar": r["protocolo"]["beta_limiar"], "semente": r["protocolo"]["semente"],
                           "n_familias": len(fam)})
        mlflow.log_metrics(_metricas_planas("regra_validacao", regra["validacao"]))
        mlflow.log_metrics(_metricas_planas("regra_teste", regra["teste"]))
        mlflow.log_text(texto, "otimizacao.md")
        mlflow.log_dict({k: v for k, v in r.items() if not k.startswith("_")}, "resumo.json")
        for caminho, fig in figs.items():
            if "/" not in caminho:
                mlflow.log_figure(fig, caminho)

        for k, f in fam.items():
            with mlflow.start_run(run_name=k, nested=True) as filho:
                mlflow.log_params({"modelo": k, "tipo": f["tipo"], "limiar": f["limiar"],
                                   "limiar_regra": regra["limiar"], "n_tentativas": f["n_tentativas"],
                                   **{f"hp_{p}": v for p, v in f["params"].items()}})
                mlflow.log_metrics(_metricas_planas("validacao", f["validacao"]))
                mlflow.log_metrics(_metricas_planas("teste", f["teste"]))
                mlflow.log_metrics({"tempo_total_s": f["tempo_total_s"], "tempo_tentativa_s": f["tempo_tentativa_s"]})
                if f["melhor_grade_cc4"]:
                    mlflow.log_metrics({"grade_cc4_validacao_pr_auc": f["melhor_grade_cc4"]["validacao"]["pr_auc"],
                                        "grade_cc4_teste_pr_auc": f["melhor_grade_cc4"]["teste"]["pr_auc"]})
                linhas = [{"tentativa": i, "valor_validacao": t["valor"], **t["params"]}
                          for i, t in enumerate(f["tentativas"])]
                mlflow.log_text(pd.DataFrame(linhas).to_csv(index=False), "tentativas.csv")
                for caminho, fig in figs.items():
                    if caminho.startswith(f"{k}/"):
                        mlflow.log_figure(fig, caminho)
                preds = {"y": te.conectou.to_numpy(), "p": f["p_teste"], "regra": te.compat_score.to_numpy(), "desc": d,
                         "todos": r["protocolo"].get("publico") == "todos"}
                for caminho, fig in _graficos(preds, f["limiar"], regra["limiar"]).items():
                    mlflow.log_figure(fig, caminho)
                    plt.close(fig)
                info = mlflow.sklearn.log_model(
                    sk_model=f["modelo"], name="modelo", input_example=exemplo,
                    signature=infer_signature(exemplo, f["modelo"].predict_proba(exemplo)[:, 1]),
                    serialization_format="cloudpickle")
                if k == r["vencedor"]:
                    info_venc = (info, filho.info.run_id)

    mv = mlflow.register_model(info_venc[0].model_uri, o["modelo_registrado"])
    MlflowClient().set_registered_model_alias(o["modelo_registrado"], o["alias"], mv.version)
    return {"mlflow_run_pai": pai.info.run_id, "mlflow_run_vencedor": info_venc[1],
            "mlflow_modelo": f"{o['modelo_registrado']}@{o['alias']}", "mlflow_versao": str(mv.version)}


# ======================================================================== CLI
def _imprimir(r: dict) -> None:
    fam, regra, chave = r["_familias"], r["regra_simples"], r["protocolo"]["metrica_escolha"]
    print(f"\n{'':20} {'CC4 val':>8} {'Optuna val':>10} {'CC4 teste':>10} {'Opt. teste':>10} {'ROC':>6} {'Brier':>6} "
          f"{'prec.':>6} {'recall':>6}")
    for k, f in fam.items():
        g, nl = f["melhor_grade_cc4"], f["teste"]["no_limiar_desconhecidos"]
        print(f"{_nome(k) + (' *' if k == r['vencedor'] else ''):20} {g['validacao'][chave]:8.3f} "
              f"{f['validacao'][chave]:10.3f} {g['teste'][chave]:10.3f} {f['teste'][chave]:10.3f} "
              f"{f['teste']['roc_auc']:6.3f} {f['teste']['brier']:6.4f} {nl['precisao']:6.3f} {nl['recall']:6.3f}")
        print(f"   params: {f['params']}  limiar {f['limiar']:.4f}")
    print(f"{'Regra simples':20} {regra['validacao'][chave]:8.3f} {'':10} {regra['teste'][chave]:10.3f}")
    print(f"(prevalência no teste {fam[r['vencedor']]['teste']['prevalencia']:.4f}; * = escolhido na validação)\n")
    for t in analise(r):
        print(" -", t.replace("**", ""))
    if r.get("mlflow_run_pai"):
        print(f"\nMLflow: run {r['mlflow_run_pai']}  ·  melhor: {r['mlflow_modelo']} v{r['mlflow_versao']}")


def main():
    ap = argparse.ArgumentParser(description="CC5: otimiza hiperparâmetros com Optuna e registra no MLflow")
    ap.add_argument("--sem-registro", action="store_true", help="não grava no MLflow nem reescreve o doc")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    print("Otimizando (busca no treino → validação; teste medido uma vez):")
    if a.sem_registro:
        r = otimizar(a.pasta, registrar=False, doc=None, figuras=None)
    else:
        r = otimizar(a.pasta)
        print(f"\nRelatório em {DOC}")
    _imprimir(r)


if __name__ == "__main__":
    main()
