"""
CC5 — otimização de hiperparâmetros do modelo selecionado no CC4.

Estratégia: otimização bayesiana com Optuna (amostrador TPE), comparada com uma busca
aleatória com o mesmo orçamento e com a grade fixa do CC4. O protocolo não muda:

  * cada tentativa ajusta o modelo no "treino" (semanas 1–8) e é avaliada pela PR-AUC
    entre desconhecidos na "validacao" (semanas 9–10);
  * a melhor configuração é reajustada, as probabilidades são calibradas na validação
    (sigmoide: transformação monótona, não muda a ordenação) e o limiar do sinal (F0,5)
    é escolhido na validação;
  * o "teste" (semanas 11–12) é medido uma única vez, no fim, só para relatar.

Tudo vai para o MLflow: um run pai, um run filho por tentativa (parâmetros e métrica) e o
modelo final, registrado como `opinion-cc5-otimizado@otimizado` (ou `-<midia>`). A análise
é escrita em docs/otimizacao.md.

    python -m opinion.modelo.otimizar                       # família vencedora do CC4
    python -m opinion.modelo.otimizar --familia random_forest --trials 60
    python -m opinion.modelo.otimizar --sem-registro --trials 5
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

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.comparar import _f, _nome, _pct, _s, ic_bootstrap, recorte_midia
from opinion.modelo.treinar import (_metricas_planas, _pipeline, avaliar_candidato, avaliar_regra, candidatos,
                                    carregar, config, escolher_limiar, mascara_desconhecidos, medir_teste,
                                    metricas, no_limiar)
from opinion.mundo import RAIZ
from opinion.pipeline.compatibilidade import MIDIAS

DOC = RAIZ / "docs" / "otimizacao.md"
FIGURAS = RAIZ / "docs" / "img" / "cc5"
NOMES_AMOSTRADOR = {"tpe": "Bayesiana (Optuna TPE)", "aleatorio": "Aleatória (Optuna RandomSampler)"}


# ======================================================================== espaço de busca
def sugerir(trial, espaco: dict) -> dict:
    """Traduz o espaço do YAML em sugestões do Optuna. Listas viram escolhas categóricas."""
    p = {}
    for nome, spec in espaco.items():
        if isinstance(spec, list):
            # Optuna só aceita escalares como categoria: listas ([64, 32]) e None viram texto JSON
            p[nome] = json.loads(trial.suggest_categorical(nome, [json.dumps(o) for o in spec]))
        elif spec.get("inteiro"):
            p[nome] = trial.suggest_int(nome, int(spec["min"]), int(spec["max"]), log=bool(spec.get("log")))
        else:
            p[nome] = trial.suggest_float(nome, float(spec["min"]), float(spec["max"]), log=bool(spec.get("log")))
    return {k: v for k, v in p.items() if v is not None}


def _como_sugestao(params: dict, espaco: dict, tipo: str | None = None) -> dict:
    """Parâmetros no formato que `sugerir` produz no Optuna (categorias como texto JSON).

    Hiperparâmetros do espaço que a grade não fixou recebem o valor padrão do estimador, para
    que a tentativa reproduza exatamente a configuração da grade."""
    completos = dict(params)
    if tipo is not None:
        padrao = _pipeline(tipo, {}).named_steps["clf"].get_params()
        padrao["max_depth"] = padrao.get("max_depth", 1) if tipo != "adaboost" else 1
        for k in espaco:
            if k not in completos and k in padrao:
                completos[k] = padrao[k]
    out = {}
    for k, v in completos.items():
        if k not in espaco:
            continue
        if isinstance(espaco[k], list):
            texto = json.dumps(list(v) if isinstance(v, tuple) else v)
            if texto in [json.dumps(o) for o in espaco[k]]:
                out[k] = texto
        else:
            out[k] = v
    return out


def familia_cc4(cfg: dict, midia: str | None) -> str | None:
    """Família vencedora do CC4 no MLflow (alias @melhor). None se ainda não houver comparação registrada."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    c = cfg["comparacao"]
    nome = c["modelo_registrado"] + (f"-{midia}" if midia else "")
    try:
        v = MlflowClient().get_model_version_by_alias(nome, c["alias"])
    except Exception:
        return None
    return MlflowClient().get_run(v.run_id).data.params.get("modelo")


# ======================================================================== otimização
def otimizar(pasta=PASTA_PROCESSADOS, cfg=None, familia: str | None = None, midia: str | None = None,
             trials: int | None = None, registrar=True, doc: Path | None = DOC,
             figuras: Path | None = FIGURAS) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = cfg or config()
    o, beta, chave = cfg["otimizacao"], float(cfg["limiar"]["beta"]), cfg["metrica"]
    trials = int(trials or o["trials"])
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    if midia is not None:
        dm, cols = recorte_midia(dm, meta, midia)
        doc = figuras = None
    familia = familia or (o["familia"] if o["familia"] != "auto" else None) or familia_cc4(cfg, midia) \
        or cfg["comparacao"]["referencia"]
    spec = cfg["comparacao"]["modelos"][familia]
    tipo = spec["tipo"]
    espaco = o["espacos"][tipo]

    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    desc_va, desc_te = mascara_desconhecidos(va).to_numpy(), mascara_desconhecidos(te).to_numpy()
    y_va = va.conectou.to_numpy()
    regra = avaliar_regra(va, te, desc_va, desc_te, beta)

    # ----- referência: a grade fixa do CC4 para a mesma família
    t0 = time.time()
    grade = [avaliar_candidato(n, t, p, tr, va, cols, desc_va, beta) for n, t, p in candidatos(cfg, {familia: spec})]
    grade_melhor = max(grade, key=lambda r: r["validacao"][chave] or 0)
    tempo_grade = time.time() - t0
    print(f"  Grade do CC4 ({len(grade)} configs): validação {chave} {grade_melhor['validacao'][chave]:.4f} "
          f"{grade_melhor['params']}")

    # ----- buscas Optuna com o mesmo orçamento
    def objetivo(trial):
        params = sugerir(trial, espaco)
        inicio = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _pipeline(tipo, params).fit(tr[cols], tr.conectou)
        p = m.predict_proba(va[cols])[:, 1]
        met = metricas(y_va, p, desc_va)
        trial.set_user_attr("params", params)
        trial.set_user_attr("segundos", time.time() - inicio)
        for k in ("pr_auc", "roc_auc_desconhecidos", "brier"):
            trial.set_user_attr(k, met[k])
        return met[chave]

    estudos = {}
    for nome in o["amostradores"]:
        sampler = (optuna.samplers.TPESampler(seed=o["semente"], n_startup_trials=min(10, trials // 3))
                   if nome == "tpe" else optuna.samplers.RandomSampler(seed=o["semente"]))
        estudo = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{familia}-{nome}")
        # ponto de partida: a melhor configuração da grade do CC4 (a busca nunca termina pior que ela)
        estudo.enqueue_trial(_como_sugestao(grade_melhor["params"], espaco, tipo), skip_if_exists=True)
        t0 = time.time()
        estudo.optimize(objetivo, n_trials=trials)
        estudos[nome] = {"estudo": estudo, "segundos": time.time() - t0}
        print(f"  {NOMES_AMOSTRADOR[nome]:<34} {trials} tentativas: melhor {estudo.best_value:.4f} "
              f"em {time.time() - t0:.0f} s  {estudo.best_trial.user_attrs['params']}")

    # melhor configuração entre todas as tentativas (todas medidas na validação)
    vencedor_amostrador = max(estudos, key=lambda k: estudos[k]["estudo"].best_value)
    melhor_trial = estudos[vencedor_amostrador]["estudo"].best_trial
    params = melhor_trial.user_attrs["params"]

    # ----- reajuste, calibração e limiar (validação); teste uma vez
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cru = _pipeline(tipo, params).fit(tr[cols], tr.conectou)
    final, metodo = calibrar(cru, va[cols], va.conectou, o.get("calibracao", "sigmoid"))
    p_va = final.predict_proba(va[cols])[:, 1]
    limiar, _ = escolher_limiar(y_va[desc_va], p_va[desc_va], beta)
    met_va = metricas(y_va, p_va, desc_va)
    met_va["no_limiar_desconhecidos"] = no_limiar(y_va[desc_va], p_va[desc_va], limiar, beta)
    otim = {"modelo": final, "limiar": limiar}
    p_te, met_te = medir_teste(otim, te, cols, desc_te, beta)
    p_te_cru = cru.predict_proba(te[cols])[:, 1]
    p_te_grade, met_te_grade = medir_teste(grade_melhor, te, cols, desc_te, beta)

    y_d = te.conectou.to_numpy()[desc_te]
    ic = ic_bootstrap(y_d, {"otimizado": p_te[desc_te], "grade_cc4": p_te_grade[desc_te],
                            "regra": te.compat_score.to_numpy()[desc_te]},
                      ["grade_cc4", "regra"], cfg["comparacao"]["bootstrap"]["n"], cfg["comparacao"]["bootstrap"]["semente"])

    import optuna.importance as oi
    try:
        importancia = oi.get_param_importances(estudos["tpe" if "tpe" in estudos else vencedor_amostrador]["estudo"])
    except Exception:       # poucas tentativas ou um único parâmetro
        importancia = {}

    r = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "midia": midia or "todas", "familia": familia, "tipo": tipo, "espaco": espaco, "trials": trials,
        "params": params, "amostrador_vencedor": vencedor_amostrador, "calibracao": metodo, "limiar": limiar,
        "grade_cc4": {"params": grade_melhor["params"], "n": len(grade), "segundos": tempo_grade,
                      "validacao": grade_melhor["validacao"], "teste": met_te_grade, "limiar": grade_melhor["limiar"]},
        "buscas": {k: {"melhor": e["estudo"].best_value, "segundos": e["segundos"],
                       "melhor_params": e["estudo"].best_trial.user_attrs["params"],
                       "tentativa_do_melhor": e["estudo"].best_trial.number}
                   for k, e in estudos.items()},
        "validacao": met_va, "teste": met_te,
        "brier_teste_sem_calibrar": float(np.mean((p_te_cru - te.conectou.to_numpy()) ** 2)),
        "regra_simples": regra, "bootstrap_teste_desconhecidos": ic,
        "importancia_hiperparametros": {k: float(v) for k, v in importancia.items()},
        "n_features": len(cols),
        "_estudos": estudos, "_modelo": final, "_cru": cru, "_va": va, "_te": te, "_cols": cols,
        "_p_te": p_te, "_p_te_cru": p_te_cru, "_desc_te": desc_te,
    }
    figs = _figuras(r)
    if figuras is not None:
        Path(figuras).mkdir(parents=True, exist_ok=True)
        for nome, fig in figs.items():
            fig.savefig(Path(figuras) / Path(nome).name, dpi=110, bbox_inches="tight")
    texto = escrever_doc(r, figuras is not None)
    if doc is not None:
        Path(doc).write_text(texto, encoding="utf-8")
    if registrar:
        r.update(_registrar(r, figs, texto, cfg, tr[cols].head(5)))
    import matplotlib.pyplot as plt
    for fig in figs.values():
        plt.close(fig)
    return r


def calibrar(modelo, X_va, y_va, metodo: str):
    """Calibra as probabilidades na validação sem reajustar o modelo (FrozenEstimator)."""
    if metodo in (None, "nenhuma"):
        return modelo, "nenhuma"
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    cal = CalibratedClassifierCV(FrozenEstimator(modelo), method=metodo)
    cal.fit(X_va, y_va)
    return cal, metodo


# ======================================================================== gráficos
def _figuras(r: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    figs = {}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (nome, e) in enumerate(r["_estudos"].items()):
        valores = [t.value for t in e["estudo"].trials]
        ax.plot(range(1, len(valores) + 1), valores, "o", ms=3, alpha=0.35, color=f"C{i}")
        ax.plot(range(1, len(valores) + 1), np.maximum.accumulate(valores), lw=2, color=f"C{i}",
                label=f"{NOMES_AMOSTRADOR[nome]} (melhor até aqui)")
    ax.axhline(r["grade_cc4"]["validacao"]["pr_auc_desconhecidos"], c="gray", ls="--",
               label=f"grade do CC4 ({r['grade_cc4']['n']} configs)")
    ax.axhline(r["regra_simples"]["validacao"]["pr_auc_desconhecidos"], c="tab:orange", ls=":", label="regra simples")
    ax.set(title=f"Histórico da otimização — {_nome(r['familia'])}", xlabel="tentativa",
           ylabel="PR-AUC desconhecidos (validação)")
    ax.legend(fontsize=8)
    figs["otimizacao/historico.png"] = fig

    if r["importancia_hiperparametros"]:
        imp = pd.Series(r["importancia_hiperparametros"]).sort_values()
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(imp) + 1.2))
        imp.plot.barh(ax=ax, color="tab:blue")
        ax.set(title="Importância dos hiperparâmetros (fANOVA, busca TPE)", xlabel="fração da variação explicada")
        figs["otimizacao/importancia_hiperparametros.png"] = fig

    te = r["_te"]
    y = te.conectou.to_numpy()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for rot, p in (("sem calibrar", r["_p_te_cru"]), (f"calibrado ({r['calibracao']})", r["_p_te"])):
        fp_, mp_ = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mp_, fp_, marker="o", label=f"{rot} (Brier {np.mean((p - y) ** 2):.4f})")
    lim = max(0.35, float(np.max(r["_p_te"])) * 1.05)
    ax.plot([0, lim], [0, lim], ls="--", c="gray", label="calibração perfeita")
    ax.set(title="Calibração — teste", xlabel="probabilidade prevista", ylabel="taxa real de conexão")
    ax.legend(fontsize=8)
    figs["otimizacao/calibracao.png"] = fig
    return figs


# ======================================================================== documento
def escrever_doc(r: dict, com_figuras: bool) -> str:
    g, ic, regra = r["grade_cc4"], r["bootstrap_teste_desconhecidos"], r["regra_simples"]
    nl, ng, nr = (r["teste"]["no_limiar_desconhecidos"], g["teste"]["no_limiar_desconhecidos"],
                  regra["teste"]["no_limiar_desconhecidos"])
    esp = "\n".join(f"| `{k}` | {_espaco_txt(v)} |" for k, v in r["espaco"].items())
    buscas = "\n".join(
        f"| {NOMES_AMOSTRADOR[k]} | {r['trials']} | {_f(b['melhor'], 4)} | {b['tentativa_do_melhor'] + 1} | "
        f"{_f(b['segundos'], 0)} s | `{_params_txt(b['melhor_params'])}` |" for k, b in r["buscas"].items())
    imp = r["importancia_hiperparametros"]
    L = [
        "# Otimização de hiperparâmetros — OpiniOn (CC5)", "",
        f"Gerado por `python -m opinion.modelo.otimizar` em {datetime.now():%d/%m/%Y %H:%M}. "
        "Todos os experimentos estão no MLflow, no experimento `opinion-cc5-otimizacao`: um run pai, "
        "um run filho por tentativa e o modelo final registrado como `opinion-cc5-otimizado@otimizado`.", "",
        "> Os encontros e o label são simulados (ver [`fontes_dados.md`](fontes_dados.md)). Os números medem "
        "o quanto o modelo recupera o processo da simulação.", "",
        "## 1. Estratégia adotada", "",
        f"- **Modelo otimizado:** {_nome(r['familia'])}, escolhido na validação do CC4 "
        "([`comparacao_modelos.md`](comparacao_modelos.md)).",
        f"- **Técnica principal:** otimização bayesiana com **Optuna** (amostrador TPE). Para saber se a "
        f"estratégia compensa, rodamos uma **busca aleatória** com o mesmo orçamento ({r['trials']} tentativas) "
        f"e comparamos as duas com a **grade fixa** do CC4 ({g['n']} configurações). As duas buscas começam pela "
        "melhor configuração da grade, então nenhuma termina pior que ela na validação.",
        "- **Função objetivo:** PR-AUC entre desconhecidos na validação temporal (semanas 9–10), a mesma "
        "métrica de escolha do CC3 e do CC4. Cada tentativa treina nas semanas 1–8.",
        "- **O teste (semanas 11–12) não participa da busca.** Ele é medido uma vez, com a configuração final.",
        f"- **Calibração:** depois da busca, as probabilidades são calibradas na validação "
        f"(`{r['calibracao']}`). A transformação é monótona: a ordenação e a PR-AUC não mudam, mas a "
        "probabilidade passa a poder ser exibida como chance na API e no dashboard.",
        "- **Limiar do sinal:** máximo de F0,5 entre desconhecidos na validação, como antes.", "",
        "## 2. Parâmetros avaliados", "", "| Hiperparâmetro | Espaço de busca |", "|---|---|", esp, "",
        "Espaços em `config/modelo.yaml → otimizacao.espacos`.", "",
        "## 3. Resultados da busca (validação)", "",
        "| Estratégia | Tentativas | Melhor PR-AUC desc. | Tentativa do melhor | Tempo | Melhor configuração |",
        "|---|---|---|---|---|---|", buscas,
        f"| Grade fixa do CC4 | {g['n']} | {_f(g['validacao']['pr_auc_desconhecidos'], 4)} | — | "
        f"{_f(g['segundos'], 0)} s | `{_params_txt(g['params'])}` |",
        f"| Regra simples | — | {_f(regra['validacao']['pr_auc_desconhecidos'], 4)} | — | — | — |", "",
        f"**Configuração final** (melhor de todas as tentativas, encontrada pela busca "
        f"{NOMES_AMOSTRADOR[r['amostrador_vencedor']].split(' (')[0].lower()}): `{_params_txt(r['params'])}`.", "",
    ]
    if com_figuras:
        L += ["![Histórico da otimização](img/cc5/historico.png)", ""]
    if imp:
        L += ["**Importância dos hiperparâmetros** (fANOVA sobre as tentativas TPE): "
              + ", ".join(f"`{k}` {_pct(v)}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])) + ".", ""]
        if com_figuras:
            L += ["![Importância dos hiperparâmetros](img/cc5/importancia_hiperparametros.png)", ""]
    L += [
        "## 4. Teste (medido uma vez)", "",
        "| Modelo | PR-AUC desc. (IC 95%) | PR-AUC geral | ROC-AUC desc. | Brier | Precisão no limiar | Recall | Sinal em |",
        "|---|---|---|---|---|---|---|---|",
        f"| {_nome(r['familia'])} otimizado e calibrado | {_f(ic['otimizado']['pr_auc'])} ({_f(ic['otimizado']['ic'][0])}–"
        f"{_f(ic['otimizado']['ic'][1])}) | {_f(r['teste']['pr_auc'])} | {_f(r['teste']['roc_auc_desconhecidos'])} | "
        f"{_f(r['teste']['brier'], 4)} | {_pct(nl['precisao'])} | {_pct(nl['recall'])} | {_pct(nl['taxa_sinal'])} |",
        f"| {_nome(r['familia'])} da grade do CC4 | {_f(ic['grade_cc4']['pr_auc'])} ({_f(ic['grade_cc4']['ic'][0])}–"
        f"{_f(ic['grade_cc4']['ic'][1])}) | {_f(g['teste']['pr_auc'])} | {_f(g['teste']['roc_auc_desconhecidos'])} | "
        f"{_f(g['teste']['brier'], 4)} | {_pct(ng['precisao'])} | {_pct(ng['recall'])} | {_pct(ng['taxa_sinal'])} |",
        f"| Regra simples | {_f(ic['regra']['pr_auc'])} ({_f(ic['regra']['ic'][0])}–{_f(ic['regra']['ic'][1])}) | "
        f"{_f(regra['teste']['pr_auc'])} | {_f(regra['teste']['roc_auc_desconhecidos'])} | não se aplica | "
        f"{_pct(nr['precisao'])} | {_pct(nr['recall'])} | {_pct(nr['taxa_sinal'])} |",
        f"| *acaso* | *{_f(r['teste']['prevalencia_desconhecidos'])}* | *{_f(r['teste']['prevalencia'])}* | *0,500* | | | | |",
        "",
        f"Diferença de PR-AUC entre desconhecidos no teste (bootstrap pareado): contra a grade do CC4 "
        f"{_s(ic['otimizado']['delta_grade_cc4'])} (IC 95% {_s(ic['otimizado']['delta_grade_cc4_ic'][0])} a "
        f"{_s(ic['otimizado']['delta_grade_cc4_ic'][1])}); contra a regra {_s(ic['otimizado']['delta_regra'])} "
        f"(IC 95% {_s(ic['otimizado']['delta_regra_ic'][0])} a {_s(ic['otimizado']['delta_regra_ic'][1])}).", "",
    ]
    if com_figuras:
        L += ["![Calibração](img/cc5/calibracao.png)", ""]
    L += ["## 5. Análise", ""] + [f"- {t}" for t in analise(r)] + [""]
    L += ["## 6. Limitações", "",
          "- **Seleção na validação.** Com dezenas de tentativas avaliadas no mesmo conjunto, o melhor valor de "
          "validação é otimista; a diferença para o teste mede esse viés. A estabilidade entre partições é "
          "avaliada no CC6 ([`validacao_cruzada.md`](validacao_cruzada.md)).",
          "- **Calibração na validação.** O mesmo conjunto escolhe hiperparâmetros, ajusta a sigmoide e o limiar. "
          "Como a sigmoide tem dois parâmetros, o risco de sobreajuste é pequeno, e o Brier do teste confirma.",
          "- **Label simulado.** Os hiperparâmetros ótimos dependem das premissas da simulação e devem ser "
          "revistos com os dados dos testes do Design (semana 16).", ""]
    return "\n".join(L)


def analise(r: dict) -> list[str]:
    g, ic, b = r["grade_cc4"], r["bootstrap_teste_desconhecidos"], r["buscas"]
    out = []
    if "tpe" in b and "aleatorio" in b:
        d = b["tpe"]["melhor"] - b["aleatorio"]["melhor"]
        out.append(f"**Bayesiana x aleatória:** com o mesmo orçamento, o TPE chegou a {_f(b['tpe']['melhor'], 4)} e a "
                   f"busca aleatória a {_f(b['aleatorio']['melhor'], 4)} ({_s(d)}). "
                   + ("A busca guiada encontrou configurações melhores." if d > 0.002 else
                      "A busca aleatória encontrou uma configuração melhor: com este orçamento, o TPE ainda "
                      "explora o espaço e a vantagem dele não aparece." if d < -0.002 else
                      "A diferença é pequena: neste espaço, a busca aleatória já cobre bem as regiões boas."))
    dg = max(x["melhor"] for x in b.values()) - g["validacao"]["pr_auc_desconhecidos"]
    out.append(f"**Contra a grade do CC4 (validação):** {_s(dg)} de PR-AUC (as duas buscas partem da melhor "
               "configuração da grade). "
               + ("A otimização encontrou uma região melhor do que a grade cobria." if dg > 0.002 else
                  "A grade do CC4 já estava perto do melhor que a família alcança nesses dados."))
    lo, hi = ic["otimizado"]["delta_grade_cc4_ic"]
    out.append(f"**No teste,** o ganho sobre a grade é {_s(ic['otimizado']['delta_grade_cc4'])} "
               f"(IC 95% {_s(lo)} a {_s(hi)}). "
               + ("O ganho se confirma fora da validação." if lo > 0 else
                  "O intervalo inclui zero: o ganho da validação não se confirma de forma consistente no teste, "
                  "o que é esperado quando muitas tentativas competem pelo mesmo conjunto de validação."))
    lo, hi = ic["otimizado"]["delta_regra_ic"]
    out.append(f"**Contra a regra simples (teste):** {_s(ic['otimizado']['delta_regra'])} (IC 95% {_s(lo)} a {_s(hi)})"
               + (", vantagem consistente." if lo > 0 else ", vantagem não consistente."))
    out.append(f"**Calibração:** o Brier do teste foi de {_f(r['brier_teste_sem_calibrar'], 4)} sem calibrar para "
               f"{_f(r['teste']['brier'], 4)} depois da sigmoide. É essa versão calibrada que a API serve como "
               "\"chance de interação em 48 h\".")
    imp = r["importancia_hiperparametros"]
    if imp:
        k = max(imp, key=imp.get)
        out.append(f"**O que mais pesa:** `{k}` explica {_pct(imp[k])} da variação da métrica entre as tentativas. "
                   "Os demais hiperparâmetros têm efeito menor, o que indica uma superfície de busca relativamente plana.")
    return out


def _espaco_txt(v) -> str:
    if isinstance(v, list):
        return ", ".join("`None`" if o is None else f"`{o}`" for o in v)
    escala = " (escala log)" if v.get("log") else ""
    tipo = "inteiro" if v.get("inteiro") else "real"
    return f"{tipo} de {v['min']} a {v['max']}{escala}"


def _params_txt(p: dict) -> str:
    return ", ".join(f"{k}={round(v, 5) if isinstance(v, float) else v}" for k, v in p.items())


# ======================================================================== MLflow
def _registrar(r: dict, figs: dict, texto: str, cfg: dict, exemplo) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    o = cfg["otimizacao"]
    sufixo = "" if r["midia"] == "todas" else f"-{r['midia']}"
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(o["experimento"] + sufixo)
    nome_modelo = o["modelo_registrado"] + sufixo
    regra = r["regra_simples"]

    with mlflow.start_run(run_name=f"otimizacao-{r['familia']}-{datetime.now():%Y%m%d-%H%M}") as pai:
        mlflow.log_params({"algoritmo": r["familia"], "modelo": r["familia"], "tipo": r["tipo"],
                           "trials_por_amostrador": r["trials"], "amostradores": ",".join(r["buscas"]),
                           "amostrador_vencedor": r["amostrador_vencedor"], "calibracao": r["calibracao"],
                           "limiar": r["limiar"], "limiar_regra": regra["limiar"],
                           "beta_limiar": cfg["limiar"]["beta"], "metrica_escolha": cfg["metrica"],
                           "params_json": json.dumps(r["params"]), "midia": r["midia"],
                           **{f"hp_{k}": v for k, v in r["params"].items()}})
        mlflow.log_metrics(_metricas_planas("validacao", r["validacao"]))
        mlflow.log_metrics(_metricas_planas("teste", r["teste"]))
        mlflow.log_metrics(_metricas_planas("grade_cc4_teste", r["grade_cc4"]["teste"]))
        mlflow.log_metrics({f"melhor_{k}": b["melhor"] for k, b in r["buscas"].items()}
                           | {"grade_cc4_validacao": r["grade_cc4"]["validacao"]["pr_auc_desconhecidos"],
                              "brier_teste_sem_calibrar": r["brier_teste_sem_calibrar"],
                              "delta_grade_cc4_teste": r["bootstrap_teste_desconhecidos"]["otimizado"]["delta_grade_cc4"],
                              "delta_regra_teste": r["bootstrap_teste_desconhecidos"]["otimizado"]["delta_regra"]})
        mlflow.log_dict({k: v for k, v in r.items() if not k.startswith("_")}, "resumo.json")
        mlflow.log_text(texto, "otimizacao.md")
        for caminho, fig in figs.items():
            mlflow.log_figure(fig, caminho)
        for nome, e in r["_estudos"].items():
            df = e["estudo"].trials_dataframe()
            mlflow.log_text(df.to_csv(index=False), f"tentativas_{nome}.csv")
            for t in e["estudo"].trials:
                with mlflow.start_run(run_name=f"{nome}-{t.number:03d}", nested=True):
                    mlflow.set_tag("amostrador", nome)
                    mlflow.log_params({"amostrador": nome, "tentativa": t.number,
                                       **{f"hp_{k}": v for k, v in t.user_attrs["params"].items()}})
                    mlflow.log_metrics({"validacao_pr_auc_desconhecidos": t.value,
                                        "validacao_pr_auc": t.user_attrs["pr_auc"],
                                        "validacao_roc_auc_desconhecidos": t.user_attrs["roc_auc_desconhecidos"],
                                        "validacao_brier": t.user_attrs["brier"],
                                        "segundos": t.user_attrs["segundos"]})
        modelo = r["_modelo"]
        info = mlflow.sklearn.log_model(
            sk_model=modelo, name="modelo", input_example=exemplo,
            signature=infer_signature(exemplo, modelo.predict_proba(exemplo)[:, 1]),
            registered_model_name=nome_modelo, serialization_format="cloudpickle")
        versao = str(info.registered_model_version)
        MlflowClient().set_registered_model_alias(nome_modelo, o["alias"], versao)
    return {"mlflow_run_pai": pai.info.run_id, "mlflow_modelo": f"{nome_modelo}@{o['alias']}", "mlflow_versao": versao}


def main():
    ap = argparse.ArgumentParser(description="CC5: otimização de hiperparâmetros com Optuna, registrada no MLflow")
    ap.add_argument("--familia", help="família do config (padrão: vencedora do CC4)")
    ap.add_argument("--midia", choices=MIDIAS)
    ap.add_argument("--trials", type=int)
    ap.add_argument("--sem-registro", action="store_true", help="não grava no MLflow nem reescreve o doc")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    a = ap.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    print("Otimizando (treino → objetivo na validação → teste uma vez):")
    sem_doc = a.sem_registro or a.midia
    r = otimizar(a.pasta, familia=a.familia, midia=a.midia, trials=a.trials, registrar=not a.sem_registro,
                 doc=None if sem_doc else DOC, figuras=None if sem_doc else FIGURAS)
    ic = r["bootstrap_teste_desconhecidos"]
    print(f"\nFinal: {_nome(r['familia'])} {r['params']} (calibração {r['calibracao']}, limiar {r['limiar']:.4f})")
    print(f"Teste, PR-AUC desconhecidos: otimizado {ic['otimizado']['pr_auc']:.3f} | grade CC4 "
          f"{ic['grade_cc4']['pr_auc']:.3f} | regra {ic['regra']['pr_auc']:.3f}")
    for t in analise(r):
        print(" -", t.replace("**", ""))
    if r.get("mlflow_modelo"):
        print(f"\nMLflow: {r['mlflow_modelo']} v{r['mlflow_versao']}  run {r['mlflow_run_pai']}")
    if not sem_doc:
        print(f"Documento: {DOC}")


if __name__ == "__main__":
    main()
