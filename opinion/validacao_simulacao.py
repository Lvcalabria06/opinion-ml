"""
Validação da simulação (CC2 — seção 4.3).

Gera docs/validacao_simulacao.md com:
  1. Premissas      cada parâmetro do simulador, valor e justificativa
  2. Sanidade       checagens automáticas com faixas plausíveis (passa / atenção / falha)
  3. Sensibilidade  regenera o dataset em 3 cenários (pessimista, base, otimista), roda o MESMO
                    pipeline e os MESMOS modelos, e verifica se as conclusões se mantêm
  4. Plano de comparação com os dados reais dos testes do Design (semana 16)

    python -m opinion.validacao_simulacao            # ~2 min
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from opinion.gerar_bootstrap import Simulador
from opinion.mundo import RAIZ, carregar_config
from opinion.pipeline.executar import rodar_tudo
from opinion.pipeline.tempo import ultimo_valor

PASTA = RAIZ / "data" / "validacao"
DOC = RAIZ / "docs" / "validacao_simulacao.md"
CENARIOS = RAIZ / "config" / "cenarios_sensibilidade.yaml"
ORDEM_CONTEXTOS = ["convencao_games", "cinema", "livraria", "campus"]

# --------------------------------------------------------------------------- premissas
PREMISSAS = [
    ("simulacao.n_usuarios", "Tamanho de uma comunidade ativa num campus/cidade média; gera ~55 mil encontros, o bastante "
     "para split temporal + validação cruzada com positivos suficientes em cada parte.", "Escala do piloto real."),
    ("simulacao.semanas", "Exigido pelo split temporal da seção 4.5 (treino 1–8, validação 9–10, teste 11–12).", "—"),
    ("simulacao.semanas_historico_previo", "Usuários já têm histórico de avaliações quando a pulseira chega; sem isso "
     "a compatibilidade seria vazia nas primeiras semanas.", "Histórico médio dos usuários do app."),
    ("personas.*.peso", "Mistura provisória de perfis de público cultural/universitário.", "Trocar pelas personas do Design."),
    ("personas.*.visitas_semana", "Frequência plausível: estudante vai ao campus quase todo dia; cinéfilo ~1–2 sessões "
     "por semana; leitor ~1 livraria por semana.", "Pergunta da coleta complementar (seção 3.1)."),
    ("personas.*.abertura_social", "Variável OCULTA que o modelo não vê (seção 4.3), para o label não ser uma regra "
     "redescobrível. Gamers sociais mais abertos, leitores reservados menos.", "Interesse em conhecer pessoas (coleta "
     "complementar) e taxa de toque nos testes."),
    ("locais.*.p_encontro", "Chance de dois desconhecidos no mesmo lugar e turno ficarem a menos de ~2 m por tempo "
     "suficiente: alta numa livraria pequena, baixa num campus ou pavilhão grandes.", "Encontros por hora nos testes reais."),
    ("locais.*.duracao_mediana_s", "Fila/saguão de cinema ~5 min, corredor de livraria ~3 min, campus ~2 min, estande de "
     "convenção ~6 min.", "Distribuição de duração dos encontros reais."),
    ("locais.convencao_games.semanas_com_evento", "Eventos são esporádicos (1 a cada ~4 semanas); cria picos que testam "
     "a estabilidade do modelo no tempo.", "—"),
    ("amigos.fracao_em_grupos / p_ir_junto", "Metade das pessoas sai com amigos às vezes; gera encontros longos entre "
     "quem já se segue (ruído realista no label).", "Proporção de encontros entre quem já se segue nos testes."),
    ("modos_descoberta.inicial", "Maioria aberta ou discreta; ~10% pausada (preocupação com privacidade: 41% na pesquisa "
     "anterior citam uso de dados).", "Modos escolhidos nos testes."),
    ("conexao.intercepto", "Calibra a taxa base: ~5–8% dos encontros entre desconhecidos viram conexão — raro, como "
     "esperado para abordagens entre estranhos.", "Taxa de conversão real (indicador da seção 3.2)."),
    ("conexao.compat", "Gosto parecido aumenta a chance de conexão (hipótese do projeto), mas não determina sozinho.",
     "Comparar taxa por faixa de compatibilidade nos dados reais."),
    ("conexao.abertura", "Pessoas abertas conectam mais; o produto dos dois exige que ambos estejam abertos.", "—"),
    ("conexao.duracao", "Encontros longos indicam conversa ou atividade em comum.", "Taxa por duração nos dados reais."),
    ("conexao.contexto", "Convenção é social por natureza; campus é casual (seção 3.3).", "Taxa por contexto nos testes."),
    ("conexao.repeticao", "Encontrar a mesma pessoa várias vezes aumenta a familiaridade.", "—"),
    ("conexao.discreto", "Modo discreto reduz levemente a chance (menos exposição).", "—"),
    ("conexao.ruido", "Fatores não observados (humor, pressa, timidez). Garante que nenhum modelo acerte tudo.", "—"),
    ("conexao.atraso_medio_h", "Parte das pessoas só abre o app no dia seguinte; ~3–4% das conexões caem depois de 48 h "
     "(label 0 mesmo assim).", "Tempo até a 1ª interação nos testes."),
    ("interacoes_de_fundo.*", "Amigos e o feed geram interações independentes de encontros — o label precisa conviver "
     "com esse ruído.", "Volume de interações por usuário no app real."),
    ("toque_interesse.*", "Toque mais provável quando há chance de conexão e o usuário é aberto. Dado secundário.",
     "Taxa de toque nos testes do protótipo."),
]


def _valor(cfg: dict, caminho: str) -> str:
    partes = caminho.split(".")
    if "*" in partes or "/" in caminho:
        if partes[0] == "personas":
            chave = partes[-1]
            return "; ".join(f"{p}: {v.get(chave)}" for p, v in cfg["personas"].items())
        if partes[0] == "locais":
            chave = partes[-1]
            return "; ".join(f"{l}: {v.get(chave)}" for l, v in cfg["locais"].items() if chave in v)
        if partes[0] == "amigos":
            return f"{cfg['amigos']['fracao_em_grupos']} / {cfg['amigos']['p_ir_junto']}"
        bloco = cfg[partes[0]]
        return "; ".join(f"{k}: {v}" for k, v in bloco.items())
    v = cfg
    for p in partes:
        v = v[p]
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    return str(v)


# ---------------------------------------------------------------------- cenários
def config_do_cenario(nome: str, mudancas: dict) -> dict:
    cfg = copy.deepcopy(carregar_config())
    for bloco, valores in (mudancas or {}).items():
        if bloco == "descricao":
            continue
        for k, v in valores.items():
            if k == "contexto_mult":
                cfg["conexao"]["contexto"] = {c: round(x * v, 3) for c, x in cfg["conexao"]["contexto"].items()}
            else:
                cfg[bloco][k] = v
    return cfg


def gerar_cenario(nome: str, cfg: dict) -> Path:
    pasta = PASTA / nome
    boot = pasta / "bootstrap"
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "simulacao.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    with contextlib.redirect_stdout(io.StringIO()):
        s = Simulador(cfg)
        s.gerar_obras(); s.gerar_usuarios(); s.gerar_rede(); s.gerar_avaliacoes(); s.gerar_modos()
        s.simular()
        s.salvar(boot)
        rodar_tudo("arquivos", staging=pasta / "staging", saida=pasta / "processed", pasta_bootstrap=boot,
                   referencia=boot / "rotulos_referencia.csv", verbose=False)
    return pasta


# ------------------------------------------------------------------------ modelos
def modelos():
    return {
        "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Árvore de decisão": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                                n_jobs=-1, random_state=0),
    }


def avaliar(pasta: Path) -> dict:
    dm = pd.read_parquet(pasta / "processed" / "dataset_modelo.parquet")
    cols = json.loads((pasta / "processed" / "metadados.json").read_text())["colunas_features"]
    tr, te = dm[dm.split == "treino"], dm[dm.split == "teste"]
    desc = te.algum_segue == 0
    r = {"taxa_conexao": float(dm.conectou.mean()), "taxa_desconhecidos": float(dm[dm.algum_segue == 0].conectou.mean()),
         "encontros": int(len(dm)), "pr_auc": {}, "pr_auc_desconhecidos": {}, "roc_auc": {}}
    pontuacoes = {"Regra simples (compatibilidade)": te.compat_score.to_numpy()}
    importancias = None
    for nome, m in modelos().items():
        m.fit(tr[cols], tr.conectou)
        pontuacoes[nome] = m.predict_proba(te[cols])[:, 1]
        if nome == "Random Forest":
            importancias = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    for nome, p in pontuacoes.items():
        r["pr_auc"][nome] = float(average_precision_score(te.conectou, p))
        r["pr_auc_desconhecidos"][nome] = float(average_precision_score(te.conectou[desc], p[desc.to_numpy()]))
        r["roc_auc"][nome] = float(roc_auc_score(te.conectou, p))
    r["prevalencia_teste"] = float(te.conectou.mean())
    r["prevalencia_teste_desconhecidos"] = float(te.conectou[desc].mean())
    ctx = dm[dm.algum_segue == 0].groupby("local_original").conectou.mean()
    r["contexto_desconhecidos"] = {k: float(v) for k, v in ctx.reindex(ORDEM_CONTEXTOS).items()}
    r["ordem_contextos"] = list(ctx.sort_values(ascending=False).index)
    r["top_features"] = list(importancias.index[:5])
    return r


# -------------------------------------------------------------------- sanidade
def checagens(pasta: Path, cfg: dict) -> list[dict]:
    boot, proc, stg = pasta / "bootstrap", pasta / "processed", pasta / "staging"
    f = pd.read_parquet(proc / "features_encontros.parquet")
    f["duracao_min"] = f.duracao_s / 60
    oc = pd.read_csv(boot / "verdade_simulacao_NAO_USAR_COMO_FEATURE" / "encontros_ocultos.csv")
    st = pd.read_parquet(stg / "status.parquet")
    it = pd.read_parquet(stg / "interesses.parquet")
    n_usuarios = cfg["simulacao"]["n_usuarios"]
    C = []

    def chk(nome, valor, ok, faixa, porque, atencao=False):
        C.append({"checagem": nome, "valor": valor, "faixa": faixa,
                  "status": "OK" if ok else ("ATENÇÃO" if atencao else "FALHA"), "por_que": porque})

    t = f.conectou.mean()
    chk("Taxa geral de conexão", f"{t:.1%}", 0.05 <= t <= 0.25, "5%–25%",
        "Problema desbalanceado, mas com positivos suficientes para treinar (seção 4.3).")
    ta, td = f[f.algum_segue == 1].conectou.mean(), f[f.algum_segue == 0].conectou.mean()
    chk("Desconhecidos conectam menos que quem já se segue", f"{td:.1%} x {ta:.1%}", td < ta, "desconhecidos < amigos",
        "Amigos interagem no app de qualquer jeito.")
    chk("Desconhecidos: taxa de conexão", f"{td:.1%}", 0.02 <= td <= 0.15, "2%–15%",
        "Abordagem entre estranhos é rara; acima de 15% seria otimista demais.")
    med = f.groupby("local").duracao_min.median()
    chk("Duração mediana por local", ", ".join(f"{k} {v:.1f} min" for k, v in med.items()),
        bool(med.between(1, 30).all()), "1–30 min em todos",
        "Encontros de segundos seriam passagens; de horas, só entre amigos.")
    chk("Nenhum encontro menor que 30 s", f"mínimo {f.duracao_s.min()} s", f.duracao_s.min() >= 30, ">= 30 s",
        "A pulseira descarta contatos curtos (gente só passando).")
    r = f.rssi_medio.mean()
    chk("RSSI médio", f"{r:.1f} dBm", -75 <= r <= -50, "-75 a -50 dBm", "Equivale a ~0,5–3 m de distância.")
    por_sem = 2 * len(f) / n_usuarios / cfg["simulacao"]["semanas"]
    chk("Encontros por usuário por semana", f"{por_sem:.1f}", 0.5 <= por_sem <= 20, "0,5–20",
        "Plausível para quem frequenta campus e eventos culturais.")
    contagem = pd.concat([f.usuario_a, f.usuario_b]).value_counts()
    top = contagem.head(max(1, len(contagem) // 10)).sum() / contagem.sum()
    chk("Concentração: 10% mais ativos", f"{top:.1%} dos encontros", top <= 0.40, "<= 40%",
        "Evita que poucos usuários dominem o dataset (e o modelo).")
    campus_util = (f[f.local == "campus"].dia_semana < 5).mean()
    chk("Campus só em dias úteis", f"{campus_util:.1%}", campus_util == 1, "100%", "Premissa de funcionamento.")
    conv = f[f.local == "convencao_games"]
    conv_ok = conv.semana.isin(cfg["locais"]["convencao_games"]["semanas_com_evento"]).all() and (conv.dia_semana >= 5).all()
    chk("Convenção só nos fins de semana de evento", "sim" if conv_ok else "não", bool(conv_ok),
        f"semanas {cfg['locais']['convencao_games']['semanas_com_evento']}, sáb/dom", "Premissa de calendário.")
    horas = f.hora.between(8, 23).mean()
    chk("Encontros entre 8h e 23h59", f"{horas:.1%}", horas == 1, "100%", "Horário de funcionamento dos locais.")
    modo_a = ultimo_valor(st.usuario_id, st.ts, st.modo, f.usuario_a, f.inicio, "aberto")
    modo_b = ultimo_valor(st.usuario_id, st.ts, st.modo, f.usuario_b, f.inicio, "aberto")
    pausados = int(((modo_a == "pausado") | (modo_b == "pausado")).sum())
    chk("Nenhum encontro com pulseira pausada", f"{pausados} encontros", pausados == 0, "0",
        "Modo pausado = pulseira não anuncia (seção 3.3).")
    sem = f.groupby("semana").conectou.mean()
    chk("Estabilidade semanal da taxa de conexão", f"desvio {sem.std() * 100:.1f} p.p.", sem.std() <= 0.015,
        "<= 1,5 p.p.", "Sem mudança brusca entre treino/validação/teste — o split temporal mede o modelo, não a base.")
    rep = 1 - f.primeiro_encontro_par.mean()
    chk("Encontros repetidos do mesmo par", f"{rep:.1%}", 0.10 <= rep <= 0.60, "10%–60%",
        "Necessário para a feature de encontros anteriores e o k-fold agrupado por par.")
    causadas = oc[oc.conexao_causada_pelo_encontro == 1].merge(f[["encontro_id", "conectou"]], on="encontro_id")
    fora = 1 - causadas.conectou.mean()
    chk("Conexões causadas que só aparecem depois de 48 h", f"{fora:.1%}", 0.01 <= fora <= 0.15, "1%–15%",
        "A janela de 48 h perde algumas conexões tardias, como na vida real.")
    tocou = f.encontro_id.isin(it.encontro_id)
    tc, tn = tocou[f.conectou == 1].mean(), tocou[f.conectou == 0].mean()
    chk("Toque mais frequente quando há conexão", f"{tc:.1%} x {tn:.1%}", tc > tn, "conectou > não conectou",
        "O toque deve carregar algum sinal de interesse.")
    d = f[f.algum_segue == 0]
    auc_rssi = roc_auc_score(d.conectou, d.rssi_medio)
    chk("Proximidade (RSSI) influencia conexão entre desconhecidos", f"AUC {auc_rssi:.3f}", auc_rssi >= 0.52,
        "AUC >= 0,52", "Esperado no mundo real (quem conversa fica mais perto). Na simulação o RSSI não entra no "
        "processo do label — premissa a revisar.", atencao=True)
    return C


# ------------------------------------------------------------------- documento
def escrever_doc(cfg, sanidade, resultados, cenarios, segundos):
    pct = lambda x: f"{100 * x:.1f}%"
    L = ["# Validação da simulação — OpiniOn (CC2)", "",
         f"Gerado por `python -m opinion.validacao_simulacao` em {datetime.now():%d/%m/%Y %H:%M} "
         f"({segundos:.0f} s). Seção 4.3 do documento do projeto.", "",
         "> O desempenho de qualquer modelo treinado com estes dados mede a **simulação**, não o comportamento real "
         "dos usuários do OpiniOn. Este documento mostra o que foi assumido, se os dados são plausíveis e se as "
         "conclusões dependem demais das premissas.", ""]

    n_ok = sum(c["status"] == "OK" for c in sanidade)
    n_at = sum(c["status"] == "ATENÇÃO" for c in sanidade)
    n_fa = sum(c["status"] == "FALHA" for c in sanidade)
    concl = resultados["_conclusoes"]
    L += ["## Resumo", "",
          f"- **Checagens de sanidade:** {n_ok} OK, {n_at} atenção, {n_fa} falha(s).",
          f"- **Sensibilidade:** {sum(c['mantem'] for c in concl)} de {len(concl)} conclusões se mantêm nos 3 cenários.",
          "- **Principal premissa a revisar:** a proximidade (RSSI) não influencia a conexão entre desconhecidos na "
          "simulação. Sugestão: incluir o RSSI no processo do label na próxima versão do simulador, ou confirmar com os "
          "dados reais da semana 16 se ele importa.", ""]

    L += ["## 1. Premissas", "",
          "Todos os valores estão em `config/simulacao.yaml`. Nenhum vem da pesquisa de monetização (n = 29), que trata "
          "de outro comportamento.", "",
          "| Parâmetro | Valor atual | Justificativa | Como validar |", "|---|---|---|---|"]
    for caminho, porque, validar in PREMISSAS:
        L.append(f"| `{caminho}` | {_valor(cfg, caminho)} | {porque} | {validar} |")

    L += ["", "### Como o label é gerado", "",
          "Para cada encontro, a chance de virar conexão é", "",
          "```", "p = sigmoide( intercepto + compat·compatibilidade_latente + abertura·(abertura_A × abertura_B)",
          "               + duracao·log(duração/120 s) + efeito_do_local + repeticao·min(encontros anteriores, 5)",
          "               + discreto·[alguém em modo discreto] + ruído )", "```", "",
          "Se sorteado, um dos dois interage com o outro no app depois de um atraso aleatório. O label do pipeline "
          "(interação em até 48 h) é calculado depois, a partir das interações — que também incluem interações entre "
          "amigos e do feed. A *compatibilidade latente* e a *abertura social* são ocultas: o modelo só vê "
          "aproximações (similaridade de notas, atividade, conexões).", ""]

    L += ["## 2. Checagens de sanidade (cenário base)", "",
          "| Status | Checagem | Valor | Faixa plausível | Por quê |", "|---|---|---|---|---|"]
    icone = {"OK": "✅", "ATENÇÃO": "⚠️", "FALHA": "❌"}
    for c in sanidade:
        L.append(f"| {icone[c['status']]} {c['status']} | {c['checagem']} | {c['valor']} | {c['faixa']} | {c['por_que']} |")

    L += ["", "## 3. Análise de sensibilidade", "",
          "O dataset foi gerado de novo em três cenários (`config/cenarios_sensibilidade.yaml`). Em cada um rodamos "
          "**o mesmo pipeline** e **os mesmos modelos simples**, treinando nas semanas 1–8 e medindo nas semanas 11–12. "
          "Os modelos aqui não são otimizados — servem só para ver se as conclusões mudam.", "",
          "| Cenário | O que muda | Encontros | Taxa de conexão | Entre desconhecidos |", "|---|---|---|---|---|"]
    for nome in ("pessimista", "base", "otimista"):
        r = resultados[nome]
        L.append(f"| **{nome}** | {cenarios[nome]['descricao']} | {r['encontros']:,} | {pct(r['taxa_conexao'])} | "
                 f"{pct(r['taxa_desconhecidos'])} |")

    nomes_mod = list(resultados["base"]["pr_auc"])
    L += ["", "### PR-AUC no teste (semanas 11–12) — todos os encontros", "",
          "| Modelo | " + " | ".join(("pessimista", "base", "otimista")) + " |", "|---|---|---|---|"]
    for m in nomes_mod:
        L.append(f"| {m} | " + " | ".join(f"{resultados[c]['pr_auc'][m]:.3f}" for c in ("pessimista", "base", "otimista")) + " |")
    L.append("| *prevalência (acaso)* | " + " | ".join(f"*{resultados[c]['prevalencia_teste']:.3f}*"
                                                    for c in ("pessimista", "base", "otimista")) + " |")
    L += ["", "### PR-AUC no teste — só desconhecidos (o público real da pulseira)", "",
          "| Modelo | pessimista | base | otimista |", "|---|---|---|---|"]
    for m in nomes_mod:
        L.append(f"| {m} | " + " | ".join(f"{resultados[c]['pr_auc_desconhecidos'][m]:.3f}"
                                         for c in ("pessimista", "base", "otimista")) + " |")
    L.append("| *prevalência (acaso)* | " + " | ".join(f"*{resultados[c]['prevalencia_teste_desconhecidos']:.3f}*"
                                                    for c in ("pessimista", "base", "otimista")) + " |")
    L += ["", "### Taxa de conexão entre desconhecidos por contexto", "",
          "| Contexto | pessimista | base | otimista |", "|---|---|---|---|"]
    for ctx in ORDEM_CONTEXTOS:
        L.append(f"| {ctx} | " + " | ".join(pct(resultados[c]["contexto_desconhecidos"][ctx])
                                           for c in ("pessimista", "base", "otimista")) + " |")
    L += ["", "### As conclusões se mantêm?", "", "| Conclusão | Mantém nos 3 cenários? | Detalhe |", "|---|---|---|"]
    for c in concl:
        L.append(f"| {c['conclusao']} | {'✅ sim' if c['mantem'] else '❌ não'} | {c['detalhe']} |")
    L += ["", "**Leitura.** Os números absolutos mudam bastante entre cenários (é esperado: as premissas mudam a "
          "dificuldade do problema). O que importa é que a **comparação** se sustente — se uma conclusão só vale no "
          "cenário base, ela não deve ser apresentada como resultado do projeto.", "",
          "Repare que o cenário pessimista tem PR-AUC *geral* mais alta que o base: nele os amigos interagem mais, e o "
          "modelo acerta reconhecendo amigos. Por isso a métrica só entre desconhecidos é a que conta para a pulseira.", ""]
    falhas = [c for c in concl if not c["mantem"] and c.get("interpretacao")]
    if falhas:
        L += ["**O que não se manteve e o que isso significa:**", ""]
        L += [f"- *{c['conclusao']}.* {c['interpretacao']}" for c in falhas]
        L.append("")

    L += ["## 4. Comparação com dados reais (semana 16)", "",
          "Quando o protótipo do Design for testado com usuários, os encontros reais entram pelo mesmo MQTT e pelo "
          "mesmo pipeline. Comparar com a simulação:", "",
          "| O que comparar | Como | Se divergir |", "|---|---|---|",
          "| Duração dos encontros por contexto | mediana e teste de Kolmogorov–Smirnov | ajustar `duracao_mediana_s` |",
          "| RSSI médio | média e histograma | ajustar o modelo de rádio da pulseira virtual |",
          "| Encontros por pessoa por hora | contagem | ajustar `p_encontro` |",
          "| Taxa de conexão (geral e por contexto) | proporção com intervalo de confiança | ajustar `intercepto` e `contexto` |",
          "| Taxa de toque e de toque mútuo | proporção | ajustar `toque_interesse` |",
          "| Fração de encontros entre quem já se segue | proporção | ajustar `amigos` |", "",
          "Com poucos participantes nos testes, a comparação será qualitativa (mesma ordem de grandeza? mesma direção "
          "dos efeitos?). Ainda assim, ela é a única evidência externa de que a simulação é razoável.", ""]

    L += ["## 5. Limitações conhecidas", "",
          "- Personas, frequências e efeitos são premissas — as personas reais do Design ainda não foram incorporadas.",
          "- O RSSI não influencia a conexão entre desconhecidos (ver checagem com ⚠️).",
          "- Dia da semana e horário não têm efeito próprio no label (só via local).",
          "- A rede de amigos é fixa: não surgem novos grupos ao longo das 12 semanas.",
          "- A taxa de silenciamento não reage aos sinais (a pulseira ainda não sinaliza).", ""]
    DOC.write_text("\n".join(L), encoding="utf-8")


def conclusoes(res: dict) -> list[dict]:
    cen = ("pessimista", "base", "otimista")
    regra = "Regra simples (compatibilidade)"
    out = []
    ok = all(res[c]["pr_auc"][m] > res[c]["pr_auc"][regra] for c in cen for m in res[c]["pr_auc"] if m != regra)
    out.append({"conclusao": "Todo modelo supera a regra simples (todos os encontros)", "mantem": ok,
                "detalhe": "; ".join(f"{c}: regra {res[c]['pr_auc'][regra]:.3f}" for c in cen)})
    ok = all(res[c]["pr_auc_desconhecidos"][m] > res[c]["pr_auc_desconhecidos"][regra]
             for c in cen for m in res[c]["pr_auc"] if m != regra)
    perdem = [f"{c}: {m}" for c in cen for m in res[c]["pr_auc"]
              if m != regra and res[c]["pr_auc_desconhecidos"][m] <= res[c]["pr_auc_desconhecidos"][regra]]
    out.append({"conclusao": "Todo modelo supera a regra simples (só desconhecidos)", "mantem": ok,
                "detalhe": ("não supera em " + "; ".join(perdem)) if perdem else "",
                "interpretacao": "Com premissas pessimistas, nem todo modelo simples supera a regra entre desconhecidos. "
                "A vantagem do Machine Learning para o público real da pulseira depende de quanto gosto, duração e "
                "contexto realmente pesam — por isso a comparação com dados reais (seção 4) é decisiva, e o CC3/CC4 "
                "deve sempre reportar o desempenho também só entre desconhecidos."})
    melhor_vs_regra = all(max(v for m, v in res[c]["pr_auc_desconhecidos"].items() if m != regra)
                          > res[c]["pr_auc_desconhecidos"][regra] for c in cen)
    out.append({"conclusao": "O melhor modelo supera a regra simples (só desconhecidos)", "mantem": melhor_vs_regra,
                "detalhe": "; ".join(f"{c}: {max(v for m, v in res[c]['pr_auc_desconhecidos'].items() if m != regra):.3f} "
                                     f"x regra {res[c]['pr_auc_desconhecidos'][regra]:.3f}" for c in cen)})
    melhores = {c: max((m for m in res[c]["pr_auc"] if m != regra), key=lambda m: res[c]["pr_auc"][m]) for c in cen}
    dif = {c: max(v for m, v in res[c]["pr_auc"].items() if m != regra)
           - sorted(v for m, v in res[c]["pr_auc"].items() if m != regra)[-2] for c in cen}
    out.append({"conclusao": "O melhor modelo é o mesmo", "mantem": len(set(melhores.values())) == 1,
                "detalhe": "; ".join(f"{c}: {m} (+{dif[c]:.3f} sobre o 2º)" for c, m in melhores.items()),
                "interpretacao": "As diferenças entre os modelos simples são pequenas (centésimos de PR-AUC) e a ordem "
                "muda conforme as premissas. Ou seja: com estes dados, não dá para afirmar que um algoritmo é 'o melhor' "
                "para o OpiniOn. A escolha no CC4 deve se apoiar em validação cruzada (CC6), em critérios como "
                "explicabilidade e calibração, e ser reavaliada com dados reais."})
    ordens = {c: res[c]["ordem_contextos"] for c in cen}
    out.append({"conclusao": "Ordem dos contextos entre desconhecidos é a mesma", "mantem": len({tuple(o) for o in ordens.values()}) == 1,
                "detalhe": "; ".join(f"{c}: {' > '.join(o)}" for c, o in ordens.items())})
    conv_campus = all(res[c]["contexto_desconhecidos"]["convencao_games"] > res[c]["contexto_desconhecidos"]["campus"] for c in cen)
    out.append({"conclusao": "Convenção converte mais que campus (desconhecidos)", "mantem": conv_campus,
                "detalhe": "; ".join(f"{c}: {res[c]['contexto_desconhecidos']['convencao_games'] / res[c]['contexto_desconhecidos']['campus']:.1f}×" for c in cen)})
    base_top = set(res["base"]["top_features"][:3])
    estavel = all(len(base_top & set(res[c]["top_features"][:3])) >= 2 for c in cen)
    out.append({"conclusao": "As 3 features mais importantes (Random Forest) se repetem (≥ 2 de 3)", "mantem": estavel,
                "detalhe": "; ".join(f"{c}: {', '.join(res[c]['top_features'][:3])}" for c in cen),
                "interpretacao": "A importância das features depende das premissas: quando a compatibilidade pesa mais "
                "(otimista), as features de gosto sobem ao topo; quando pesa menos (pessimista), dominam as de relação "
                "prévia. Portanto, rankings de importância obtidos nestes dados NÃO devem ser apresentados como "
                "descoberta sobre os usuários reais — apenas como consequência das premissas."})
    taxas = [res[c]["taxa_conexao"] for c in cen]
    out.append({"conclusao": "Problema continua desbalanceado (3%–30% positivos)", "mantem": all(0.03 <= t <= 0.30 for t in taxas),
                "detalhe": ", ".join(f"{t:.1%}" for t in taxas)})
    return out


def main():
    t0 = time.time()
    cenarios = yaml.safe_load(CENARIOS.read_text(encoding="utf-8"))
    resultados = {}
    for nome in ("pessimista", "base", "otimista"):
        t = time.time()
        cfg = config_do_cenario(nome, cenarios[nome])
        pasta = gerar_cenario(nome, cfg)
        resultados[nome] = avaliar(pasta)
        r = resultados[nome]
        print(f"[{nome:<10}] {time.time() - t:5.1f} s | conexão {r['taxa_conexao']:.1%} | PR-AUC teste: "
              + ", ".join(f"{m.split(' (')[0]} {v:.3f}" for m, v in r["pr_auc"].items()))
    resultados["_conclusoes"] = conclusoes(resultados)
    sanidade = checagens(PASTA / "base", carregar_config())
    (PASTA / "resultados.json").write_text(json.dumps({"sanidade": sanidade, **resultados}, indent=2, ensure_ascii=False))
    escrever_doc(carregar_config(), sanidade, resultados, cenarios, time.time() - t0)
    for c in sanidade:
        print(f"  {c['status']:<8} {c['checagem']}: {c['valor']}")
    for c in resultados["_conclusoes"]:
        print(f"  {'mantém' if c['mantem'] else 'NÃO mantém':<11} {c['conclusao']}")
    print(f"\nDocumento: {DOC}")


if __name__ == "__main__":
    main()
