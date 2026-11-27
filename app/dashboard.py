"""
Dashboard do OpiniOn (CC9).

    streamlit run app/dashboard.py                    # http://localhost:8501
    OPINION_API_URL=http://localhost:8000 streamlit run app/dashboard.py

Lê tudo da API de inferência (CC7): encontros que chegam pela pulseira (MQTT → ingestão →
banco), as inferências do modelo, os sinais emitidos, as métricas do modelo registrado no
MLflow e os indicadores da seção 3.2 (taxa de conversão, precisão dos sinais e taxa de
silenciamento). A página se atualiza sozinha.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("OPINION_API_URL", "http://localhost:8000").rstrip("/")
NIVEL = {"nenhum": "sem sinal", "periferico_leve": "leve", "periferico_perceptivel": "perceptível"}

st.set_page_config(page_title="OpiniOn — painel", page_icon="📡", layout="wide")


def api(caminho: str, metodo: str = "get", **kw):
    try:
        r = requests.request(metodo, f"{API}{caminho}", timeout=60, **kw)
    except requests.RequestException as ex:
        return None, f"API indisponível em {API}: {ex}"
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail", r.text)
        except ValueError:
            return None, r.text
    return r.json(), None


def pct(x, casas=1):
    return "—" if x is None else f"{100 * x:.{casas}f}%".replace(".", ",")


with st.sidebar:
    st.title("OpiniOn")
    st.caption("Pulseira → MQTT → banco → Airflow → modelo → API → este painel")
    intervalo = st.select_slider("Atualizar a cada", options=[5, 10, 30, 60, 300], value=10,
                                 format_func=lambda s: f"{s} s")
    limite = st.slider("Linhas nas tabelas", 10, 200, 50, step=10)
    saude, erro = api("/saude")
    if erro:
        st.error(erro)
    else:
        st.success(f"API ok · banco {saude.get('banco')}")
        st.caption(f"Modelo: {saude.get('modelo')}")
    st.caption(f"API: {API}")

st.title("Encontros e sinais em tempo real")


@st.fragment(run_every=intervalo)
def painel_ao_vivo():
    ind, erro = api("/indicadores")
    if erro:
        st.warning(f"Indicadores indisponíveis: {erro}")
        ind = {}
    av, sm, sil = ind.get("aviso_afinidade", {}), ind.get("sinal_modelo", {}), ind.get("silenciamento", {})
    c = st.columns(5)
    c[0].metric("Encontros recebidos", f"{ind.get('encontros', 0):,}".replace(",", "."))
    c[1].metric("Taxa de conversão (48 h)", pct(ind.get("taxa_conversao")),
                help="Encontros com a janela de 48 h encerrada que viraram interação no app.")
    c[2].metric("Precisão do aviso de afinidade", pct(av.get("precisao")),
                help=f"{av.get('com_label', 0)} avisos com resultado conhecido de {av.get('sinais', 0)} emitidos.")
    c[3].metric("Precisão do sinal do modelo", pct(sm.get("precisao")),
                help=f"{sm.get('com_label', 0)} sinais com resultado conhecido de {sm.get('sinais', 0)} emitidos. "
                     "Encontros ao vivo só ganham resultado 48 h depois.")
    c[4].metric("Taxa de silenciamento", pct(sil.get("taxa")),
                help="Sinais seguidos de pausa da pulseira em até 24 h (fadiga de notificação).")

    inf, erro = api(f"/inferencias?limite={limite}")
    esq, dir_ = st.columns([3, 2])
    with esq:
        st.subheader("Últimas inferências do modelo")
        if erro:
            st.info(erro)
        elif not inf:
            st.info("Nenhuma inferência ainda. O consumidor MQTT pontua cada encontro novo da pulseira ao vivo; "
                    "use também a aba **Simular encontro**.")
        else:
            df = pd.DataFrame(inf)
            df["chance"] = df.probabilidade
            df["sinal"] = df.nivel.map(NIVEL)
            st.dataframe(df[["criado_em", "encontro_id", "local", "usuario_a", "usuario_b", "chance", "sinal",
                             "midias_aviso", "latencia_ms", "origem"]],
                         hide_index=True, width="stretch",
                         column_config={"chance": st.column_config.ProgressColumn("chance 48 h", min_value=0,
                                                                                    max_value=1, format="%.3f")})
    with dir_:
        st.subheader("Sinais emitidos")
        if inf:
            df = pd.DataFrame(inf)
            st.bar_chart(df.nivel.map(NIVEL).value_counts(), horizontal=True, x_label="encontros")
            st.caption(f"Chance média nas últimas {len(df)} inferências: {pct(df.probabilidade.mean())}")
        st.markdown(f"**Aviso de afinidade (na proximidade):** {av.get('sinais', 0):,} sinais".replace(",", "."))
        if av.get("por_midia"):
            st.bar_chart(pd.Series(av["por_midia"], name="avisos"), horizontal=True)

    enc, erro = api(f"/encontros/recentes?limite={limite}")
    st.subheader("Últimos encontros recebidos pela ingestão (MQTT)")
    if erro:
        st.info(erro)
    elif enc:
        st.dataframe(pd.DataFrame(enc), hide_index=True, width="stretch")


aba_vivo, aba_ind, aba_modelo, aba_sim = st.tabs(["Ao vivo", "Indicadores", "Modelo", "Simular encontro"])
with aba_vivo:
    painel_ao_vivo()

with aba_ind:
    ind, erro = api("/indicadores")
    if erro:
        st.warning(erro)
    else:
        st.caption(f"Calculado em {ind.get('calculado_em')} · interações conhecidas até "
                   f"{ind.get('dados_de_interacao_ate')}")
        a, b = st.columns(2)
        with a:
            st.subheader("Conversão por local")
            if ind.get("conversao_por_local"):
                st.bar_chart(pd.Series(ind["conversao_por_local"], name="taxa de conversão"))
        with b:
            st.subheader("Fadiga de notificação")
            s = ind.get("silenciamento", {})
            st.metric("Sinais seguidos de pausa (24 h)", f"{s.get('seguidos_de_pausa_24h', 0)} de {s.get('sinais', 0)}")
            st.metric("Pulseiras pausadas agora", f"{s.get('pulseiras_pausadas_agora', 0)} de {s.get('pulseiras_com_status', 0)}")
        if ind.get("por_dia"):
            dia = pd.DataFrame(ind["por_dia"]).set_index("dia")
            st.subheader("Encontros e avisos por dia")
            st.line_chart(dia[["encontros", "avisos"]])
            st.subheader("Taxa de conversão por dia")
            st.line_chart(dia[["taxa_conversao"]])
        with st.expander("Definições"):
            st.markdown(
                "- **Taxa de conversão:** encontros que terminaram em interação no app (seguir, abrir perfil, "
                "curtir/comentar review, salvar obra) em até 48 h, entre os que já completaram as 48 h.\n"
                "- **Precisão dos sinais:** entre os encontros que receberam sinal, a fração que converteu.\n"
                "- **Taxa de silenciamento:** sinais seguidos de troca da pulseira para *pausado* em até 24 h.")

with aba_modelo:
    m, erro = api("/modelo")
    if erro:
        st.warning(erro)
    else:
        c = st.columns(4)
        c[0].metric("Modelo servido", f"{m['nome']} v{m['versao']}")
        c[1].metric("Algoritmo", m.get("algoritmo") or "—")
        c[2].metric("Limiar do sinal", f"{m['limiar']:.3f}")
        c[3].metric("Calibração", m.get("calibracao"))
        met = m.get("metricas", {})
        linhas = []
        for nome, chave in (("PR-AUC entre desconhecidos", "pr_auc_desconhecidos"), ("PR-AUC geral", "pr_auc"),
                            ("ROC-AUC entre desconhecidos", "roc_auc_desconhecidos"), ("Brier", "brier"),
                            ("Precisão no limiar (desc.)", "no_limiar_desconhecidos_precisao"),
                            ("Recall no limiar (desc.)", "no_limiar_desconhecidos_recall"),
                            ("Fração com sinal (desc.)", "no_limiar_desconhecidos_taxa_sinal"),
                            ("Prevalência entre desconhecidos", "prevalencia_desconhecidos")):
            linhas.append({"métrica": nome, "validação": met.get(f"validacao_{chave}"), "teste": met.get(f"teste_{chave}")})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch",
                     column_config={"validação": st.column_config.NumberColumn(format="%.3f"),
                                    "teste": st.column_config.NumberColumn(format="%.3f")})
        st.caption(f"Run do MLflow: {m['run_id']} · carregado em {m['carregado_em']}. Métricas sobre dados "
                   "simulados (encontros e label); veja docs/comparacao_modelos.md, otimizacao.md e validacao_cruzada.md.")

with aba_sim:
    st.write("Pontua um encontro na hora, pelo mesmo caminho da pulseira (features só com o passado, modelo do MLflow).")
    with st.form("simular"):
        c = st.columns(5)
        ua = c[0].text_input("Usuário A", "usr-00694")
        ub = c[1].text_input("Usuário B", "usr-01029")
        local = c[2].selectbox("Local", ["cinema", "livraria", "campus", "convencao_games"])
        dur = c[3].number_input("Duração (s)", 30, 7200, 600, step=30)
        rssi = c[4].number_input("RSSI médio (dBm)", -110, 0, -58)
        registrar = st.checkbox("Gravar em inferências (aparece no painel ao vivo)", value=True)
        if st.form_submit_button("Estimar chance"):
            r, erro = api(f"/prever/usuarios?registrar={str(registrar).lower()}", "post",
                          json={"usuario_a": ua, "usuario_b": ub, "local": local, "duracao_s": dur, "rssi_medio": rssi})
            if erro:
                st.error(erro)
            else:
                c = st.columns(4)
                c[0].metric("Chance de interação em 48 h", pct(r["probabilidade"]))
                c[1].metric("Sinal na pulseira", NIVEL[r["nivel"]])
                c[2].metric("Já se seguem?", "sim" if r["algum_segue"] else "não")
                c[3].metric("Encontros anteriores do par", r["encontros_anteriores_par"])
                st.bar_chart(pd.Series(r["afinidade_por_midia"], name="afinidade"), horizontal=True)
                st.caption(f"Aviso de afinidade: {', '.join(r['midias_aviso']) or 'nenhuma mídia acima do corte'} · "
                           f"{r['modelo']} ({r['algoritmo']}) · {r['latencia_ms']} ms")
