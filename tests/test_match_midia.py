import pandas as pd

from opinion.pipeline.compatibilidade import calcular
from opinion.pipeline.executar import config


def test_match_independente_por_midia_e_tempo():
    obras = pd.DataFrame([
        ("a", "musica", "rock"), ("b", "musica", "rock"),
        ("c", "filme", "drama"), ("d", "filme", "drama"),
        ("e", "livro", "rock"), ("f", "jogo", "outros_jogo"), ("g", "jogo", "outros_jogo")
    ], columns=["obra_id", "midia", "generos"])
    notas = pd.DataFrame([
        ("ana", "a", 5, 1), ("bia", "b", 4, 1), ("ana", "c", 5, 1),
        ("bia", "d", 4, 3), ("bia", "e", 4, 1), ("ana", "f", 5, 1), ("bia", "g", 5, 1)
    ], columns=["usuario_id", "obra_id", "nota", "dia"])
    notas["criado_em"] = pd.to_datetime(notas.dia, unit="D")
    encontros = pd.DataFrame({"encontro_id": ["x", "y"], "usuario_a": "ana",
                              "usuario_b": "bia", "inicio": pd.to_datetime([3, 4], unit="D")})
    resultado = calcular(encontros, notas, obras, config())
    assert resultado.match_musica.tolist() == [1, 1]
    assert resultado.match_filme.tolist() == [0, 1]
    assert resultado.historico_comum_livro.tolist() == [0, 0]
    assert resultado.match_jogo.tolist() == [0, 0]
    assert resultado.n_obras_comuns.tolist() == [0, 0]


def test_recorte_inclui_os_negativos_e_exclui_gosto_de_outras_midias():
    from opinion.modelo.comparar import recorte_midia

    df = pd.DataFrame({"historico_comum_filme": [1, 1, 0], "match_filme": [1, 0, 0],
                       "compat_score_filme": [0.8, 0, 0], "compat_score_musica": [1, 1, 1],
                       "compat_score": [1, 1, 1], "contexto": [0, 1, 0]})
    meta = {"colunas_features": df.columns.tolist(),
            "grupos_features": {"compatibilidade": list(df.columns[:-1])}}
    recorte, cols = recorte_midia(df, meta, "filme")
    assert recorte.index.tolist() == [0, 1]
    assert cols == ["contexto", "compat_score", "match_filme"]
    assert recorte.compat_score.tolist() == [0.8, 0]


def test_aviso_na_proximidade_usa_apenas_o_passado_e_envia_uma_vez(monkeypatch):
    import json
    from types import SimpleNamespace

    from opinion import contrato
    from opinion.db import Banco
    from opinion.ingestao import Ingestao

    banco = Banco("sqlite:///:memory:")
    banco.criar_esquema()
    banco.executar_varios("INSERT INTO usuarios (usuario_id, pulseira_id) VALUES (?,?)",
                          [("ana", "p1"), ("bia", "p2")])
    banco.executar_varios("INSERT INTO obras (obra_id, midia, generos) VALUES (?,?,?)",
                          [("a", "musica", "rock"), ("b", "musica", "rock"), ("c", "filme", "drama")])
    banco.executar_varios("INSERT INTO avaliacoes (avaliacao_id,usuario_id,obra_id,nota,criado_em) "
                          "VALUES (?,?,?,?,?)", [
                              ("1", "ana", "a", 5, "2026-01-01"), ("2", "bia", "b", 5, "2026-01-01"),
                              ("3", "ana", "c", 4, "2026-01-01"), ("4", "bia", "c", 4, "2026-01-02T10:00:00")])
    app = Ingestao(banco)
    enviados = []

    def publicar(topico, mensagem, qos):
        enviados.append((topico, json.loads(mensagem)))
        return SimpleNamespace(rc=0)

    monkeypatch.setattr(app.cliente, "publish", publicar)
    t = pd.Timestamp("2026-01-02T10:00:00")
    msg = {"v": 1, "encontro_id": "e1", "inicio": t.isoformat(), "local": "cinema",
           "usuario_a": contrato.token_do_dia("ana", t.date()), "usuario_b": contrato.token_do_dia("bia", t.date())}

    def emitir(tipo, campos):
        app.ao_receber(None, None, SimpleNamespace(topic=contrato.topico("p1", tipo),
                                                   payload=json.dumps(campos).encode()))

    emitir("proximidade", msg)
    emitir("proximidade", msg)
    emitir("encontro", {**msg, "duracao_s": 30, "rssi_medio": -50})
    assert len(enviados) == 1
    assert enviados[0][0] == contrato.topico("p1", "feedback")
    assert enviados[0][1]["midias"] == ["musica"]
    assert "música" in enviados[0][1]["mensagem"]
    assert app.stats["duplicadas"] == 1 and app.stats["rejeitadas"] == 0
    assert banco.um("SELECT COUNT(*) FROM encontros")[0] == 1

    emitir("proximidade", {**msg, "encontro_id": "e2", "inicio": "2026-01-02T11:00:00"})
    assert enviados[-1][1]["midias"] == ["filme", "musica"]
    banco.executar("INSERT INTO status_pulseira (usuario_id, modo, ts) VALUES (?,?,?)",
                   ("bia", "pausado", t.isoformat()))
    emitir("proximidade", {**msg, "encontro_id": "e3"})
    emitir("proximidade", {**msg, "encontro_id": "e4", "inicio": "2026-01-02T09:00:00",
                          "usuario_b": "u-desconhecido"})
    assert len(enviados) == 2 and app.stats["rejeitadas"] == 1


def test_sem_afinidade_e_sem_historico_ficam_sem_aviso():
    from opinion.pipeline.compatibilidade import motivos_do_sinal
    from opinion.sinal_categoria import afinidades, perfil

    a = perfil([("a", "musica", "rock", 5)])
    for b in [perfil([("b", "musica", "rock", 1)]), perfil([("c", "livro", "rock", 5)]), perfil([])]:
        assert motivos_do_sinal(afinidades(a, b, config()), config()) == []
