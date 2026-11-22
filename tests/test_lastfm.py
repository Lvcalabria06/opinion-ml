"""Fonte Last.fm (HetRec 2011): notas por quintil, gêneros por tags e tabelas geradas."""
import json
import numpy as np
import pandas as pd
import pytest

from opinion.fontes import lastfm
from opinion.fontes.lastfm import generos_dos_artistas, notas_por_quintil, preparar


def test_nota_por_quintil_por_usuario():
    escutas = pd.Series([10, 20, 30, 40, 50, 7])
    usuarios = pd.Series(["a"] * 5 + ["b"])
    notas = notas_por_quintil(escutas, usuarios)
    assert notas.tolist() == [1, 2, 3, 4, 5, 5]


def _tags(valores: list[str]):
    tags = pd.DataFrame({"tagID": range(len(set(valores))), "tagValue": sorted(set(valores))})
    id_de = dict(zip(tags.tagValue, tags.tagID))
    uso = pd.DataFrame({"userID": range(len(valores)), "artistID": 1,
                        "tagID": [id_de[v] for v in valores]})
    return uso, tags


def test_artista_sem_tag_mapeada_vira_outros_musica():
    cfg = lastfm.config()
    cfg["lastfm"]["min_usos_tag"] = 1
    uso, tags = _tags(["seen live"])
    assert generos_dos_artistas(uso, tags, cfg).loc[1] == "outros_musica"
    uso, tags = _tags(["rock", "rock", "indie"])
    g = generos_dos_artistas(uso, tags, cfg).loc[1].split("|")
    assert "rock_metal" in g and "indie_alternativo" in g
    assert g[0] == "rock_metal"          # a mais usada vem primeiro


def test_min_usos_tag_filtra_tags_raras():
    cfg = lastfm.config()   # min_usos_tag = 2
    uso, tags = _tags(["rock", "rock", "indie"])
    assert generos_dos_artistas(uso, tags, cfg).loc[1] == "rock_metal"


def test_preparar_gera_tabelas_reais(tmp_path):
    cfg = lastfm.config()
    origem = lastfm.RAIZ / cfg["lastfm"]["pasta_origem"]
    if not origem.exists():
        pytest.skip("dataset Last.fm não baixado")
    preparar(origem, tmp_path, cfg)
    usuarios = pd.read_csv(tmp_path / "usuarios.csv")
    obras = pd.read_csv(tmp_path / "obras.csv")
    av = pd.read_csv(tmp_path / "avaliacoes.csv")
    am = pd.read_csv(tmp_path / "amizades.csv")
    assert (len(usuarios), len(obras), len(av), len(am)) == (1892, 17632, 92834, 12717)
    assert (am.usuario_a < am.usuario_b).all()
    assert not am.duplicated().any()
    assert av.nota.between(1, 5).all()
    assert (av.criado_em == "2026-06-01").all()
    assert set(av.usuario_id) <= set(usuarios.usuario_id) and set(av.obra_id) <= set(obras.obra_id)
    assert (tmp_path / "fonte.json").exists()
    # artists.dat é UTF-8: acentos não podem virar mojibake
    assert "Hildur Guðnadóttir" in set(obras.titulo)
    assert obras.titulo.str.contains("Ã[°©º³¡]").sum() < 5  # mojibake típico (resíduo legítimo mínimo)


# ------------------------------------------------------------------ simulador (modo lastfm)
from opinion.gerar_bootstrap import Simulador
from opinion.mundo import CONFIG_PADRAO, RAIZ, carregar_config


def _cfg_lastfm(semanas=2):
    cfg = carregar_config(CONFIG_PADRAO)
    cfg["usuarios"] = {"fonte": "lastfm", "pasta": "data/reais/lastfm"}
    cfg["simulacao"]["semanas"] = semanas
    return cfg


def _pula_sem_dados():
    if not (RAIZ / "data/reais/lastfm/amizades.csv").exists():
        pytest.skip("dados Last.fm não preparados")


def _preparar(s):
    s.gerar_obras(); s.gerar_usuarios(); s.gerar_rede(); s.gerar_avaliacoes(); s.gerar_modos()
    return s


def test_encontros_nao_dependem_da_amizade():
    _pula_sem_dados()
    real = Simulador(_cfg_lastfm())
    embaralhado = Simulador(_cfg_lastfm())
    n, uid = len(embaralhado.reais["usuarios"]), embaralhado.reais["usuarios"].usuario_id.tolist()
    sorteio = np.random.default_rng(7)
    pares = set()
    while len(pares) < len(real.reais["amizades"]):
        a, b = sorteio.integers(n, size=2)
        if a != b:
            pares.add((uid[min(a, b)], uid[max(a, b)]))
    embaralhado.reais["amizades"] = pd.DataFrame(sorted(pares), columns=["usuario_a", "usuario_b"])
    for s in (real, embaralhado):
        _preparar(s).simular()
    cols = ["encontro_id", "usuario_a", "usuario_b", "inicio", "local", "duracao_s", "rssi_medio"]
    pd.testing.assert_frame_equal(pd.DataFrame(real.encontros)[cols], pd.DataFrame(embaralhado.encontros)[cols])
    assert len(real.encontros) > 0 and real.interacoes == [] and embaralhado.interacoes == []


def test_rotulo_de_referencia_e_a_amizade():
    _pula_sem_dados()
    s = _preparar(Simulador(_cfg_lastfm()))
    am = s.reais["amizades"].iloc[0]
    amigos = {"usuario_publicou": am.usuario_b, "usuario_outro": am.usuario_a}
    nao = next((a, b) for a in s.uid[:20] for b in s.uid[20:40]
               if tuple(sorted((a, b))) not in set(zip(s.reais["amizades"].usuario_a, s.reais["amizades"].usuario_b)))
    s.ocultos = [{"encontro_id": "e-1", **amigos, "fim": s.inicio},
                 {"encontro_id": "e-2", "usuario_publicou": nao[0], "usuario_outro": nao[1], "fim": s.inicio}]
    assert s.rotular().set_index("encontro_id").conectou.to_dict() == {"e-1": 1, "e-2": 0}


# ---------- Pipeline (Task 3): rótulo por amizade, split por pares, features excluídas ----------
from opinion.pipeline import label, preparacao
from opinion.pipeline.executar import config as config_pipeline


def test_label_amizade():
    cfg = config_pipeline()
    cfg["label"]["tipo"] = "amizade"
    t = pd.Timestamp("2026-08-01 20:00")
    enc = pd.DataFrame({"encontro_id": ["x", "y"], "usuario_a": ["u1", "u1"], "usuario_b": ["u2", "u3"],
                        "inicio": [t, t], "fim": [t + pd.Timedelta(minutes=10)] * 2})
    seg = pd.DataFrame({"seguidor": ["u1", "u2"], "seguido": ["u2", "u1"],
                        "desde": [pd.Timestamp("2026-06-01")] * 2})
    inter = pd.DataFrame({"origem": [], "destino": [], "tipo": [], "criado_em": pd.to_datetime([])})
    r, corte = label.calcular(enc, inter, cfg, seguidores=seg)
    r = r.set_index("encontro_id")
    assert r.conectou.to_dict() == {"x": 1, "y": 0}
    assert (r.label_disponivel == 1).all()
    assert isinstance(corte, pd.Timestamp)


def test_split_por_pares_nao_mistura_pares():
    cfg = config_pipeline()
    n = 1000
    par = pd.Series([f"p{i // 3}" for i in range(n * 3)])          # 3 encontros por par
    conectou = pd.Series([1 if (i // 3) % 20 == 0 else 0 for i in range(n * 3)])
    s = preparacao.split_por_pares(par, conectou, cfg)
    assert (pd.DataFrame({"p": par, "s": s}).groupby("p").s.nunique() == 1).all()
    prop = s.groupby(par).first().value_counts(normalize=True)
    for nome, alvo in cfg["split"]["proporcoes"].items():
        assert abs(prop[nome] - alvo) <= 0.02
    for nome in ("treino", "validacao", "teste"):
        assert conectou[s == nome].sum() > 0


def test_features_excluidas_fora_do_modelo():
    cfg = config_pipeline()
    excluir = cfg["features"]["excluir"]
    assert excluir
    cols = preparacao.colunas_numericas(cfg)
    assert not set(excluir) & set(cols)
    assert set(preparacao.colunas_numericas(None)) >= set(excluir)


def test_validacao_amizade_sem_falhas(tmp_path, monkeypatch):
    """Com label.tipo: amizade a validação checa só os encontros + independência, sem falhas."""
    from opinion import validacao_simulacao as v

    if not v.modo_amizade() or not (v.RAIZ / "data" / "staging" / "encontros.parquet").exists():
        pytest.skip("precisa de data/ gerado com label.tipo: amizade")
    monkeypatch.setattr(v, "DOC", tmp_path / "validacao.md")
    sanidade = v.main_amizade()
    assert sanidade and not [c for c in sanidade if c["status"] == "FALHA"]
    assert any("independência" in c["checagem"] for c in sanidade)
    texto = (tmp_path / "validacao.md").read_text(encoding="utf-8")
    assert "rótulo é real" in texto
    assert "## 3. Análise de sensibilidade" not in texto and "pessimista" not in texto
    assert "razão entre 0,85 e 1,15" in texto and "aumentada no experimento" not in texto
    assert "multiplicada por 3" in texto


def test_relatorio_lastfm_descreve_rotulo_real(tmp_path):
    """O relatório do modo lastfm não pode alegar dados sintéticos de rótulo, causalidade, 48 h nem split temporal."""
    _pula_sem_dados()
    s = _preparar(Simulador(_cfg_lastfm()))
    s.simular()
    s.salvar(tmp_path)
    texto = (tmp_path / "relatorio_simulacao.md").read_text(encoding="utf-8")
    for proibido in ("sintétic", "48 h", "causou", "split temporal"):
        assert proibido not in texto, proibido
    assert "amizade" in texto
    dados = json.loads((tmp_path / "relatorio_simulacao.json").read_text(encoding="utf-8"))
    assert "taxa_conexao_causada" not in dados
