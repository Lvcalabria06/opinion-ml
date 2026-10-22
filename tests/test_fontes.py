"""Mapeamento de categorias da Amazon -> mídia e gêneros do OpiniOn (sem baixar nada)."""
from __future__ import annotations

import pandas as pd

from opinion.fontes.amazon2023 import classificar_itens, config


def _meta(**kw) -> pd.DataFrame:
    base = {"parent_asin": "X", "titulo": "Título", "main_category": "", "categorias": "",
            "genero_detalhe": "", "texto": ""}
    base.update(kw)
    return pd.DataFrame([base])


def test_livro_e_jogo_viram_midia_certa():
    cfg = config()
    livro = classificar_itens(_meta(categorias="Books|Literature & Fiction|Fantasy"), "Books", cfg)
    assert livro.midia.iloc[0] == "livro"
    assert "fantasia" in livro.generos.iloc[0]
    jogo = classificar_itens(_meta(categorias="Video Games|PC|Games", texto="an action RPG"), "Video_Games", cfg)
    assert jogo.midia.iloc[0] == "jogo"
    assert "rpg" in jogo.generos.iloc[0]


def test_serie_detectada_por_categoria_e_titulo():
    cfg = config()
    tv = classificar_itens(_meta(categorias="Movies & TV|TV"), "Movies_and_TV", cfg)
    assert tv.midia.iloc[0] == "serie"
    filme = classificar_itens(_meta(categorias="Movies & TV|Movies|Drama", genero_detalhe="Drama"),
                              "Movies_and_TV", cfg)
    assert filme.midia.iloc[0] == "filme"
    assert "drama" in filme.generos.iloc[0]
    temporada = classificar_itens(_meta(titulo="Show: Season 2", categorias="Movies & TV"), "Movies_and_TV", cfg)
    assert temporada.midia.iloc[0] == "serie"


def test_exclui_livro_tecnico_e_acessorio_de_jogo():
    cfg = config()
    tec = classificar_itens(_meta(categorias="Books|Computers & Technology"), "Books", cfg)
    assert tec.excluir.iloc[0]
    acc = classificar_itens(_meta(categorias="Video Games|Accessories|Headsets"), "Video_Games", cfg)
    assert acc.excluir.iloc[0]
