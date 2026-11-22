"""
Last.fm (HetRec 2011, GroupLens) -> usuários, artistas, escutas e amizades REAIS do OpiniOn.

Por que esta fonte: traz amizades reais entre usuários (user_friends) e o quanto cada um escutou
cada artista. O dataset não tem data de escuta nem nota; por isso:
  - nota 1-5 = quintil das escutas DENTRO do usuário (transformação nossa);
  - todas as avaliações recebem a mesma data (config lastfm.data_perfil);
  - as datas das tags NÃO são usadas (têm anos inválidos).

    python -m opinion.fontes.lastfm     # lê data/externo/lastfm/... e gera data/reais/lastfm/
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opinion.fontes import PASTA_REAIS
from opinion.fontes.amazon2023 import _regex, config
from opinion.mundo import RAIZ

NOME = "lastfm"


def _ler(origem: Path, arquivo: str, encoding: str = "latin-1") -> pd.DataFrame:
    # arquivos separados por tab; aspas aparecem nos nomes (não são delimitador).
    # A codificação varia por arquivo (artists.dat é UTF-8; tags.dat é latin-1).
    return pd.read_csv(origem / arquivo, sep="	", encoding=encoding, quoting=csv.QUOTE_NONE,
                       dtype=str, keep_default_na=False)



def notas_por_quintil(escutas: pd.Series, usuarios: pd.Series) -> pd.Series:
    """Nota 1-5 = ceil(posição relativa das escutas dentro do usuário * 5)."""
    pct = escutas.groupby(usuarios).rank(method="average", pct=True)
    return np.ceil(pct * 5).clip(1, 5).astype(int)


def generos_dos_artistas(tags_usuario: pd.DataFrame, tags: pd.DataFrame, cfg: dict) -> pd.Series:
    """Gêneros do OpiniOn por artista (índice artistID) a partir das tags dos usuários."""
    c = cfg["lastfm"]
    regras = {g: _regex(p) for g, p in cfg["generos"].items()}
    por_tag = {tid: [g for g, rx in regras.items() if rx.search(str(v).lower())]
               for tid, v in zip(tags.tagID, tags.tagValue)}
    usos = tags_usuario.groupby(["artistID", "tagID"]).size().rename("n").reset_index()
    usos = usos[usos.n >= c["min_usos_tag"]].sort_values(["artistID", "n"], ascending=[True, False],
                                                          kind="stable")
    generos: dict = {}
    for art, tid in zip(usos.artistID, usos.tagID):
        lista = generos.setdefault(art, [])
        for g in por_tag.get(tid, []):
            if g not in lista and len(lista) < c["max_generos_por_artista"]:
                lista.append(g)
    todos = tags_usuario.artistID.unique()
    return pd.Series({a: "|".join(generos.get(a) or []) or "outros_musica" for a in todos},
                     name="generos").rename_axis("artistID")


def preparar(origem: Path, saida: Path, cfg: dict | None = None) -> dict:
    cfg = cfg or config()
    c = cfg["lastfm"]
    t = time.time()
    artistas = _ler(origem, "artists.dat", cfg["lastfm"].get("encoding_artists", "utf-8"))
    escutas = _ler(origem, "user_artists.dat")
    amigos = _ler(origem, "user_friends.dat")
    tags = _ler(origem, "tags.dat", cfg["lastfm"].get("encoding_tags", "latin-1"))
    tags_usuario = _ler(origem, "user_taggedartists.dat")[["userID", "artistID", "tagID"]]

    escutas["escutas"] = escutas.weight.astype(int)
    escutas["nota"] = notas_por_quintil(escutas.escutas, escutas.userID)
    escutas["usuario_id"] = "usr-" + escutas.userID.astype(int).map("{:05d}".format)
    escutas["obra_id"] = "lfm-" + escutas.artistID
    escutas = escutas.sort_values(["userID", "artistID"], key=lambda s: s.astype(int), kind="stable")

    generos = generos_dos_artistas(tags_usuario, tags, cfg)
    ouvintes = escutas.groupby("artistID").size()
    obras = pd.DataFrame({
        "obra_id": "lfm-" + artistas.id, "titulo": artistas.name.str.strip(), "midia": "musica",
        "generos": artistas.id.map(generos).fillna("outros_musica"),
        "popularidade": (artistas.id.map(ouvintes).fillna(0) / ouvintes.max()).round(4),
    }).sort_values("obra_id")

    avaliacoes = pd.DataFrame({
        "avaliacao_id": [f"av-{i + 1:06d}" for i in range(len(escutas))],
        "usuario_id": escutas.usuario_id.to_numpy(), "obra_id": escutas.obra_id.to_numpy(),
        "nota": escutas.nota.to_numpy(), "criado_em": c["data_perfil"],
        "escutas": escutas.escutas.to_numpy(),
    })
    por_usuario = escutas.groupby(["usuario_id", "userID"]).size().rename("n_avaliacoes").reset_index()
    usuarios = pd.DataFrame({
        "usuario_id": por_usuario.usuario_id, "lastfm_user_id": por_usuario.userID,
        "n_avaliacoes": por_usuario.n_avaliacoes, "midias": "musica",
        "primeira_avaliacao": c["data_perfil"],
    }).sort_values("usuario_id")

    a = "usr-" + amigos.userID.astype(int).map("{:05d}".format)
    b = "usr-" + amigos.friendID.astype(int).map("{:05d}".format)
    amizades = pd.DataFrame({"usuario_a": np.minimum(a, b), "usuario_b": np.maximum(a, b)}) \
        .drop_duplicates().sort_values(["usuario_a", "usuario_b"])

    saida.mkdir(parents=True, exist_ok=True)
    usuarios.to_csv(saida / "usuarios.csv", index=False)
    obras.to_csv(saida / "obras.csv", index=False)
    avaliacoes.to_csv(saida / "avaliacoes.csv", index=False)
    amizades.to_csv(saida / "amizades.csv", index=False)

    fonte = {
        "fonte": "Last.fm (HetRec 2011)",
        "url": "http://www.lastfm.com",
        "citacao": "Cantador, I.; Brusilovsky, P.; Kuflik, T. 2nd Workshop on Information Heterogeneity "
                   "and Fusion in Recommender Systems (HetRec 2011). RecSys 2011. Dados do site Last.fm "
                   "(http://www.lastfm.com).",
        "licenca": "uso não comercial",
        "arquivos_usados": ["artists.dat", "user_artists.dat", "user_friends.dat", "tags.dat",
                            "user_taggedartists.dat"],
        "transformacoes": ["nota 1-5 = quintil das escutas dentro de cada usuário",
                           f"criado_em fixo em {c['data_perfil']} (sem data de escuta na fonte)",
                           "gêneros a partir das tags (>= min_usos_tag usos), via seção generos",
                           "datas das tags não usadas (anos inválidos)"],
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "usuarios": int(len(usuarios)), "obras": int(len(obras)),
        "avaliacoes": int(len(avaliacoes)), "amizades": int(len(amizades)),
        "obras_sem_genero_reconhecido": int((obras.generos == "outros_musica").sum()),
    }
    (saida / "fonte.json").write_text(json.dumps(fonte, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK em {time.time() - t:.0f} s: {fonte['usuarios']} usuários, {fonte['obras']:,} obras, "
          f"{fonte['avaliacoes']:,} avaliações, {fonte['amizades']:,} amizades -> {saida}")
    return fonte


def main():
    cfg = config()
    preparar(RAIZ / cfg["lastfm"]["pasta_origem"], PASTA_REAIS / NOME, cfg)


if __name__ == "__main__":
    main()
