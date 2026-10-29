"""
Amazon Reviews 2023 (McAuley Lab, UC San Diego) -> usuários, obras e avaliações REAIS do OpiniOn.

Por que esta fonte: é o único dataset público grande em que o MESMO usuário real avalia livros,
filmes, séries, jogos e música, com nota de 1 a 5 e data. Juntar pessoas diferentes de
Spotify, Netflix e Steam exigiria inventar que são a mesma pessoa.

    python -m opinion.fontes.amazon2023 baixar     # notas 5-core (~300 MB) + metadados filtrados
    python -m opinion.fontes.amazon2023 preparar   # gera data/reais/amazon2023/ (versionado no Git)

O que vem da fonte (real): quem avaliou o quê, a nota, a data, título e categorias das obras.
O que é transformação nossa (documentada em docs/fontes_dados.md): filtro de usuários e de
itens, mapeamento das categorias para os gêneros do OpiniOn, separação filme/série e o
deslocamento de calendário (semanas inteiras) para a linha do tempo da simulação.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from opinion.fontes import CONFIG_FONTES, PASTA_EXTERNO, PASTA_REAIS

NOME = "amazon2023"
INICIO_SIMULACAO = pd.Timestamp("2026-07-06")
_RE_ASIN = re.compile(r'"parent_asin":\s*"([^"]+)"')


def config() -> dict:
    return yaml.safe_load(Path(CONFIG_FONTES).read_text(encoding="utf-8"))


# ======================================================================== download
def _baixar_arquivo(url: str, destino: Path) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        tamanho = int(r.headers["Content-Length"])
    if destino.exists() and destino.stat().st_size == tamanho:
        print(f"  já baixado: {destino.name}")
        return
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if tmp.stat().st_size != tamanho:
        raise IOError(f"download incompleto: {url}")
    tmp.rename(destino)
    print(f"  baixado: {destino.name} ({tamanho / 1e6:.0f} MB)")


def _extrair_metadados(origem, asins: set[str], destino: Path, com_texto: bool) -> None:
    """Lê o .jsonl.gz (arquivo local ou URL) e guarda só os itens que aparecem nas notas."""
    if destino.exists():
        print(f"  já extraído: {destino.name}")
        return
    linhas, t, lidas = [], time.time(), 0
    if hasattr(origem, "open") or (isinstance(origem, Path) and origem.exists()):
        raw = Path(origem).open("rb")
    else:
        raw = urllib.request.urlopen(origem, timeout=300)
    with raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for linha in f:
            lidas += 1
            m = _RE_ASIN.search(linha)
            if not m or m.group(1) not in asins:
                continue
            d = json.loads(linha)
            det = d.get("details") or {}
            item = {"parent_asin": d["parent_asin"], "titulo": (d.get("title") or "").strip(),
                    "main_category": d.get("main_category") or "",
                    "categorias": "|".join(d.get("categories") or []),
                    "genero_detalhe": str(det.get("Genre") or "")}
            if com_texto:
                texto = " ".join((d.get("features") or []) + (d.get("description") or []))
                item["texto"] = texto[:800]
            linhas.append(item)
            if lidas % 100_000 == 0:
                print(f"    {lidas:,} itens lidos, {len(linhas):,} guardados ({time.time() - t:.0f} s)", flush=True)
    pd.DataFrame(linhas).to_parquet(destino, index=False)
    print(f"  extraído: {destino.name} ({len(linhas):,} de {lidas:,} itens, {time.time() - t:.0f} s)")


def baixar(cfg: dict | None = None, pasta: Path = PASTA_EXTERNO / NOME) -> None:
    a = (cfg or config())["amazon2023"]
    pasta.mkdir(parents=True, exist_ok=True)
    # categorias menores primeiro (Books tem ~5 GB de metadados)
    ordem = sorted(a["categorias"], key=lambda c: 0 if c != "Books" else 1)
    for cat in ordem:
        print(f"[{cat}]")
        notas = pasta / f"{cat}.csv.gz"
        _baixar_arquivo(f"{a['url_base']}/benchmark/5core/rating_only/{cat}.csv.gz", notas)
        asins = set(pd.read_csv(notas, usecols=["parent_asin"], dtype=str).parent_asin)
        meta_gz = pasta / f"meta_{cat}.jsonl.gz"
        url = f"{a['url_base']}/raw/meta_categories/meta_{cat}.jsonl.gz"
        try:
            _baixar_arquivo(url, meta_gz)
            origem = meta_gz
        except Exception as e:
            print(f"  aviso: download local falhou ({e}); lendo direto da URL")
            origem = url
        _extrair_metadados(origem, asins, pasta / f"meta_{cat}.parquet", com_texto=(cat == "Video_Games"))


# ======================================================================== preparação
def _regex(palavras: list[str]) -> re.Pattern:
    return re.compile(r"(?<![\w&])(?:" + "|".join(re.escape(p) for p in palavras) + r")(?![\w&])")


def classificar_itens(meta: pd.DataFrame, cat: str, cfg: dict) -> pd.DataFrame:
    """Mídia, gêneros do OpiniOn e exclusões de cada item de uma categoria da Amazon."""
    a, rm = cfg["amazon2023"], cfg["regras_midia"]
    cats = meta.categorias.str.lower()
    base = (cats + " | " + meta.genero_detalhe.str.lower())
    texto_genero = base + (" | " + meta.titulo.str.lower() + " | " + meta.texto.str.lower()
                           if cat == "Video_Games" else "")
    midia = pd.Series(a["categorias"][cat], index=meta.index)
    excluir = pd.Series(False, index=meta.index)
    if cat == "Books":
        excluir = cats.str.contains(_regex(a["livros_excluir"]))
    elif cat == "Video_Games":
        eh_jogo = cats.str.split("|").apply(lambda c: "games" in c)
        excluir = ~eh_jogo | cats.str.contains(_regex(a["jogos_excluir"]))
    elif cat == "Movies_and_TV":
        serie = (cats.str.split("|").apply(lambda c: any(x in c for x in rm["serie_categorias"]))
                 | meta.titulo.str.lower().str.contains(_regex(rm["serie_titulo"])))
        midia[serie] = "serie"

    regras = {g: _regex(p) for g, p in cfg["generos"].items()}
    generos = pd.Series([""] * len(meta), index=meta.index)
    for g, rx in regras.items():
        tem = texto_genero.str.contains(rx)
        generos[tem] = generos[tem] + "|" + g
    generos = generos.str.strip("|")
    generos[generos == ""] = "outros_" + midia[generos == ""]   # sem gênero reconhecido: só a mídia
    return pd.DataFrame({"parent_asin": meta.parent_asin, "titulo": meta.titulo, "midia": midia,
                         "generos": generos, "categoria_amazon": cat, "excluir": excluir})


def _carregar(pasta: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    notas, itens = [], []
    for cat in cfg["amazon2023"]["categorias"]:
        n = pd.read_csv(pasta / f"{cat}.csv.gz", dtype={"user_id": str, "parent_asin": str})
        n["categoria_amazon"] = cat
        notas.append(n)
        meta = pd.read_parquet(pasta / f"meta_{cat}.parquet")
        if "texto" not in meta:
            meta["texto"] = ""
        itens.append(classificar_itens(meta.fillna(""), cat, cfg))
    notas = pd.concat(notas, ignore_index=True)
    notas["ts"] = pd.to_datetime(notas.timestamp, unit="ms").dt.floor("s")
    itens = pd.concat(itens, ignore_index=True).drop_duplicates("parent_asin")
    pop = notas.groupby("parent_asin").size().rename("n_avaliacoes_amazon")
    return notas, itens.merge(pop, left_on="parent_asin", right_index=True, how="left")


def selecionar_usuarios(notas: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    a = cfg["amazon2023"]
    s = a["selecao"]
    ini = pd.Timestamp(a["inicio_real"])
    de, ate = ini - pd.Timedelta(weeks=a["semanas_historico"]), ini + pd.Timedelta(weeks=a["semanas_simulacao"])
    hist = notas[notas.ts < ate]
    ativos = hist[hist.ts >= de].groupby("user_id").size()
    midias = hist.groupby("user_id").categoria_amazon.nunique()
    elegiveis = ativos[ativos >= s["min_avaliacoes_na_janela"]].index.intersection(
        midias[midias >= s["min_midias"]].index)
    h = hist[hist.user_id.isin(elegiveis)]
    dominante = h.assign(m=h.midia.replace({"serie": "filme"})).groupby(["user_id", "m"]).size() \
        .unstack(fill_value=0).idxmax(axis=1)

    rng = np.random.default_rng(s["semente"])
    n_alvo, restante = s["n_usuarios"], set(elegiveis)
    escolhidos = []
    for m, frac in s["cota_midia_dominante"].items():
        pool = np.sort([u for u in dominante[dominante == m].index if u in restante])
        k = min(int(round(frac * n_alvo)), len(pool))
        pegos = rng.choice(pool, size=k, replace=False) if k else np.array([], dtype=object)
        escolhidos += list(pegos)
        restante -= set(pegos)
    # completa até n_alvo com o que sobrou (filme/livro, em geral)
    falta = n_alvo - len(escolhidos)
    if falta > 0:
        pool = np.sort(list(restante))
        if len(pool) < falta:
            raise ValueError(f"só {len(escolhidos) + len(pool)} usuários elegíveis; pedimos {n_alvo}")
        escolhidos += list(rng.choice(pool, size=falta, replace=False))

    sel = h[h.user_id.isin(escolhidos)].sort_values(["user_id", "ts"])
    sel = sel.groupby("user_id", group_keys=False).tail(s["max_avaliacoes_por_usuario"])
    resumo = {"usuarios_elegiveis": int(len(elegiveis)),
              "elegiveis_por_midia_dominante": dominante.value_counts().to_dict(),
              "janela_real": [str(de.date()), str(ate.date())]}
    return sel, resumo


def preparar(cfg: dict | None = None, origem: Path = PASTA_EXTERNO / NOME, saida: Path = PASTA_REAIS / NOME) -> dict:
    cfg = cfg or config()
    a = cfg["amazon2023"]
    t = time.time()
    notas, itens = _carregar(origem, cfg)
    n_total = len(notas)
    notas = notas.merge(itens[["parent_asin", "midia", "excluir"]], on="parent_asin", how="inner")
    sem_meta = n_total - len(notas)
    notas = notas[~notas.excluir].drop(columns="excluir")
    sel, resumo = selecionar_usuarios(notas, cfg)

    desloc_dias = (INICIO_SIMULACAO - pd.Timestamp(a["inicio_real"])).days
    if desloc_dias % 7:
        raise ValueError("inicio_real precisa ser uma segunda-feira (deslocamento em semanas inteiras)")
    desloc = pd.Timedelta(days=desloc_dias)

    ids = {u: f"usr-{i + 1:05d}" for i, u in enumerate(sorted(sel.user_id.unique()))}
    sel = sel.drop_duplicates(["user_id", "parent_asin"], keep="last").sort_values(["ts", "user_id"])
    obras = itens[itens.parent_asin.isin(sel.parent_asin)].copy()
    obras["obra_id"] = "amz-" + obras.parent_asin
    obras["popularidade"] = (obras.n_avaliacoes_amazon / obras.n_avaliacoes_amazon.max()).round(4)
    obras = obras.sort_values("obra_id")

    avaliacoes = pd.DataFrame({
        "avaliacao_id": [f"av-{i + 1:06d}" for i in range(len(sel))],
        "usuario_id": sel.user_id.map(ids).to_numpy(),
        "obra_id": ("amz-" + sel.parent_asin).to_numpy(),
        "nota": sel.rating.round().astype(int).clip(1, 5).to_numpy(),
        "criado_em": (sel.ts + desloc).dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "criado_em_original": sel.ts.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
    })
    por_usuario = sel.groupby("user_id").agg(n=("rating", "size"), primeira=("ts", "min"),
                                              midias=("midia", lambda m: "|".join(sorted(set(m)))))
    usuarios = pd.DataFrame({
        "usuario_id": por_usuario.index.map(ids), "amazon_user_id": por_usuario.index,
        "n_avaliacoes": por_usuario.n.to_numpy(), "midias": por_usuario.midias.to_numpy(),
        "primeira_avaliacao": (por_usuario.primeira + desloc).dt.strftime("%Y-%m-%d").to_numpy(),
    }).sort_values("usuario_id")

    saida.mkdir(parents=True, exist_ok=True)
    usuarios.to_csv(saida / "usuarios.csv", index=False)
    obras[["obra_id", "titulo", "midia", "generos", "popularidade", "parent_asin", "categoria_amazon"]] \
        .to_csv(saida / "obras.csv", index=False)
    avaliacoes.to_csv(saida / "avaliacoes.csv", index=False)

    fonte = {
        "fonte": "Amazon Reviews 2023 (McAuley Lab, UC San Diego)",
        "url": "https://amazon-reviews-2023.github.io/",
        "citacao": "Hou, Y.; Li, J.; He, Z.; Yan, A.; Chen, X.; McAuley, J. Bridging Language and Items "
                   "for Retrieval and Recommendation. arXiv:2403.03952, 2024.",
        "arquivos_usados": {c: f"benchmark/5core/rating_only/{c}.csv.gz + raw/meta_categories/meta_{c}.jsonl.gz"
                            for c in a["categorias"]},
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "deslocamento_semanas": desloc_dias // 7,
        "notas_5core_total": int(n_total),
        "notas_sem_metadados": int(sem_meta),
        **resumo,
        "usuarios": int(len(usuarios)), "obras": int(len(obras)), "avaliacoes": int(len(avaliacoes)),
        "obras_por_midia": obras.midia.value_counts().to_dict(),
        "avaliacoes_por_midia": sel.midia.value_counts().to_dict(),
        "obras_sem_genero_reconhecido": int(obras.generos.str.startswith("outros_").sum()),
    }
    (saida / "fonte.json").write_text(json.dumps(fonte, indent=2, ensure_ascii=False))
    print(f"OK em {time.time() - t:.0f} s: {fonte['usuarios']} usuários, {fonte['obras']:,} obras, "
          f"{fonte['avaliacoes']:,} avaliações -> {saida}")
    return fonte


def main():
    ap = argparse.ArgumentParser(description="Amazon Reviews 2023 -> dados reais do OpiniOn")
    ap.add_argument("etapa", choices=["baixar", "preparar", "tudo"])
    e = ap.parse_args().etapa
    if e in ("baixar", "tudo"):
        baixar()
    if e in ("preparar", "tudo"):
        preparar()


if __name__ == "__main__":
    main()
