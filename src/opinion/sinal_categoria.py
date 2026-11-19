"""Decisão do aviso de proximidade. Uma categoria é avaliada independentemente das demais."""
from collections import defaultdict
from math import sqrt

from opinion.pipeline.compatibilidade import MIDIAS


def perfil(notas):
    """Notas anteriores ao encontro, como tuplas (obra_id, mídia, gêneros, nota)."""
    por_midia = {}
    for obra, midia, generos, nota in notas:
        pesos, obras = por_midia.setdefault(midia, (defaultdict(float), defaultdict(list)))
        obras[obra].append(nota)
        for genero in set((generos or "").split("|")):
            if genero and not genero.startswith("outros_"):
                pesos[genero] += nota - 3
    return por_midia


def _preferidos(pesos):
    positivos = [g for g in pesos if pesos[g] > 0]
    return set(sorted(positivos, key=lambda g: (-pesos[g], g))[:3])


def afinidades(a, b, cfg):
    """Equivalente por par ao cálculo vetorizado usado no treino."""
    res = {}
    pesos = cfg["compat_score"]
    favorita = cfg["features"]["nota_favorita"]
    for m in MIDIAS:
        score = 0.
        match = False
        if m in a and m in b:
            ga, oa = a[m]
            gb, ob = b[m]
            norma = sqrt(sum(x*x for x in ga.values()) * sum(x*x for x in gb.values()))
            sim = round(sum(ga[g]*gb[g] for g in ga.keys() & gb.keys()) / norma, 4) if norma else 0.
            pares = [(x, y) for o in oa.keys() & ob.keys() for x in oa[o] for y in ob[o]]
            fav = sum(x >= favorita and y >= favorita for x, y in pares)
            match = bool(fav or (sim > 0 and _preferidos(ga) & _preferidos(gb)))
            if match:
                acordo = round(1 - sum(abs(x-y) for x, y in pares)/(4*len(pares)), 4) if pares else 0.5
                score = round(pesos["peso_generos"]*(sim+1)/2 + pesos["peso_notas"]*acordo
                              + pesos["peso_favoritas"]*min(fav, 3)/3, 4)
        res[f"match_{m}"] = match
        res[f"compat_score_{m}"] = score
    return res


def feedback(encontro_id, midias, ts):
    nomes = {"filme": "filmes", "serie": "séries", "livro": "livros", "jogo": "jogos", "musica": "música"}
    categorias = [nomes[m] for m in midias]
    assunto = " e ".join(filter(None, [", ".join(categorias[:-1]), categorias[-1]]))
    return {"v": 1, "encontro_id": encontro_id, "nivel": "periferico_leve", "padrao": "luz_suave",
            "ts": ts, "midias": midias, "mensagem": f"Alguém por perto tem alta afinidade com você em {assunto}."}
