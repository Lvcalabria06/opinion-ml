"""
Gerador do bootstrap dataset (CC1) — 12 semanas de encontros entre usuários do OpiniOn.

O que é simulado
  * usuários, obras e avaliações: REAIS (Amazon Reviews 2023) quando
    config/simulacao.yaml -> usuarios.fonte = amazon2023; senão, sintéticos
  * grupos de amigos e quem segue quem
  * idas a cinema, livraria, campus e convenção de games; quem está no mesmo lugar e
    turno pode se encontrar -> a pulseira virtual registra o ENCONTRO
  * interações no app depois dos encontros (seguir, abrir perfil, curtir/comentar
    review, salvar obra) -> base do LABEL "conectou em até 48 h"
  * interações de fundo (amigos e feed), que não vêm de encontros (ruído realista)
  * toques de interesse na pulseira e trocas de modo de descoberta

O label não é uma regra: sai de um processo probabilístico com variáveis OCULTAS
(abertura social e compatibilidade "verdadeira") + ruído, conforme seção 4.3.

    python -m opinion.gerar_bootstrap                 # usa config/simulacao.yaml
    python -m opinion.gerar_bootstrap --usuarios 300 --saida data/teste
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from opinion.contrato import token_do_dia
from opinion.mundo import GENEROS, PASTA_BOOTSTRAP, CONFIG_PADRAO, RAIZ, atributos_encontro, carregar_config, sigmoide

TIPOS_INTERACAO = ["abrir_perfil", "seguir", "curtir_review", "comentar_review", "salvar_obra"]
PESOS_INTERACAO_CONEXAO = [0.35, 0.25, 0.25, 0.10, 0.05]


def _hm(s: str) -> timedelta:
    h, m = s.split(":")
    return timedelta(hours=int(h), minutes=int(m))


MIDIAS = ["filme", "serie", "jogo", "livro", "musica"]


def carregar_reais(pasta: Path) -> dict[str, pd.DataFrame]:
    """Usuários, obras e avaliações reais preparados por opinion.fontes (ex.: amazon2023)."""
    pasta = Path(pasta) if Path(pasta).is_absolute() else RAIZ / pasta
    if not (pasta / "avaliacoes.csv").exists():
        raise FileNotFoundError(f"{pasta} não tem os dados reais. Rode: python -m opinion.fontes.amazon2023 tudo")
    av = pd.read_csv(pasta / "avaliacoes.csv", parse_dates=["criado_em"])
    return {"usuarios": pd.read_csv(pasta / "usuarios.csv"), "obras": pd.read_csv(pasta / "obras.csv"),
            "avaliacoes": av.sort_values(["criado_em", "avaliacao_id"]).reset_index(drop=True)}


class Simulador:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg["simulacao"]
        self.rng = np.random.default_rng(s["semente"])
        self.inicio = datetime.fromisoformat(s["data_inicio"])
        self.dias = 7 * s["semanas"]
        self.fim = self.inicio + timedelta(days=self.dias)
        self.fonte = cfg.get("usuarios", {}).get("fonte", "sintetico")
        self.reais = carregar_reais(cfg["usuarios"]["pasta"]) if self.fonte != "sintetico" else None
        self.n = len(self.reais["usuarios"]) if self.reais else s["n_usuarios"]
        self.generos = GENEROS

    # ================================================================ catálogo
    def gerar_obras(self):
        if self.reais:
            self.obras = self.reais["obras"][["obra_id", "titulo", "midia", "generos", "popularidade"]].copy()
            self.generos = sorted({g for gs in self.obras.generos for g in gs.split("|")})
            self.obra_vet = {o: np.isin(self.generos, gs.split("|")).astype(float)
                             for o, gs in zip(self.obras.obra_id, self.obras.generos)}
            return
        rng, n = self.rng, self.cfg["simulacao"]["n_obras"]
        midias = rng.choice(["filme", "serie", "jogo", "livro", "musica"], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        linhas, cont = [], defaultdict(int)
        for i, m in enumerate(midias):
            cont[m] += 1
            gens = rng.choice(GENEROS, size=rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2]), replace=False)
            linhas.append({"obra_id": f"o-{i+1:04d}", "titulo": f"{m.capitalize()} {cont[m]:03d}",
                           "midia": m, "generos": "|".join(gens), "popularidade": float(rng.pareto(1.2) + 1)})
        self.obras = pd.DataFrame(linhas)
        self.obras["popularidade"] = (self.obras.popularidade / self.obras.popularidade.max()).round(4)
        self.obra_vet = {r.obra_id: np.isin(GENEROS, r.generos.split("|")).astype(float) for r in self.obras.itertuples()}
        self.obras_por_midia = {m: (g.obra_id.to_numpy(), (g.popularidade.to_numpy() + 0.02) / (g.popularidade + 0.02).sum())
                                for m, g in self.obras.groupby("midia")}

    # ================================================================ usuários
    def _perfil_real(self):
        """Persona, atividade e gosto a partir das avaliações reais de cada pessoa."""
        P, av = self.cfg["personas"], self.reais["avaliacoes"]
        idx = {u: i for i, u in enumerate(self.uid)}
        midia_obra = dict(zip(self.obras.obra_id, self.obras.midia))
        ui = av.usuario_id.map(idx).to_numpy()

        mix = np.zeros((self.n, len(MIDIAS)))
        np.add.at(mix, (ui, av.obra_id.map(midia_obra).map(MIDIAS.index).to_numpy()), 1.0)
        nomes = list(P)
        alvo = np.array([[P[p]["midias"].get(m, 0.0) for m in MIDIAS] for p in nomes])
        cos = (mix / np.linalg.norm(mix, axis=1, keepdims=True)) @ (alvo / np.linalg.norm(alvo, axis=1, keepdims=True)).T
        self.persona = np.array(nomes)[cos.argmax(1)]

        semanas = self.cfg["usuarios"].get("semanas_atividade", 38)
        recentes = av[av.criado_em >= self.fim - timedelta(weeks=semanas)]
        self.atividade = np.maximum(np.bincount(recentes.usuario_id.map(idx), minlength=self.n) / semanas, 0.05)

        peso = np.clip(av.nota.to_numpy() - 2.0, 0, None)            # 1–2 estrelas não puxam o gosto
        vet = np.stack([self.obra_vet[o] for o in av.obra_id])
        vet = vet / vet.sum(1, keepdims=True)
        self.gosto = np.zeros((self.n, len(self.generos)))
        np.add.at(self.gosto, ui, vet * peso[:, None])
        self.gosto += 1e-6

    def gerar_usuarios(self):
        rng, P = self.rng, self.cfg["personas"]
        if self.reais:
            self.uid = self.reais["usuarios"].usuario_id.tolist()
            self._perfil_real()
        else:
            nomes = list(P)
            pesos = np.array([P[p]["peso"] for p in nomes]); pesos /= pesos.sum()
            self.persona = rng.choice(nomes, size=self.n, p=pesos)
            self.uid = [f"usr-{i+1:05d}" for i in range(self.n)]
            self.atividade = np.array([P[p]["avaliacoes_semana"] for p in self.persona]) * rng.lognormal(0, 0.4, self.n)
            self.gosto = rng.dirichlet(np.full(len(GENEROS), 0.6), self.n)
        self.abertura = np.array([rng.beta(*P[p]["abertura_social"]) for p in self.persona])
        normas = np.linalg.norm(self.gosto, axis=1, keepdims=True)
        self.gosto_unit = self.gosto / normas
        # média da compatibilidade latente entre pares aleatórios (para centralizar o efeito)
        i, j = rng.integers(0, self.n, 5000), rng.integers(0, self.n, 5000)
        c = (self.gosto_unit[i] * self.gosto_unit[j]).sum(1)
        self.compat_media, self.compat_dp = float(c.mean()), float(c.std())

        if self.reais:
            primeira = pd.to_datetime(self.reais["usuarios"].primeira_avaliacao)
            cadastro = [min(d, self.inicio - timedelta(days=7)).date().isoformat() for d in primeira]
        else:
            cadastro = [(self.inicio - timedelta(days=int(d))).date().isoformat() for d in rng.integers(7 * 26, 400, self.n)]
        self.usuarios = pd.DataFrame({"usuario_id": self.uid, "pulseira_id": [f"PUL-{i+1:04d}" for i in range(self.n)],
                                      "data_cadastro": cadastro})

    def compat_latente(self, a: int, b: int) -> float:
        """Compatibilidade "verdadeira" padronizada (0 = média da população)."""
        return (float(self.gosto_unit[a] @ self.gosto_unit[b]) - self.compat_media) / self.compat_dp

    # ================================================================ amigos e seguidores
    def gerar_rede(self):
        rng, A = self.rng, self.cfg["amigos"]
        ordem = rng.permutation(self.n)
        alvo = int(self.n * A["fracao_em_grupos"])
        self.grupo = np.full(self.n, -1)
        self.membros: dict[int, list[int]] = {}
        k = g = 0
        while k < alvo:
            tam = int(rng.integers(A["tamanho_grupo"][0], A["tamanho_grupo"][1] + 1))
            mem = [int(x) for x in ordem[k:k + tam]]
            if len(mem) < 2:
                break
            self.membros[g] = mem
            self.grupo[mem] = g
            k += tam
            g += 1
        self.segue: dict[tuple[int, int], datetime] = {}  # (seguidor, seguido) -> desde
        for mem in self.membros.values():
            for a, b in combinations(mem, 2):
                d = self.inicio - timedelta(days=int(rng.integers(10, 300)))
                self.segue[(a, b)] = d
                self.segue[(b, a)] = d
        # seguidores "de feed": cada um segue alguns perfis ativos que não conhece
        peso = self.atividade / self.atividade.sum()
        for a in range(self.n):
            for b in rng.choice(self.n, size=rng.poisson(3), p=peso):
                if b != a and (a, int(b)) not in self.segue:
                    self.segue[(a, int(b))] = self.inicio - timedelta(days=int(rng.integers(1, 200)))

    def sao_amigos(self, a: int, b: int) -> bool:
        return self.grupo[a] >= 0 and self.grupo[a] == self.grupo[b]

    # ================================================================ avaliações
    def gerar_avaliacoes(self):
        if self.reais:
            av = self.reais["avaliacoes"]
            av = av[av.criado_em < self.fim + timedelta(days=2)]
            self.avaliacoes = av[["avaliacao_id", "usuario_id", "obra_id", "nota", "criado_em"]].reset_index(drop=True)
            idx = {u: i for i, u in enumerate(self.uid)}
            self.aval_por_usuario = defaultdict(list)
            for u, t, o in zip(av.usuario_id, av.criado_em, av.obra_id):
                self.aval_por_usuario[idx[u]].append((t.to_pydatetime(), o))
            for u in self.aval_por_usuario:
                self.aval_por_usuario[u].sort()
            self.aval_id = {(r.usuario_id, r.obra_id): r.avaliacao_id for r in self.avaliacoes.itertuples()}
            return
        rng, P = self.rng, self.cfg["personas"]
        semanas_prev = self.cfg["simulacao"]["semanas_historico_previo"]
        ini_prev = self.inicio - timedelta(weeks=semanas_prev)
        total_s = (self.fim - ini_prev).total_seconds()
        linhas = []
        self.aval_por_usuario: dict[int, list[tuple[datetime, str]]] = defaultdict(list)
        for u in range(self.n):
            mid = P[self.persona[u]]["midias"]
            ms, mp = list(mid), np.array(list(mid.values()))
            n_prev = rng.poisson(self.atividade[u] * semanas_prev * 0.6)
            n_sim = rng.poisson(self.atividade[u] * self.dias / 7)
            instantes = sorted(ini_prev + timedelta(seconds=float(x)) for x in
                               np.concatenate([rng.uniform(0, semanas_prev * 7 * 86400, n_prev),
                                               rng.uniform(semanas_prev * 7 * 86400, total_s, n_sim)]))
            vistas = set()
            for t in instantes:
                m = rng.choice(ms, p=mp / mp.sum())
                ids, pr = self.obras_por_midia[m]
                o = str(rng.choice(ids, p=pr))
                if o in vistas:
                    continue
                vistas.add(o)
                v = self.obra_vet[o]
                afin = float(self.gosto[u] @ v) / max(v.sum(), 1) * len(GENEROS)
                nota = int(np.clip(round(2.7 + 1.3 * math.tanh(afin - 1) + rng.normal(0, 0.8)), 1, 5))
                linhas.append((self.uid[u], o, nota, t))
                self.aval_por_usuario[u].append((t, o))
        self.avaliacoes = pd.DataFrame(linhas, columns=["usuario_id", "obra_id", "nota", "criado_em"])
        self.avaliacoes.insert(0, "avaliacao_id", [f"av-{i+1:06d}" for i in range(len(self.avaliacoes))])
        self.aval_id = {(r.usuario_id, r.obra_id): r.avaliacao_id for r in self.avaliacoes.itertuples()}

    # ================================================================ modos de descoberta
    def gerar_modos(self):
        rng, M = self.rng, self.cfg["modos_descoberta"]
        modos = list(M["inicial"]); p = np.array(list(M["inicial"].values()))
        self.modo_t: dict[int, list[datetime]] = {}
        self.modo_v: dict[int, list[str]] = {}
        self.status = []
        for u in range(self.n):
            m = str(rng.choice(modos, p=p / p.sum()))
            t0 = self.inicio - timedelta(minutes=int(rng.integers(1, 600)))
            ts, vs = [t0], [m]
            for w in range(self.dias // 7):
                if rng.random() < M["p_trocar_por_semana"]:
                    outros = [x for x in modos if x != m]  # volta mais para aberto/discreto que para pausado
                    po = np.array([M["inicial"][x] for x in outros])
                    m = str(rng.choice(outros, p=po / po.sum()))
                    ts.append(self.inicio + timedelta(weeks=w, seconds=float(rng.uniform(0, 7 * 86400))))
                    vs.append(m)
            self.modo_t[u], self.modo_v[u] = ts, vs
            for t, v in zip(ts, vs):
                self.status.append({"pulseira_id": f"PUL-{u+1:04d}", "usuario_id": self.uid[u], "modo": v, "ts": t})

    def modo_em(self, u: int, t: datetime) -> str:
        i = bisect.bisect_right(self.modo_t[u], t) - 1
        return self.modo_v[u][max(i, 0)]

    # ================================================================ presença e encontros
    def _presencas_do_dia(self, d: int) -> dict[tuple, set[int]]:
        rng, P, L, A = self.rng, self.cfg["personas"], self.cfg["locais"], self.cfg["amigos"]
        dia = self.inicio + timedelta(days=d)
        dow, semana = dia.weekday(), d // 7 + 1
        pres: dict[tuple, set[int]] = defaultdict(set)

        def colocar(u, chave_lista):
            for ch in chave_lista:
                pres[ch].add(u)

        for ctx in ("cinema", "livraria", "campus"):
            cl = L[ctx]
            if dow not in cl["dias"]:
                continue
            p_dia = np.array([P[p]["visitas_semana"][ctx] for p in self.persona]) / len(cl["dias"])
            vai = np.nonzero(rng.random(self.n) < np.clip(p_dia, 0, 0.95))[0]
            ja = set()
            for u in vai:
                if u in ja:
                    continue
                ch = [(ctx, int(rng.integers(cl["instancias"])), int(rng.integers(len(cl["turnos"]))))]
                colocar(int(u), ch); ja.add(int(u))
                if self.grupo[u] >= 0:  # amigos vão junto
                    for f in self.membros[self.grupo[u]]:
                        if f != u and f not in ja and rng.random() < A["p_ir_junto"]:
                            colocar(f, ch); ja.add(f)

        cg = L["convencao_games"]
        if semana in cg["semanas_com_evento"] and dow in (5, 6):
            p_ir = np.array([P[p]["convencao"] for p in self.persona])
            nt, k = len(cg["turnos"]), cg["turnos_por_visita"]
            ja = set()
            for u in np.nonzero(rng.random(self.n) < p_ir)[0]:
                if u in ja:
                    continue
                t0 = int(rng.integers(0, nt - k + 1))
                ch = [("convencao_games", 0, t) for t in range(t0, t0 + k)]
                colocar(int(u), ch); ja.add(int(u))
                if self.grupo[u] >= 0:
                    for f in self.membros[self.grupo[u]]:
                        if f != u and f not in ja and rng.random() < 0.7:
                            colocar(f, ch); ja.add(f)
        return pres

    def simular(self):
        rng, L, A, C = self.rng, self.cfg["locais"], self.cfg["amigos"], self.cfg["conexao"]
        T = self.cfg["toque_interesse"]
        self.encontros, self.toques, self.interacoes, self.ocultos = [], [], [], []
        n_anteriores: dict[tuple[int, int], int] = defaultdict(int)
        seq = 0

        for d in range(self.dias):
            dia = self.inicio + timedelta(days=d)
            vistos_hoje: set[tuple] = set()
            for (ctx, inst, turno), grupo in sorted(self._presencas_do_dia(d).items()):
                cl = L[ctx]
                base_t = dia + _hm(cl["turnos"][turno])
                ativos = sorted(u for u in grupo if self.modo_em(u, base_t) != "pausado")
                for a, b in combinations(ativos, 2):
                    chave = (ctx, inst, a, b)
                    if chave in vistos_hoje:
                        continue
                    amigos = self.sao_amigos(a, b)
                    if rng.random() >= (A["p_encontro"] if amigos else cl["p_encontro"]):
                        continue
                    inicio = base_t + timedelta(seconds=float(rng.uniform(0, 80 * 60)))
                    # o modo pode ter mudado entre o início do turno e o encontro: pulseira pausada não registra
                    if "pausado" in (self.modo_em(a, inicio), self.modo_em(b, inicio)):
                        continue
                    vistos_hoje.add(chave)
                    seq += 1
                    dur, rssi = atributos_encontro(rng, cl, amigos, A)
                    fim = inicio + timedelta(seconds=dur)
                    anteriores = n_anteriores[(a, b)]
                    n_anteriores[(a, b)] += 1
                    pub, outro = (a, b) if rng.random() < 0.5 else (b, a)  # quem publica
                    eid = f"e-{seq:06d}"
                    self.encontros.append({
                        "encontro_id": eid, "pulseira_id": f"PUL-{pub+1:04d}",
                        "usuario_a": token_do_dia(self.uid[pub], inicio.date()),
                        "usuario_b": token_do_dia(self.uid[outro], inicio.date()),
                        "inicio": inicio, "duracao_s": dur, "rssi_medio": rssi, "local": ctx,
                    })
                    # ------------------------------------------ processo do label (oculto)
                    seguem_mutuo = (a, b) in self.segue and (b, a) in self.segue
                    discreto = "discreto" in (self.modo_em(a, inicio), self.modo_em(b, inicio))
                    z = (C["intercepto"] + C["compat"] * self.compat_latente(a, b)
                         + C["abertura"] * self.abertura[a] * self.abertura[b]
                         + C["duracao"] * math.log(dur / 120) + C["contexto"][ctx]
                         + C["repeticao"] * min(anteriores, 5) + (C["discreto"] if discreto else 0.0)
                         + rng.normal(0, C["ruido"]))
                    p = sigmoide(z)
                    causou = (not seguem_mutuo) and rng.random() < p
                    if causou:
                        self._interacoes_de_conexao(a, b, fim, eid)
                    for u in (a, b):
                        if rng.random() < T["base"] + T["ganho"] * p * self.abertura[u]:
                            ts = fim + timedelta(seconds=float(rng.uniform(5, 600)))
                            self.toques.append({"encontro_id": eid, "pulseira_id": f"PUL-{u+1:04d}",
                                                "usuario": token_do_dia(self.uid[u], ts.date()), "ts": ts})
                    self.ocultos.append({
                        "encontro_id": eid, "usuario_publicou": self.uid[pub], "usuario_outro": self.uid[outro],
                        "amigos": int(amigos), "ja_se_seguiam_mutuamente": int(seguem_mutuo),
                        "compat_latente_z": round(self.compat_latente(a, b), 4),
                        "abertura_publicou": round(float(self.abertura[pub]), 4),
                        "abertura_outro": round(float(self.abertura[outro]), 4),
                        "p_conexao_latente": round(p, 5), "conexao_causada_pelo_encontro": int(causou),
                        "fim": fim,
                    })
            if (d + 1) % 14 == 0:
                print(f"  semana {(d + 1) // 7:>2}: {len(self.encontros):,} encontros até agora")
        self._interacoes_de_fundo()

    # ================================================================ interações no app
    def _alvo(self, origem: int, destino: int, tipo: str, t: datetime) -> tuple[str, str | None]:
        """Ajusta o tipo conforme o que existe (review/obra do outro antes de t)."""
        if tipo in ("curtir_review", "comentar_review", "salvar_obra"):
            lst = self.aval_por_usuario.get(destino, [])
            k = bisect.bisect_right(lst, (t, "￿"))
            if k == 0:
                return "abrir_perfil", None
            _, obra = lst[int(self.rng.integers(max(0, k - 30), k))]
            return tipo, obra
        if tipo == "seguir" and (origem, destino) in self.segue:
            return "abrir_perfil", None
        return tipo, None

    def _registrar(self, origem: int, destino: int, tipo: str, t: datetime, causa: str | None):
        tipo, obra = self._alvo(origem, destino, tipo, t)
        if tipo == "seguir":
            self.segue[(origem, destino)] = t
        self.interacoes.append({
            "origem": self.uid[origem], "destino": self.uid[destino], "tipo": tipo, "obra_id": obra,
            "avaliacao_id": self.aval_id.get((self.uid[destino], obra)) if tipo in ("curtir_review", "comentar_review") else None,
            "criado_em": t, "_causa": causa,
        })

    def _interacoes_de_conexao(self, a: int, b: int, fim: datetime, eid: str):
        rng, C = self.rng, self.cfg["conexao"]
        pa = self.abertura[a] / (self.abertura[a] + self.abertura[b])
        primeiro, segundo = (a, b) if rng.random() < pa else (b, a)
        t1 = fim + timedelta(hours=float(rng.exponential(C["atraso_medio_h"])))
        self._registrar(primeiro, segundo, str(rng.choice(TIPOS_INTERACAO, p=PESOS_INTERACAO_CONEXAO)), t1, eid)
        if rng.random() < C["p_reciproca"]:
            t2 = t1 + timedelta(hours=float(rng.exponential(10)))
            self._registrar(segundo, primeiro, str(rng.choice(TIPOS_INTERACAO, p=PESOS_INTERACAO_CONEXAO)), t2, eid)

    def _interacoes_de_fundo(self):
        """Interações que não vêm de encontros: entre amigos, com quem já segue e pelo feed."""
        rng, F = self.rng, self.cfg["interacoes_de_fundo"]
        total_dias = self.dias + 2  # +48 h para não "cortar" o label dos últimos encontros
        tipos, pesos = ["curtir_review", "comentar_review", "abrir_perfil", "salvar_obra"], [0.5, 0.15, 0.25, 0.1]
        for (a, b), desde in list(self.segue.items()):
            taxa = F["entre_amigos_por_dia"] if self.sao_amigos(a, b) else F["entre_seguidos_por_dia"]
            for x in rng.uniform(0, total_dias * 86400, rng.poisson(taxa * total_dias)):
                t = self.inicio + timedelta(seconds=float(x))
                if t > desde:
                    self._registrar(a, b, str(rng.choice(tipos, p=pesos)), t, None)
        peso = self.atividade / self.atividade.sum()
        for a in range(self.n):
            for x in rng.uniform(0, total_dias * 86400, rng.poisson(F["aleatorias_por_usuario_dia"] * total_dias)):
                b = int(rng.choice(self.n, p=peso))
                if b != a:
                    self._registrar(a, b, str(rng.choice(["abrir_perfil", "curtir_review"], p=[0.5, 0.5])),
                                    self.inicio + timedelta(seconds=float(x)), None)

    # ================================================================ label de referência
    def rotular(self) -> pd.DataFrame:
        """conectou = 1 se houve qualquer interação entre os dois em até 48 h após o fim (seção 4.4)."""
        por_par: dict[tuple[str, str], list[datetime]] = defaultdict(list)
        for r in self.interacoes:
            por_par[tuple(sorted((r["origem"], r["destino"])))].append(r["criado_em"])
        for v in por_par.values():
            v.sort()
        linhas = []
        for o in self.ocultos:
            ts = por_par.get(tuple(sorted((o["usuario_publicou"], o["usuario_outro"]))), [])
            k = bisect.bisect_right(ts, o["fim"])
            linhas.append({"encontro_id": o["encontro_id"],
                           "conectou": int(k < len(ts) and ts[k] <= o["fim"] + timedelta(hours=48))})
        return pd.DataFrame(linhas)

    # ================================================================ saída
    def salvar(self, saida: Path):
        app, pul, ver = saida / "historico_app", saida / "pulseira", saida / "verdade_simulacao_NAO_USAR_COMO_FEATURE"
        for p in (app, pul, ver):
            p.mkdir(parents=True, exist_ok=True)
        fmt = lambda df, cols: df.assign(**{c: pd.to_datetime(df[c]).dt.strftime("%Y-%m-%dT%H:%M:%S") for c in cols})

        self.usuarios.to_csv(app / "usuarios.csv", index=False)
        self.obras.to_csv(app / "obras.csv", index=False)
        fmt(self.avaliacoes, ["criado_em"]).to_csv(app / "avaliacoes.csv", index=False)
        seg = pd.DataFrame([{"seguidor": self.uid[a], "seguido": self.uid[b], "desde": t} for (a, b), t in self.segue.items()])
        fmt(seg.sort_values("desde"), ["desde"]).to_csv(app / "seguidores.csv", index=False)
        inter = pd.DataFrame(self.interacoes).sort_values("criado_em").reset_index(drop=True)
        inter.insert(0, "interacao_id", [f"i-{i+1:07d}" for i in range(len(inter))])
        causa = inter.pop("_causa")
        fmt(inter, ["criado_em"]).to_csv(app / "interacoes.csv", index=False)

        enc = pd.DataFrame(self.encontros)
        fmt(enc, ["inicio"]).to_csv(pul / "encontros.csv", index=False)
        fmt(pd.DataFrame(self.toques).sort_values("ts"), ["ts"]).to_csv(pul / "interesses.csv", index=False)
        st = pd.DataFrame(self.status).sort_values("ts")
        st["usuario"] = [token_do_dia(u, t.date()) for u, t in zip(st.usuario_id, st.ts)]
        fmt(st[["pulseira_id", "usuario", "modo", "ts"]], ["ts"]).to_csv(pul / "status.csv", index=False)

        rot = self.rotular()
        rot.to_csv(saida / "rotulos_referencia.csv", index=False)
        oc = pd.DataFrame(self.ocultos)
        fmt(oc, ["fim"]).to_csv(ver / "encontros_ocultos.csv", index=False)
        pd.DataFrame({"usuario_id": self.uid, "persona": self.persona, "abertura_social": self.abertura.round(4),
                      "atividade": self.atividade.round(3)}).to_csv(ver / "usuarios_ocultos.csv", index=False)
        inter.assign(causa_encontro=causa).loc[causa.notna(), ["interacao_id", "causa_encontro"]].to_csv(
            ver / "interacoes_causadas_por_encontro.csv", index=False)
        self.relatorio(saida, enc, rot, oc, inter, seg)

    def relatorio(self, saida, enc, rot, oc, inter, seg):
        df = enc.merge(rot, on="encontro_id").merge(oc, on="encontro_id")
        df["semana"] = (pd.to_datetime(df.inicio) - self.inicio).dt.days // 7 + 1
        pers = dict(zip(self.uid, self.persona))
        df["persona_publicou"] = df.usuario_publicou.map(pers)
        pct = lambda x: f"{100 * x:.1f}%"
        causadas_fora = oc[oc.conexao_causada_pelo_encontro == 1].merge(rot, on="encontro_id")
        dados = {
            "usuarios": self.n, "obras": len(self.obras), "avaliacoes": len(self.avaliacoes),
            "seguidores": len(seg), "interacoes": len(inter), "encontros": len(enc),
            "toques_interesse": len(self.toques), "trocas_de_modo": len(self.status),
            "taxa_conectou": round(float(df.conectou.mean()), 4),
            "taxa_conexao_causada": round(float(df.conexao_causada_pelo_encontro.mean()), 4),
        }
        (saida / "relatorio_simulacao.json").write_text(json.dumps(dados, indent=2, ensure_ascii=False))

        def tabela(col, nome):
            g = df.groupby(col).agg(encontros=("encontro_id", "size"), conectou=("conectou", "mean"),
                                    dur_mediana_s=("duracao_s", "median"), rssi_medio=("rssi_medio", "mean"))
            out = [f"| {nome} | Encontros | % conectou | Duração mediana | RSSI médio |", "|---|---|---|---|---|"]
            for k, r in g.iterrows():
                out.append(f"| {k} | {int(r.encontros):,} | {pct(r.conectou)} | {r.dur_mediana_s:.0f} s | {r.rssi_medio:.1f} dBm |")
            return out

        L = [
            "# Relatório da simulação — bootstrap dataset OpiniOn (CC1)", "",
            f"Semente {self.cfg['simulacao']['semente']} · {self.n} usuários · {self.dias // 7} semanas "
            f"({self.inicio:%d/%m/%Y} a {(self.fim - timedelta(days=1)):%d/%m/%Y}) · todos os parâmetros são premissas "
            "da equipe (`config/simulacao.yaml`).", "",
            "> Dados **sintéticos**. Qualquer desempenho medido com eles é desempenho sobre a simulação, "
            "não sobre o comportamento real dos usuários.", "",
            "## Volume", "",
            "| Item | Quantidade |", "|---|---|",
            *[f"| {k} | {v:,} |" for k, v in dados.items() if isinstance(v, int)], "",
            "## Label `conectou` (interação em até 48 h)", "",
            f"- Encontros com label positivo: **{pct(dados['taxa_conectou'])}** (problema desbalanceado, como esperado)",
            f"- Encontros que de fato *causaram* conexão (variável oculta): {pct(dados['taxa_conexao_causada'])}",
            f"- Conexões causadas cuja 1ª interação veio depois de 48 h (label = 0 mesmo assim): "
            f"{pct(1 - causadas_fora.conectou.mean()) if len(causadas_fora) else '—'}",
            f"- Positivos entre amigos (que já se seguem): {pct(df[df.amigos == 1].conectou.mean())} · "
            f"entre desconhecidos: {pct(df[df.amigos == 0].conectou.mean())}", "",
            "A diferença entre *conectou* e *causou* é intencional: amigos interagem no app de qualquer jeito, e "
            "algumas conexões só acontecem depois da janela. O modelo precisa lidar com esse ruído.", "",
            "## Por contexto", "", *tabela("local", "Local"), "",
            "## Por semana (para o split temporal 1–8 / 9–10 / 11–12)", "",
            "| Semana | Encontros | % conectou |", "|---|---|---|",
            *[f"| {s} | {int(n):,} | {pct(c)} |" for s, (n, c) in
              df.groupby("semana").agg(n=("encontro_id", "size"), c=("conectou", "mean")).iterrows()], "",
            "Picos nas semanas 3, 7 e 11 = fins de semana com convenção de games.", "",
            "## Por persona de quem publicou", "", *tabela("persona_publicou", "Persona"), "",
        ]
        (saida / "relatorio_simulacao.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        return dados


def main():
    ap = argparse.ArgumentParser(description="Gera o bootstrap dataset de encontros do OpiniOn")
    ap.add_argument("--config", default=str(CONFIG_PADRAO))
    ap.add_argument("--saida", default=str(PASTA_BOOTSTRAP))
    ap.add_argument("--usuarios", type=int)
    ap.add_argument("--semente", type=int)
    ap.add_argument("--pular-se-existir", action="store_true")
    a = ap.parse_args()
    saida = Path(a.saida)
    if a.pular_se_existir and (saida / "rotulos_referencia.csv").exists():
        print(f"Bootstrap já existe em {saida} — nada a fazer.")
        return
    cfg = carregar_config(a.config)
    if a.usuarios:
        cfg["simulacao"]["n_usuarios"] = a.usuarios
    if a.semente is not None:
        cfg["simulacao"]["semente"] = a.semente

    s = Simulador(cfg)
    print("Gerando catálogo, usuários, rede e avaliações...")
    s.gerar_obras(); s.gerar_usuarios(); s.gerar_rede(); s.gerar_avaliacoes(); s.gerar_modos()
    print("Simulando 12 semanas de encontros...")
    s.simular()
    s.salvar(saida)
    d = json.loads((saida / "relatorio_simulacao.json").read_text())
    print(f"OK: {d['encontros']:,} encontros | {d['interacoes']:,} interações | conectou {d['taxa_conectou']:.1%} -> {saida}")


if __name__ == "__main__":
    main()
