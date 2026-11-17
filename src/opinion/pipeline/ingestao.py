"""
Etapa 1 — Ingestão para o pipeline.

Lê as tabelas do banco operacional (preenchidas pela ingestão MQTT do CC1 e pelo
histórico do app) e grava uma "foto" padronizada em data/staging/ (Parquet).
As etapas seguintes leem só do staging, nunca do banco direto: assim cada execução
da DAG trabalha sobre um retrato consistente dos dados.

Também aceita os arquivos do bootstrap como fonte (modo "arquivos"), usado pela
análise de sensibilidade: ela gera vários datasets e não passa cada um pelo MQTT.
Nesse modo os IDs rotativos são traduzidos exatamente como a ingestão MQTT faz.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from opinion import contrato as C
from opinion.pipeline import PASTA_STAGING

TABELAS = ("encontros", "avaliacoes", "obras", "seguidores", "interacoes", "status", "interesses", "usuarios")


def _datas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    return df


def _padronizar(f: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    _datas(f["encontros"], ["inicio"])
    f["encontros"]["fim"] = f["encontros"]["inicio"] + pd.to_timedelta(f["encontros"]["duracao_s"], unit="s")
    _datas(f["avaliacoes"], ["criado_em"])
    _datas(f["seguidores"], ["desde"])
    _datas(f["interacoes"], ["criado_em"])
    _datas(f["status"], ["ts"])
    _datas(f["interesses"], ["ts"])
    return f


# ----------------------------------------------------------------- fontes
def carregar_do_banco(banco) -> dict[str, pd.DataFrame]:
    q = lambda sql, cols: pd.DataFrame(banco.todos(sql), columns=cols)
    f = {
        "usuarios": q("SELECT usuario_id, pulseira_id, data_cadastro FROM usuarios",
                      ["usuario_id", "pulseira_id", "data_cadastro"]),
        "encontros": q("SELECT encontro_id, pulseira_id, usuario_a, usuario_b, inicio, duracao_s, rssi_medio, local "
                       "FROM encontros", ["encontro_id", "pulseira_id", "usuario_a", "usuario_b", "inicio",
                                          "duracao_s", "rssi_medio", "local"]),
        "avaliacoes": q("SELECT avaliacao_id, usuario_id, obra_id, nota, criado_em FROM avaliacoes",
                        ["avaliacao_id", "usuario_id", "obra_id", "nota", "criado_em"]),
        "obras": q("SELECT obra_id, titulo, midia, generos, popularidade FROM obras",
                   ["obra_id", "titulo", "midia", "generos", "popularidade"]),
        "seguidores": q("SELECT seguidor, seguido, desde FROM seguidores", ["seguidor", "seguido", "desde"]),
        "interacoes": q("SELECT interacao_id, origem, destino, tipo, obra_id, criado_em FROM interacoes",
                        ["interacao_id", "origem", "destino", "tipo", "obra_id", "criado_em"]),
        "status": q("SELECT pulseira_id, usuario_id, modo, ts FROM status_pulseira",
                    ["pulseira_id", "usuario_id", "modo", "ts"]),
        "interesses": q("SELECT encontro_id, pulseira_id, usuario_id, ts FROM interesses",
                        ["encontro_id", "pulseira_id", "usuario_id", "ts"]),
    }
    f["encontros"]["duracao_s"] = f["encontros"]["duracao_s"].astype(int)
    f["encontros"]["rssi_medio"] = f["encontros"]["rssi_medio"].astype(float)
    f["avaliacoes"]["nota"] = f["avaliacoes"]["nota"].astype(int)
    return _padronizar(f)


def carregar_de_arquivos(pasta_bootstrap: Path) -> dict[str, pd.DataFrame]:
    app, pul = Path(pasta_bootstrap) / "historico_app", Path(pasta_bootstrap) / "pulseira"
    usuarios = pd.read_csv(app / "usuarios.csv")
    resolv = C.ResolvedorTokens(usuarios.usuario_id.tolist())
    dono = dict(zip(usuarios.pulseira_id, usuarios.usuario_id))

    enc = pd.read_csv(pul / "encontros.csv")
    ini = pd.to_datetime(enc.inicio, format="ISO8601")
    enc["usuario_a"] = [resolv.resolver(t, d.to_pydatetime()) for t, d in zip(enc.usuario_a, ini)]
    enc["usuario_b"] = [resolv.resolver(t, d.to_pydatetime()) for t, d in zip(enc.usuario_b, ini)]
    st = pd.read_csv(pul / "status.csv")
    st["usuario_id"] = st.pulseira_id.map(dono)
    it = pd.read_csv(pul / "interesses.csv")
    it["usuario_id"] = it.pulseira_id.map(dono)
    f = {
        "usuarios": usuarios,
        "encontros": enc,
        "avaliacoes": pd.read_csv(app / "avaliacoes.csv"),
        "obras": pd.read_csv(app / "obras.csv"),
        "seguidores": pd.read_csv(app / "seguidores.csv"),
        "interacoes": pd.read_csv(app / "interacoes.csv").drop(columns=["avaliacao_id"], errors="ignore"),
        "status": st[["pulseira_id", "usuario_id", "modo", "ts"]],
        "interesses": it[["encontro_id", "pulseira_id", "usuario_id", "ts"]],
    }
    return _padronizar(f)


# ------------------------------------------------------------- checagens
def checar_fontes(f: dict[str, pd.DataFrame]) -> dict:
    """Problemas de consistência que impedem o pipeline de seguir (levanta erro) + estatísticas."""
    e, u = f["encontros"], set(f["usuarios"].usuario_id)
    erros = []
    if e.empty:
        erros.append("nenhum encontro no banco — a ingestão MQTT rodou?")
    if e.encontro_id.duplicated().any():
        erros.append(f"{e.encontro_id.duplicated().sum()} encontro_id duplicados")
    desconhecidos = (~e.usuario_a.isin(u) | ~e.usuario_b.isin(u)).sum()
    if desconhecidos:
        erros.append(f"{desconhecidos} encontros com usuário fora da tabela usuarios")
    if (e.usuario_a == e.usuario_b).any():
        erros.append("encontros de um usuário com ele mesmo")
    if f["avaliacoes"].empty:
        erros.append("nenhuma avaliação — histórico do app não foi carregado?")
    if erros:
        raise ValueError("Ingestão inconsistente: " + "; ".join(erros))
    return {
        "linhas": {k: int(len(v)) for k, v in f.items()},
        "encontros_de": str(e.inicio.min()), "encontros_ate": str(e.inicio.max()),
        "interacoes_ate": str(f["interacoes"].criado_em.max()),
    }


# --------------------------------------------------------------- staging
def salvar_staging(f: dict[str, pd.DataFrame], pasta: Path = PASTA_STAGING, origem: str = "") -> dict:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    resumo = checar_fontes(f)
    for nome in TABELAS:
        f[nome].to_parquet(pasta / f"{nome}.parquet", index=False)
    resumo.update({"origem": origem, "extraido_em": datetime.now().isoformat(timespec="seconds")})
    (pasta / "_resumo_ingestao.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    return resumo


def ler_staging(pasta: Path = PASTA_STAGING) -> dict[str, pd.DataFrame]:
    return {nome: pd.read_parquet(Path(pasta) / f"{nome}.parquet") for nome in TABELAS}


def executar(pasta: Path = PASTA_STAGING, fonte: str = "banco", pasta_bootstrap: Path | None = None) -> dict:
    if fonte == "banco":
        from opinion.db import Banco

        b = Banco()
        f, origem = carregar_do_banco(b), b.url
    else:
        from opinion.mundo import PASTA_BOOTSTRAP

        origem = str(pasta_bootstrap or PASTA_BOOTSTRAP)
        f = carregar_de_arquivos(Path(origem))
    return salvar_staging(f, pasta, origem)
