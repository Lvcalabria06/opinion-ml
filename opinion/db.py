"""
Banco de dados do OpiniOn (CC1).

PostgreSQL no Docker Compose; SQLite para rodar local sem Docker.
A escolha vem de DATABASE_URL:
    postgresql://<usuario>:<senha>@localhost:5432/<base>
    sqlite:///data/opinion.db          (padrão)

Dois grupos de tabelas:
  * histórico do app  — o que o aplicativo já guarda (usuários, obras, avaliações,
                        seguidores, interações). Carregado do bootstrap.
  * pulseira          — o que chega pelo MQTT (encontros, interesses, status),
                        gravado pelo serviço de ingestão.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path

from opinion.mundo import RAIZ

URL_PADRAO = f"sqlite:///{RAIZ / 'data' / 'opinion.db'}"

DDL = """
-- ============================== histórico do app
CREATE TABLE IF NOT EXISTS usuarios (
    usuario_id    TEXT PRIMARY KEY,
    pulseira_id   TEXT UNIQUE,
    data_cadastro DATE
);
CREATE TABLE IF NOT EXISTS obras (
    obra_id      TEXT PRIMARY KEY,
    titulo       TEXT,
    midia        TEXT,
    generos      TEXT,
    popularidade REAL
);
CREATE TABLE IF NOT EXISTS avaliacoes (
    avaliacao_id TEXT PRIMARY KEY,
    usuario_id   TEXT REFERENCES usuarios(usuario_id),
    obra_id      TEXT REFERENCES obras(obra_id),
    nota         INTEGER CHECK (nota BETWEEN 1 AND 5),
    criado_em    {TS}
);
CREATE INDEX IF NOT EXISTS ix_avaliacoes_usuario ON avaliacoes(usuario_id, criado_em);
CREATE TABLE IF NOT EXISTS seguidores (
    seguidor TEXT REFERENCES usuarios(usuario_id),
    seguido  TEXT REFERENCES usuarios(usuario_id),
    desde    {TS},
    PRIMARY KEY (seguidor, seguido)
);
CREATE TABLE IF NOT EXISTS interacoes (
    interacao_id TEXT PRIMARY KEY,
    origem       TEXT REFERENCES usuarios(usuario_id),
    destino      TEXT REFERENCES usuarios(usuario_id),
    tipo         TEXT,
    obra_id      TEXT,
    avaliacao_id TEXT,
    criado_em    {TS}
);
CREATE INDEX IF NOT EXISTS ix_interacoes_par ON interacoes(origem, destino, criado_em);

-- ============================== pulseira (via MQTT)
CREATE TABLE IF NOT EXISTS mqtt_mensagens (
    id          {SERIAL},
    topico      TEXT NOT NULL,
    tipo        TEXT,
    pulseira_id TEXT,
    payload     TEXT,
    valida      INTEGER,
    erro        TEXT,
    recebido_em {TS}
);
CREATE TABLE IF NOT EXISTS encontros (
    encontro_id TEXT PRIMARY KEY,
    pulseira_id TEXT NOT NULL,
    token_a     TEXT NOT NULL,
    token_b     TEXT NOT NULL,
    usuario_a   TEXT REFERENCES usuarios(usuario_id),
    usuario_b   TEXT REFERENCES usuarios(usuario_id),
    inicio      {TS} NOT NULL,
    fim         {TS} NOT NULL,
    duracao_s   INTEGER,
    rssi_medio  REAL,
    local       TEXT,
    recebido_em {TS}
);
CREATE INDEX IF NOT EXISTS ix_encontros_inicio ON encontros(inicio);
CREATE INDEX IF NOT EXISTS ix_encontros_par ON encontros(usuario_a, usuario_b);
CREATE TABLE IF NOT EXISTS interesses (
    id          {SERIAL},
    encontro_id TEXT,
    pulseira_id TEXT,
    usuario_id  TEXT REFERENCES usuarios(usuario_id),
    ts          {TS},
    recebido_em {TS},
    UNIQUE (encontro_id, usuario_id)
);
CREATE TABLE IF NOT EXISTS status_pulseira (
    id          {SERIAL},
    pulseira_id TEXT,
    usuario_id  TEXT REFERENCES usuarios(usuario_id),
    modo        TEXT,
    ts          {TS},
    recebido_em {TS},
    UNIQUE (pulseira_id, ts)
);
"""


def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Banco:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("DATABASE_URL", URL_PADRAO)
        self.postgres = self.url.startswith("postgres")
        if self.postgres:
            import psycopg

            self.conn = psycopg.connect(self.url, autocommit=True)
        else:
            caminho = self.url.replace("sqlite:///", "")
            Path(caminho).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(caminho, check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

    def criar_esquema(self):
        ddl = DDL.format(SERIAL="SERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT",
                         TS="TIMESTAMP" if self.postgres else "TEXT")
        for stmt in ddl.split(";"):
            if "CREATE" in stmt:  # trechos só com comentário são ignorados
                self.conn.execute(stmt)

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    def executar(self, sql: str, params: tuple = ()):
        return self.conn.execute(self._sql(sql), params)

    def executar_varios(self, sql: str, linhas: list[tuple], lote: int = 5000):
        for i in range(0, len(linhas), lote):
            parte = linhas[i:i + lote]
            if self.postgres:
                with self.conn.transaction(), self.conn.cursor() as cur:
                    cur.executemany(self._sql(sql), parte)
            else:
                self.conn.execute("BEGIN")
                self.conn.executemany(sql, parte)
                self.conn.execute("COMMIT")

    def um(self, sql: str, params: tuple = ()):
        return self.executar(sql, params).fetchone()

    def todos(self, sql: str, params: tuple = ()):
        return self.executar(sql, params).fetchall()
