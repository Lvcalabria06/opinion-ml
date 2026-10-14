"""
Serviço de ingestão (CC1): recebe as mensagens da pulseira pelo MQTT e grava no banco.

Para cada mensagem em opinion/pulseira/+/encontro | interesse | status:
  1. guarda a mensagem bruta em `mqtt_mensagens` (auditoria, inclusive as rejeitadas)
  2. valida contra o contrato (opinion/contrato.py)
  3. traduz os IDs anônimos e rotativos para o usuário real
  4. confere se a pulseira que publicou pertence mesmo ao usuário informado
  5. grava em `encontros`, `interesses` ou `status_pulseira`
     (reentregas do QoS 1 com o mesmo encontro_id são ignoradas)

    python -m opinion.ingestao
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta

import paho.mqtt.client as mqtt

from opinion import contrato as C
from opinion.db import Banco, agora

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [ingestao] %(message)s")
log = logging.getLogger(__name__)


class Ingestao:
    def __init__(self, banco: Banco):
        self.banco = banco
        banco.criar_esquema()
        usuarios = banco.todos("SELECT usuario_id, pulseira_id FROM usuarios")
        if not usuarios:
            sys.exit("Tabela usuarios vazia — rode antes: python -m opinion.carregar_historico_app")
        self.dono = {p: u for u, p in usuarios}
        self.tokens = C.ResolvedorTokens([u for u, _ in usuarios])
        self.stats = {"recebidas": 0, "gravadas": 0, "rejeitadas": 0, "duplicadas": 0}
        self.cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="opinion-ingestao", clean_session=False)
        self.cliente.on_connect = self.ao_conectar
        self.cliente.on_message = self.ao_receber

    def ao_conectar(self, cliente, _ud, _flags, rc, _props=None):
        if rc != 0:
            log.error("Falha ao conectar no broker (%s)", rc)
            return
        cliente.subscribe([(C.filtro(t), C.QOS) for t in ("encontro", "interesse", "status")])
        log.info("Conectado em %s:%s — aguardando mensagens em %s/+/{encontro,interesse,status}",
                 C.BROKER_HOST, C.BROKER_PORTA, C.PREFIXO)

    # ----------------------------------------------------------------- mensagem
    def ao_receber(self, _c, _ud, msg):
        self.stats["recebidas"] += 1
        partes = msg.topic.split("/")
        pulseira, tipo = (partes[2], partes[3]) if len(partes) == 4 else (None, None)
        texto = msg.payload.decode("utf-8", "replace")
        try:
            p = json.loads(texto)
            erro = C.validar(tipo, p)
            if erro is None:
                erro = self.gravar(tipo, pulseira, p)
        except json.JSONDecodeError:
            erro = "payload não é JSON"
        except Exception as ex:  # nunca derrubar o serviço por uma mensagem
            log.exception("Erro inesperado em %s", msg.topic)
            erro = f"erro interno: {ex}"

        valida = erro is None or erro == "duplicada"
        self.banco.executar(
            "INSERT INTO mqtt_mensagens (topico, tipo, pulseira_id, payload, valida, erro, recebido_em) VALUES (?,?,?,?,?,?,?)",
            (msg.topic, tipo, pulseira, texto, int(valida), erro, agora()))
        if erro == "duplicada":
            self.stats["duplicadas"] += 1
        elif erro:
            self.stats["rejeitadas"] += 1
            log.warning("Rejeitada (%s): %s", erro, msg.topic)
        else:
            self.stats["gravadas"] += 1
        if self.stats["recebidas"] % 5000 == 0:
            log.info("Progresso: %s", self.stats)

    def _usuario(self, token: str, quando: datetime) -> str | None:
        return self.tokens.resolver(token, quando)

    def gravar(self, tipo: str, pulseira: str, p: dict) -> str | None:
        if pulseira not in self.dono:
            return "pulseira não cadastrada"

        if tipo == "encontro":
            inicio = datetime.fromisoformat(p["inicio"])
            ua, ub = self._usuario(p["usuario_a"], inicio), self._usuario(p["usuario_b"], inicio)
            if ua is None or ub is None:
                return "ID rotativo não reconhecido"
            if ua != self.dono[pulseira]:
                return "usuario_a não é o dono da pulseira que publicou"
            fim = inicio + timedelta(seconds=float(p["duracao_s"]))
            cur = self.banco.executar(
                "INSERT INTO encontros (encontro_id, pulseira_id, token_a, token_b, usuario_a, usuario_b, inicio, fim, "
                "duracao_s, rssi_medio, local, recebido_em) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                (p["encontro_id"], pulseira, p["usuario_a"], p["usuario_b"], ua, ub, inicio.isoformat(),
                 fim.isoformat(timespec="seconds"), int(p["duracao_s"]), float(p["rssi_medio"]), p["local"], agora()))
            return "duplicada" if cur.rowcount == 0 else None

        quando = datetime.fromisoformat(p["ts"])
        u = self._usuario(p["usuario"], quando)
        if u is None:
            return "ID rotativo não reconhecido"
        if u != self.dono[pulseira]:
            return "usuario não é o dono da pulseira que publicou"
        if tipo == "interesse":
            cur = self.banco.executar(
                "INSERT INTO interesses (encontro_id, pulseira_id, usuario_id, ts, recebido_em) VALUES (?,?,?,?,?) "
                "ON CONFLICT DO NOTHING", (p["encontro_id"], pulseira, u, quando.isoformat(), agora()))
        else:
            cur = self.banco.executar(
                "INSERT INTO status_pulseira (pulseira_id, usuario_id, modo, ts, recebido_em) VALUES (?,?,?,?,?) "
                "ON CONFLICT DO NOTHING", (pulseira, u, p["modo"], quando.isoformat(), agora()))
        return "duplicada" if cur.rowcount == 0 else None

    # ---------------------------------------------------------------------- run
    def rodar(self):
        for tentativa in range(30):
            try:
                self.cliente.connect(C.BROKER_HOST, C.BROKER_PORTA, keepalive=30)
                break
            except OSError:
                log.info("Broker indisponível em %s:%s, tentando de novo (%d)...", C.BROKER_HOST, C.BROKER_PORTA, tentativa + 1)
                time.sleep(2)
        else:
            sys.exit("Não foi possível conectar ao broker MQTT.")

        def parar(*_):
            log.info("Encerrando. Totais: %s", self.stats)
            self.cliente.disconnect()

        signal.signal(signal.SIGINT, parar)
        signal.signal(signal.SIGTERM, parar)
        self.cliente.loop_forever()


if __name__ == "__main__":
    Ingestao(Banco()).rodar()
