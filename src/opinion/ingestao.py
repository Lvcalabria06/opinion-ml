"""
Serviço de ingestão (CC1): recebe as mensagens da pulseira pelo MQTT e grava no banco.

Para cada mensagem em opinion/pulseira/+/proximidade | encontro | interesse | status:
  1. guarda a mensagem bruta em `mqtt_mensagens` (auditoria, inclusive as rejeitadas)
  2. valida contra o contrato (src/opinion/contrato.py)
  3. traduz os IDs anônimos e rotativos para o usuário real
  4. confere se a pulseira que publicou pertence mesmo ao usuário informado
  5. em proximidade, avalia cada categoria e publica feedback à pulseira publicadora
  6. grava encontros, interesses e status; deduplica reentregas do QoS 1

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
from opinion.pipeline.compatibilidade import motivos_do_sinal
from opinion.pipeline.executar import config
from opinion.sinal_categoria import afinidades, feedback, perfil

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [ingestao] %(message)s")
log = logging.getLogger(__name__)


class Ingestao:
    def __init__(self, banco: Banco):
        self.banco = banco
        self.politica = config()
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
        cliente.subscribe([(C.filtro(t), C.QOS) for t in ("proximidade", "encontro", "interesse", "status")])
        log.info("Conectado em %s:%s — aguardando mensagens em %s/+/{proximidade,encontro,interesse,status}",
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

    def _avisar(self, pulseira, encontro_id, ua, ub, inicio):
        perfis = []
        for usuario in (ua, ub):
            modo = self.banco.um("SELECT modo FROM status_pulseira WHERE usuario_id = ? AND ts <= ? "
                                 "ORDER BY ts DESC LIMIT 1", (usuario, inicio))
            if modo and modo[0] == "pausado":
                return
            notas = self.banco.todos("SELECT a.obra_id, o.midia, o.generos, a.nota FROM avaliacoes a "
                                     "JOIN obras o ON o.obra_id = a.obra_id "
                                     "WHERE a.usuario_id = ? AND a.criado_em < ?", (usuario, inicio))
            perfis.append(perfil(notas))
        midias = motivos_do_sinal(afinidades(*perfis, self.politica), self.politica)
        if not midias:
            return None
        cur = self.banco.executar("INSERT INTO avisos_proximidade (encontro_id, pulseira_id, midias, inicio) "
                                  "VALUES (?,?,?,?) ON CONFLICT DO NOTHING",
                                  (encontro_id, pulseira, "|".join(midias), inicio))
        if cur.rowcount == 0:
            return "duplicada"
        try:
            payload = feedback(encontro_id, midias, agora())
            envio = self.cliente.publish(C.topico(pulseira, "feedback"), json.dumps(payload, ensure_ascii=False),
                                         qos=C.QOS)
            if envio.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"Falha ao enfileirar feedback MQTT (rc={envio.rc})")
        except Exception:
            self.banco.executar("DELETE FROM avisos_proximidade WHERE encontro_id = ?", (encontro_id,))
            raise
        return None

    def gravar(self, tipo: str, pulseira: str, p: dict) -> str | None:
        if pulseira not in self.dono:
            return "pulseira não cadastrada"

        if tipo in ("proximidade", "encontro"):
            inicio = datetime.fromisoformat(p["inicio"])
            ua, ub = self._usuario(p["usuario_a"], inicio), self._usuario(p["usuario_b"], inicio)
            if ua is None or ub is None:
                return "ID rotativo não reconhecido"
            if ua != self.dono[pulseira]:
                return "usuario_a não é o dono da pulseira que publicou"
            if tipo == "proximidade":
                return self._avisar(pulseira, p["encontro_id"], ua, ub, inicio.isoformat())
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
