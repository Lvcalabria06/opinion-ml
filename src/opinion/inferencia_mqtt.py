"""
Consumidor de inferência (CC7): pontua automaticamente cada encontro novo que chega pelo MQTT.

    python -m opinion.inferencia_mqtt

Assina opinion/pulseira/+/encontro (o fim de cada encontro). Para cada mensagem:
  1. valida contra o contrato e traduz os IDs rotativos (como a ingestão);
  2. calcula as features com as funções do pipeline, só com dados anteriores ao início;
  3. estima a chance de amizade com o modelo de produção do MLflow;
  4. grava em `inferencias` (o dashboard e a API leem dali);
  5. se a chance passa do limiar, publica o sinal em opinion/pulseira/{id}/feedback
     (nunca para quem está com a pulseira em modo "pausado").

A ingestão (opinion.ingestao) continua gravando o encontro em paralelo: os dois serviços
assinam o mesmo tópico com client_ids diferentes.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time

import paho.mqtt.client as mqtt

from opinion import contrato as C
from opinion.db import Banco, agora
from opinion.inferencia import Inferencia, mensagem_feedback

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [inferencia] %(message)s")
log = logging.getLogger(__name__)

SQL_INSERIR = ("INSERT INTO inferencias (encontro_id, pulseira_id, usuario_a, usuario_b, inicio, fim, local, "
               "probabilidade, limiar, sinal, nivel, midias_aviso, compat_score, algum_segue, modelo, latencia_ms, "
               "feedback_enviado, origem, criado_em) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
               "ON CONFLICT DO NOTHING")


def gravar(banco: Banco, r: dict, enviado: bool, origem: str) -> bool:
    cur = banco.executar(SQL_INSERIR, (
        r["encontro_id"], r.get("pulseira_id"), r["usuario_a"], r["usuario_b"], r["inicio"], r["fim"], r["local"],
        r["probabilidade"], r["limiar"], int(r["sinal"]), r["nivel"], "|".join(r["midias_aviso"]), r["compat_score"],
        r["algum_segue"], r["modelo"], r["latencia_ms"], int(enviado), origem, agora()))
    return cur.rowcount != 0


def pausado(banco: Banco, usuario: str, ate: str) -> bool:
    m = banco.um("SELECT modo FROM status_pulseira WHERE usuario_id = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
                 (usuario, ate))
    return bool(m and m[0] == "pausado")


class ConsumidorInferencia:
    def __init__(self, banco: Banco, inferencia: Inferencia):
        self.banco, self.inf = banco, inferencia
        banco.criar_esquema()
        self.stats = {"recebidas": 0, "pontuadas": 0, "sinais": 0, "rejeitadas": 0, "duplicadas": 0}
        self.cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="opinion-inferencia", clean_session=False)
        self.cliente.on_connect = self.ao_conectar
        self.cliente.on_message = self.ao_receber

    def ao_conectar(self, cliente, _ud, _flags, rc, _props=None):
        if rc != 0:
            log.error("Falha ao conectar no broker (%s)", rc)
            return
        cliente.subscribe(C.filtro("encontro"), C.QOS)
        log.info("Conectado em %s:%s — pontuando %s", C.BROKER_HOST, C.BROKER_PORTA, C.filtro("encontro"))

    def processar(self, topico: str, texto: str) -> dict | None:
        partes = topico.split("/")
        pulseira = partes[2] if len(partes) == 4 else None
        p = json.loads(texto)
        erro = C.validar("encontro", p)
        if erro:
            raise ValueError(erro)
        if self.banco.um("SELECT 1 FROM inferencias WHERE encontro_id = ?", (p["encontro_id"],)):
            self.stats["duplicadas"] += 1
            return None
        enc = self.inf.traduzir(p)
        dono = self.inf.dados()["usuarios"].set_index("usuario_id").pulseira_id.get(enc["usuario_a"])
        if dono != pulseira:
            raise ValueError("usuario_a não é o dono da pulseira que publicou")
        enc["pulseira_id"] = pulseira
        r = self.inf.prever(enc)
        enviar = r["sinal"] and not pausado(self.banco, r["usuario_a"], r["fim"])
        if not gravar(self.banco, r, enviar, "mqtt"):
            self.stats["duplicadas"] += 1
            return None
        if enviar:
            self.cliente.publish(C.topico(pulseira, "feedback"), json.dumps(mensagem_feedback(r), ensure_ascii=False),
                                 qos=C.QOS)
            self.stats["sinais"] += 1
        self.stats["pontuadas"] += 1
        log.info("%s %s×%s em %s: chance %.1f%% (limiar %.1f%%) → %s [%s ms]", r["encontro_id"], r["usuario_a"],
                 r["usuario_b"], r["local"], 100 * r["probabilidade"], 100 * r["limiar"],
                 r["nivel"] if enviar else "sem sinal", r["latencia_ms"])
        return r

    def ao_receber(self, _c, _ud, msg):
        self.stats["recebidas"] += 1
        try:
            self.processar(msg.topic, msg.payload.decode("utf-8", "replace"))
        except Exception as ex:      # uma mensagem ruim nunca derruba o serviço
            self.stats["rejeitadas"] += 1
            log.warning("Não pontuada (%s): %s", ex, msg.topic)
        if self.stats["recebidas"] % 100 == 0:
            log.info("Progresso: %s", self.stats)

    def rodar(self):
        self.inf.modelo()            # falha cedo se não houver modelo registrado
        self.inf.dados()
        for tentativa in range(30):
            try:
                self.cliente.connect(C.BROKER_HOST, C.BROKER_PORTA, keepalive=30)
                break
            except OSError:
                log.info("Broker indisponível, tentando de novo (%d)...", tentativa + 1)
                time.sleep(2)
        else:
            sys.exit("Não foi possível conectar ao broker MQTT.")

        def parar(*_):
            log.info("Encerrando. Totais: %s", self.stats)
            self.cliente.disconnect()

        signal.signal(signal.SIGINT, parar)
        signal.signal(signal.SIGTERM, parar)
        self.cliente.loop_forever()


def main():
    espera = int(os.getenv("OPINION_ESPERA_MODELO_S", "0"))
    inf = Inferencia()
    for tentativa in range(max(1, espera // 15)):
        try:
            inf.modelo()
            break
        except RuntimeError as ex:
            if espera <= 0:
                sys.exit(str(ex))
            log.info("Aguardando um modelo registrado no MLflow (%s)...", ex)
            time.sleep(15)
    ConsumidorInferencia(Banco(), inf).rodar()


if __name__ == "__main__":
    main()
