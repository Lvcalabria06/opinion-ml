"""
CHECKPOINT 1 (semana 10): "Um encontro publicado pela pulseira virtual é recebido pelo
broker, persistido no banco e aparece em uma consulta simples."

O teste:
  1. escolhe dois usuários cadastrados e monta os IDs rotativos de hoje
  2. publica UM encontro novo em opinion/pulseira/{id}/encontro
  3. espera o serviço de ingestão gravar e procura o encontro no banco
  4. mostra a linha encontrada -> CHECKPOINT 1 OK

Pré-requisito: broker MQTT e `python -m opinion.ingestao` rodando.

    python -m opinion.checkpoint1
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timedelta

import paho.mqtt.client as mqtt

from opinion import contrato as C
from opinion.db import Banco


def main():
    banco = Banco()
    banco.criar_esquema()
    dois = banco.todos("SELECT usuario_id, pulseira_id FROM usuarios ORDER BY usuario_id LIMIT 2")
    if len(dois) < 2:
        sys.exit("Sem usuários no banco. Rode antes: python -m opinion.carregar_historico_app")
    (ua, pa), (ub, _) = dois
    agora = datetime.now().replace(microsecond=0)
    inicio = agora - timedelta(seconds=340)
    eid = f"e-checkpoint-{uuid.uuid4().hex[:6]}"
    payload = {"v": C.VERSAO, "encontro_id": eid, "usuario_a": C.token_do_dia(ua, inicio.date()),
               "usuario_b": C.token_do_dia(ub, inicio.date()), "inicio": inicio.isoformat(),
               "duracao_s": 340, "rssi_medio": -61.0, "local": "cinema"}

    print("1) Publicando encontro de teste")
    print(f"   tópico : {C.topico(pa, 'encontro')}")
    print(f"   payload: {json.dumps(payload, ensure_ascii=False)}")
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"opinion-checkpoint-{uuid.uuid4().hex[:4]}")
    try:
        c.connect(C.BROKER_HOST, C.BROKER_PORTA, keepalive=10)
    except OSError:
        sys.exit(f"   FALHOU: broker MQTT não está acessível em {C.BROKER_HOST}:{C.BROKER_PORTA}")
    c.loop_start()
    info = c.publish(C.topico(pa, "encontro"), json.dumps(payload), qos=C.QOS)
    info.wait_for_publish(timeout=10)
    print(f"2) Broker confirmou o recebimento (PUBACK, QoS {C.QOS}): {'sim' if info.is_published() else 'NÃO'}")
    c.loop_stop(); c.disconnect()

    print("3) Procurando o encontro no banco", end="", flush=True)
    linha = None
    for _ in range(20):
        linha = banco.um("SELECT encontro_id, usuario_a, usuario_b, inicio, fim, duracao_s, rssi_medio, local, recebido_em "
                         "FROM encontros WHERE encontro_id = ?", (eid,))
        if linha:
            break
        print(".", end="", flush=True)
        time.sleep(0.5)
    print()
    if not linha:
        erro = banco.um("SELECT erro FROM mqtt_mensagens WHERE payload LIKE ? ORDER BY id DESC LIMIT 1", (f"%{eid}%",))
        motivo = f"mensagem chegou mas foi rejeitada: {erro[0]}" if erro else "o serviço de ingestão está rodando?"
        sys.exit(f"   FALHOU: encontro não apareceu no banco ({motivo})")
    cols = ["encontro_id", "usuario_a", "usuario_b", "inicio", "fim", "duracao_s", "rssi_medio", "local", "recebido_em"]
    print("4) Encontrado com a consulta: SELECT * FROM encontros WHERE encontro_id = '" + eid + "'")
    for k, v in zip(cols, linha):
        print(f"   {k:<12} {v}")
    print(f"   (os IDs rotativos foram traduzidos para os usuários reais: {ua} e {ub})")
    print("\nCHECKPOINT 1 OK ✔")


if __name__ == "__main__":
    main()
