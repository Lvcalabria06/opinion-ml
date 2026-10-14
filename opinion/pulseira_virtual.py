"""
Pulseira virtual do OpiniOn — publica encontros, toques de interesse e status via MQTT,
seguindo o contrato da seção 4.2 (opinion/contrato.py).

A equipe de CC trabalha com a pulseira virtual do Sprint 1 ao SR2; o protótipo físico
é do Design e, quando conectado, usa o mesmo contrato.

Modos
  replay   publica as 12 semanas do bootstrap (data/bootstrap/pulseira/) em ordem
           cronológica. É assim que o dataset entra no banco: pelo mesmo caminho
           MQTT -> ingestão que os dados reais usariam.
  ao-vivo  gera encontros novos agora, continuamente (demonstração / checkpoints).

Para simplificar, um único cliente MQTT publica em nome de todas as pulseiras
virtuais (cada uma no seu próprio tópico opinion/pulseira/{id}/...).
Também escuta opinion/pulseira/+/feedback e mostra os sinais recebidos (CC7 em diante).

    python -m opinion.pulseira_virtual replay                 # ~500 mensagens/s (≈ 2 min)
    python -m opinion.pulseira_virtual replay --velocidade 3600   # 1 h simulada por segundo
    python -m opinion.pulseira_virtual ao-vivo --por-minuto 30
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from opinion import contrato as C
from opinion.mundo import PASTA_BOOTSTRAP, atributos_encontro, carregar_config

logging.basicConfig(level="INFO", format="%(asctime)s [pulseira] %(message)s")
log = logging.getLogger(__name__)


class Publicador:
    def __init__(self, duplicar: float = 0.0, invalidas: float = 0.0):
        self.duplicar, self.invalidas = duplicar, invalidas
        self.enviadas = {"encontro": 0, "interesse": 0, "status": 0}
        self.feedbacks = 0
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"opinion-pulseira-virtual-{uuid.uuid4().hex[:6]}")
        self.c.max_inflight_messages_set(200)
        self.c.on_message = self._feedback
        for tentativa in range(30):
            try:
                self.c.connect(C.BROKER_HOST, C.BROKER_PORTA, keepalive=30)
                break
            except OSError:
                log.info("Aguardando broker em %s:%s (%d)...", C.BROKER_HOST, C.BROKER_PORTA, tentativa + 1)
                time.sleep(2)
        else:
            sys.exit("Broker MQTT indisponível.")
        self.c.subscribe(C.filtro("feedback"), qos=C.QOS)
        self.c.loop_start()
        self._ultimo = None

    def _feedback(self, _c, _u, msg):
        if not msg.topic.endswith("/feedback"):
            return
        self.feedbacks += 1
        p = json.loads(msg.payload)
        log.info("Feedback em %s: nível=%s padrão=%s", msg.topic.split("/")[2], p.get("nivel"), p.get("padrao"))

    def publicar(self, pulseira_id: str, tipo: str, payload: dict):
        if random.random() < self.invalidas:  # falha injetada: campo obrigatório some
            payload = {k: v for k, v in payload.items() if k not in ("rssi_medio", "modo", "ts")}
        texto = json.dumps(payload, ensure_ascii=False)
        self._ultimo = self.c.publish(C.topico(pulseira_id, tipo), texto, qos=C.QOS)
        if random.random() < self.duplicar:  # falha injetada: reentrega
            self._ultimo = self.c.publish(C.topico(pulseira_id, tipo), texto, qos=C.QOS)
        self.enviadas[tipo] += 1

    def fechar(self):
        if self._ultimo is not None:
            self._ultimo.wait_for_publish(timeout=60)
        time.sleep(1)
        self.c.loop_stop()
        self.c.disconnect()


# ======================================================================= replay
def carregar_fila(pasta) -> list[tuple[datetime, str, str, dict]]:
    """Junta encontros, interesses e status do bootstrap numa fila cronológica."""
    fila = []
    for r in pd.read_csv(pasta / "status.csv").itertuples():
        fila.append((datetime.fromisoformat(r.ts), r.pulseira_id, "status",
                     {"v": C.VERSAO, "usuario": r.usuario, "modo": r.modo, "ts": r.ts}))
    for r in pd.read_csv(pasta / "encontros.csv").itertuples():
        ini = datetime.fromisoformat(r.inicio)
        # a pulseira só publica quando o encontro termina
        fila.append((ini + timedelta(seconds=int(r.duracao_s)), r.pulseira_id, "encontro", {
            "v": C.VERSAO, "encontro_id": r.encontro_id, "usuario_a": r.usuario_a, "usuario_b": r.usuario_b,
            "inicio": r.inicio, "duracao_s": int(r.duracao_s), "rssi_medio": float(r.rssi_medio), "local": r.local}))
    for r in pd.read_csv(pasta / "interesses.csv").itertuples():
        fila.append((datetime.fromisoformat(r.ts), r.pulseira_id, "interesse",
                     {"v": C.VERSAO, "encontro_id": r.encontro_id, "usuario": r.usuario, "ts": r.ts}))
    fila.sort(key=lambda x: x[0])
    return fila


def replay(a):
    fila = carregar_fila(PASTA_BOOTSTRAP / "pulseira")
    if a.limite:
        fila = fila[:a.limite]
    log.info("Replay de %d mensagens (%s a %s), velocidade %s", len(fila), fila[0][0], fila[-1][0],
             "máxima" if a.velocidade <= 0 else f"{a.velocidade:g}x")
    pub = Publicador(a.duplicar, a.invalidas)
    t_sim0, t_real0 = fila[0][0], time.monotonic()
    for i, (t, pulseira, tipo, payload) in enumerate(fila, 1):
        if a.velocidade > 0:
            espera = (t - t_sim0).total_seconds() / a.velocidade - (time.monotonic() - t_real0)
        else:
            espera = 0.0
        # teto de vazão: o broker guarda uma fila limitada por assinante; se a ingestão
        # não acompanhar, mensagens excedentes seriam descartadas
        espera = max(espera, i / a.max_por_segundo - (time.monotonic() - t_real0))
        if espera > 0:
            time.sleep(espera)
        pub.publicar(pulseira, tipo, payload)
        if i % 5000 == 0 or i == len(fila):
            log.info("%d/%d publicadas (relógio simulado: %s)", i, len(fila), t.strftime("%d/%m %H:%M"))
    pub.fechar()
    log.info("Fim do replay: %s", pub.enviadas)


# ======================================================================= ao vivo
PESOS_HORA = {  # que tipo de local é mais provável em cada faixa de horário
    "manha": {"campus": 0.75, "livraria": 0.25},
    "tarde": {"campus": 0.5, "livraria": 0.25, "cinema": 0.25},
    "noite": {"cinema": 0.6, "livraria": 0.15, "campus": 0.25},
    "madrugada": {"cinema": 1.0},
}


def ao_vivo(a):
    cfg = carregar_config()
    rng = np.random.default_rng(a.semente)
    random.seed(a.semente)
    usuarios = pd.read_csv(PASTA_BOOTSTRAP / "historico_app" / "usuarios.csv")
    oculto = pd.read_csv(PASTA_BOOTSTRAP / "verdade_simulacao_NAO_USAR_COMO_FEATURE" / "usuarios_ocultos.csv")
    u = usuarios.merge(oculto, on="usuario_id")
    status = pd.read_csv(PASTA_BOOTSTRAP / "pulseira" / "status.csv").groupby("pulseira_id").modo.last()
    u["modo"] = u.pulseira_id.map(status).fillna("aberto")
    pub = Publicador(a.duplicar, a.invalidas)
    intervalo = 60.0 / a.por_minuto
    log.info("Modo ao vivo: ~%g encontros/min entre %d usuários. Ctrl+C para parar.", a.por_minuto, len(u))
    fim = time.monotonic() + a.duracao_min * 60 if a.duracao_min else None
    try:
        while fim is None or time.monotonic() < fim:
            agora = datetime.now().replace(microsecond=0)
            h = agora.hour
            faixa = "madrugada" if h < 6 else "manha" if h < 12 else "tarde" if h < 18 else "noite"
            pesos = PESOS_HORA[faixa] if not a.contexto else {a.contexto: 1.0}
            local = str(rng.choice(list(pesos), p=list(pesos.values())))
            # quem frequenta esse tipo de local é mais provável de estar lá
            vis = [cfg["personas"][p]["visitas_semana"].get(local, cfg["personas"][p]["convencao"]) for p in u.persona]
            vis = np.array(vis) * (u.modo != "pausado").to_numpy()
            ia, ib = rng.choice(len(u), size=2, replace=False, p=vis / vis.sum())
            dur, rssi = atributos_encontro(rng, cfg["locais"][local], False, cfg["amigos"])
            inicio = agora - timedelta(seconds=dur)
            pa, pb = u.iloc[ia], u.iloc[ib]
            eid = f"e-live-{uuid.uuid4().hex[:8]}"
            pub.publicar(pa.pulseira_id, "encontro", {
                "v": C.VERSAO, "encontro_id": eid,
                "usuario_a": C.token_do_dia(pa.usuario_id, inicio.date()),
                "usuario_b": C.token_do_dia(pb.usuario_id, inicio.date()),
                "inicio": inicio.isoformat(), "duracao_s": dur, "rssi_medio": rssi, "local": local})
            log.info("Encontro %s: %s x %s em %s (%d s, %.0f dBm)", eid, pa.pulseira_id, pb.pulseira_id, local, dur, rssi)
            for p in (pa, pb):
                if rng.random() < 0.05 + 0.1 * p.abertura_social:
                    pub.publicar(p.pulseira_id, "interesse", {"v": C.VERSAO, "encontro_id": eid,
                                                              "usuario": C.token_do_dia(p.usuario_id, agora.date()),
                                                              "ts": agora.isoformat()})
            if rng.random() < 0.05:  # alguém troca o modo de descoberta
                k = int(rng.integers(len(u)))
                novo = str(rng.choice([m for m in C.MODOS if m != u.at[k, "modo"]]))
                u.at[k, "modo"] = novo
                pub.publicar(u.at[k, "pulseira_id"], "status", {"v": C.VERSAO, "modo": novo, "ts": agora.isoformat(),
                                                                "usuario": C.token_do_dia(u.at[k, "usuario_id"], agora.date())})
            time.sleep(intervalo * rng.uniform(0.5, 1.5))
    except KeyboardInterrupt:
        pass
    finally:
        pub.fechar()
        log.info("Encerrado: %s | feedbacks recebidos: %d", pub.enviadas, pub.feedbacks)


def main():
    ap = argparse.ArgumentParser(description="Pulseira virtual do OpiniOn (MQTT)")
    sub = ap.add_subparsers(dest="modo", required=True)
    r = sub.add_parser("replay", help="publica as 12 semanas do bootstrap")
    r.add_argument("--velocidade", type=float, default=0, help="segundos simulados por segundo real (0 = máximo)")
    r.add_argument("--limite", type=int, help="publica só as N primeiras mensagens")
    r.add_argument("--max-por-segundo", type=float, default=500,
                   help="teto de mensagens/s para a ingestão acompanhar (padrão 500)")
    v = sub.add_parser("ao-vivo", help="gera encontros agora, continuamente")
    v.add_argument("--por-minuto", type=float, default=20)
    v.add_argument("--duracao-min", type=float, default=0, help="0 = até Ctrl+C")
    v.add_argument("--contexto", choices=C.LOCAIS, help="força um tipo de local (ex.: convencao_games)")
    v.add_argument("--semente", type=int, default=7)
    for p in (r, v):
        p.add_argument("--duplicar", type=float, default=0.0, help="fração reenviada (testa deduplicação)")
        p.add_argument("--invalidas", type=float, default=0.0, help="fração corrompida (testa validação)")
    a = ap.parse_args()
    replay(a) if a.modo == "replay" else ao_vivo(a)


if __name__ == "__main__":
    main()
