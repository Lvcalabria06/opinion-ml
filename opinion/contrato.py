"""
Contrato de mensagens MQTT da pulseira (seção 4.2 do documento do projeto).

Tópicos
  opinion/pulseira/{id}/encontro    pulseira -> backend   encontro detectado
  opinion/pulseira/{id}/interesse   pulseira -> backend   toque do usuário
  opinion/pulseira/{id}/status      pulseira -> backend   modo de descoberta
  opinion/pulseira/{id}/feedback    backend  -> pulseira  sinal a emitir (usado a partir do CC7)

IDs de usuário nas mensagens são ANÔNIMOS E ROTATIVOS: mudam todo dia e só o
backend (que conhece o segredo) consegue saber quem é quem.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from datetime import date, datetime

PREFIXO = "opinion/pulseira"
VERSAO = 1
QOS = 1

BROKER_HOST = os.getenv("MQTT_HOST", "localhost")
BROKER_PORTA = int(os.getenv("MQTT_PORT", "1883"))
_segredo = os.getenv("OPINION_SEGREDO_TOKEN")
if not _segredo:
    raise RuntimeError("Variável OPINION_SEGREDO_TOKEN não definida. Crie o arquivo .env: cp .env.example .env")
SEGREDO_TOKEN = _segredo.encode()

LOCAIS = ("convencao_games", "cinema", "livraria", "campus")
MODOS = ("aberto", "discreto", "pausado")
NIVEIS_FEEDBACK = ("nenhum", "periferico_leve", "periferico_perceptivel")


def topico(pulseira_id: str, tipo: str) -> str:
    return f"{PREFIXO}/{pulseira_id}/{tipo}"


def filtro(tipo: str) -> str:
    return f"{PREFIXO}/+/{tipo}"


# ------------------------------------------------------- IDs rotativos
def token_do_dia(usuario_id: str, dia: date) -> str:
    """ID anônimo do usuário naquele dia: u- + 10 hex de um HMAC(segredo, usuario|dia)."""
    msg = f"{usuario_id}|{dia.isoformat()}".encode()
    return "u-" + hmac.new(SEGREDO_TOKEN, msg, hashlib.sha256).hexdigest()[:10]


class ResolvedorTokens:
    """Traduz token rotativo -> usuario_id, calculando os tokens do dia sob demanda."""

    def __init__(self, usuarios: list[str]):
        self.usuarios = list(usuarios)
        self._cache: dict[date, dict[str, str]] = {}

    def resolver(self, token: str, quando: datetime) -> str | None:
        for dia in (quando.date(), date.fromordinal(quando.date().toordinal() - 1)):  # tolera virada do dia
            if dia not in self._cache:
                self._cache[dia] = {token_do_dia(u, dia): u for u in self.usuarios}
                if len(self._cache) > 30:
                    self._cache.pop(next(iter(self._cache)))
            if token in self._cache[dia]:
                return self._cache[dia][token]
        return None


# ------------------------------------------------------------ validação
def _ts(v) -> datetime | None:
    try:
        return datetime.fromisoformat(v)
    except (TypeError, ValueError):
        return None


def validar(tipo: str, p: dict) -> str | None:
    """Retorna None se a mensagem está de acordo com o contrato, ou o motivo do erro."""
    if not isinstance(p, dict):
        return "payload não é um objeto JSON"
    if p.get("v") != VERSAO:
        return f"versão ausente ou não suportada: {p.get('v')}"
    if tipo == "encontro":
        for c in ("encontro_id", "usuario_a", "usuario_b", "inicio", "local"):
            if not isinstance(p.get(c), str) or not p[c]:
                return f"campo ausente ou inválido: {c}"
        if p["usuario_a"] == p["usuario_b"]:
            return "usuario_a igual a usuario_b"
        if _ts(p["inicio"]) is None:
            return "inicio não é data/hora ISO-8601"
        if not isinstance(p.get("duracao_s"), (int, float)) or isinstance(p.get("duracao_s"), bool) or p["duracao_s"] <= 0:
            return "duracao_s ausente ou inválida"
        if not isinstance(p.get("rssi_medio"), (int, float)) or not -110 <= p["rssi_medio"] <= 0:
            return "rssi_medio ausente ou fora da faixa (-110 a 0 dBm)"
        if p["local"] not in LOCAIS:
            return f"local desconhecido: {p['local']}"
    elif tipo == "interesse":
        for c in ("encontro_id", "usuario", "ts"):
            if not isinstance(p.get(c), str) or not p[c]:
                return f"campo ausente ou inválido: {c}"
        if _ts(p["ts"]) is None:
            return "ts não é data/hora ISO-8601"
    elif tipo == "status":
        if p.get("modo") not in MODOS:
            return f"modo inválido: {p.get('modo')}"
        if not isinstance(p.get("usuario"), str) or _ts(p.get("ts")) is None:
            return "usuario ou ts ausente/inválido"
    else:
        return f"tipo de mensagem desconhecido: {tipo}"
    return None
