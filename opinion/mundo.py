"""
Peças do "mundo simulado" usadas tanto pelo gerador do bootstrap quanto pela
pulseira virtual no modo ao vivo.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG_PADRAO = RAIZ / "config" / "simulacao.yaml"
PASTA_BOOTSTRAP = RAIZ / "data" / "bootstrap"

GENEROS = [
    "acao", "aventura", "comedia", "drama", "terror", "suspense",
    "ficcao_cientifica", "fantasia", "romance", "documental", "indie", "estrategia",
]

# modelo de rádio usado para gerar o RSSI que a pulseira virtual reporta
TX_1M, N_PERDA, SIGMA_RSSI = -59.0, 2.2, 3.0


def carregar_config(caminho: str | Path = CONFIG_PADRAO) -> dict:
    return yaml.safe_load(Path(caminho).read_text(encoding="utf-8"))


def atributos_encontro(rng: np.random.Generator, cfg_local: dict, amigos: bool, cfg_amigos: dict) -> tuple[int, float]:
    """Duração (s) e RSSI médio (dBm) de um encontro."""
    mediana = cfg_local["duracao_mediana_s"] * (cfg_amigos["mult_duracao"] if amigos else 1)
    dur = float(np.clip(rng.lognormal(math.log(mediana), 0.8), 30, 4 * 3600))
    dist = float(np.clip(rng.lognormal(math.log(0.8 if amigos else 1.5), 0.35), 0.3, 4.0))
    rssi = TX_1M - 10 * N_PERDA * math.log10(dist) + rng.normal(0, SIGMA_RSSI)
    return int(round(dur)), round(float(np.clip(rssi, -95, -35)), 1)


def sigmoide(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))
