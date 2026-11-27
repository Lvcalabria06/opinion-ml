"""
Inferência sobre encontros novos (CC7): o modelo validado integrado ao pipeline da solução.

`Inferencia` junta três coisas:
  * os dados do app e da pulseira (banco operacional, ou o staging do pipeline), recarregados
    periodicamente para incluir encontros, avaliações e interações recentes;
  * o modelo de produção do MLflow (`opinion-producao@producao`; na falta dele, o otimizado do
    CC5 ou o vencedor do CC4), recarregado quando o alias passa a apontar para outra versão;
  * as MESMAS funções do pipeline do CC2 para calcular as features, usando só o que existia
    antes do início do encontro (`opinion.modelo.pontuar.matriz`).

É usada pela API (`api/main.py`) e pelo consumidor MQTT (`opinion.inferencia_mqtt`).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime

import pandas as pd

from opinion import contrato as C
from opinion.modelo.pontuar import matriz, traduzir
from opinion.modelo.treinar import config
from opinion.pipeline import PASTA_STAGING, ingestao
from opinion.pipeline.compatibilidade import MIDIAS, motivos_do_sinal
from opinion.pipeline.executar import config as config_pipeline

log = logging.getLogger(__name__)
NIVEIS = ("nenhum", "periferico_leve", "periferico_perceptivel")


def candidatos_modelo(cfg: dict) -> list[tuple[str, str]]:
    """Ordem de preferência: produção (Airflow) → otimizado (CC5) → melhor da comparação (CC4)."""
    return [(cfg["producao"]["modelo_registrado"], cfg["producao"]["alias"]),
            (cfg["otimizacao"]["modelo_registrado"], cfg["otimizacao"]["alias"]),
            (cfg["comparacao"]["modelo_registrado"], cfg["comparacao"]["alias"])]


class Inferencia:
    def __init__(self, fonte: str | None = None, recarregar_s: float | None = None, cfg: dict | None = None):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        self.cfg = cfg or config()
        self.cfg_p = config_pipeline()
        self.fonte = fonte or os.getenv("OPINION_FONTE_INFERENCIA", "banco")
        self.recarregar_s = float(recarregar_s if recarregar_s is not None else os.getenv("OPINION_RECARREGAR_S", 120))
        self.fator_perceptivel = float(self.cfg["producao"].get("faixa_perceptivel_fator", 1.5))
        self._lock = threading.RLock()
        self._f = None
        self._f_em = 0.0
        self._modelo = None
        self._info: dict = {}
        self._checado_em = 0.0
        self._resolvedor = None
        self._modelos_midia: dict = {}

    # ------------------------------------------------------------------ dados
    def dados(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            if self._f is None or time.time() - self._f_em > self.recarregar_s:
                t0 = time.time()
                if self.fonte == "banco":
                    from opinion.db import Banco
                    self._f = ingestao.carregar_do_banco(Banco())
                elif self.fonte == "arquivos":
                    from opinion.mundo import PASTA_BOOTSTRAP
                    self._f = ingestao.carregar_de_arquivos(PASTA_BOOTSTRAP)
                else:
                    self._f = ingestao.ler_staging(PASTA_STAGING)
                self._f_em = time.time()
                self._resolvedor = C.ResolvedorTokens(self._f["usuarios"].usuario_id.tolist())
                log.info("Dados recarregados de %s em %.1f s (%d encontros)", self.fonte, time.time() - t0,
                         len(self._f["encontros"]))
            return self._f

    # ----------------------------------------------------------------- modelo
    def modelo(self):
        with self._lock:
            if self._modelo is None or time.time() - self._checado_em > self.recarregar_s:
                self._carregar_modelo()
            return self._modelo, self._info

    def _carregar_modelo(self):
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
        cliente = MlflowClient()
        self._checado_em = time.time()
        erros = []
        for nome, alias in candidatos_modelo(self.cfg):
            try:
                v = cliente.get_model_version_by_alias(nome, alias)
            except Exception as ex:
                erros.append(f"{nome}@{alias}: {type(ex).__name__}")
                continue
            if self._info.get("nome") == nome and self._info.get("versao") == v.version:
                return                                  # nada mudou
            run = cliente.get_run(v.run_id)
            modelo = mlflow.sklearn.load_model(f"models:/{nome}@{alias}")
            p = run.data.params
            m = run.data.metrics
            self._modelo = modelo
            self._info = {"nome": nome, "alias": alias, "versao": v.version, "run_id": v.run_id,
                          "uri": f"models:/{nome}@{alias}",
                          "algoritmo": p.get("algoritmo") or p.get("modelo"), "params": p.get("params_json"),
                          "calibracao": p.get("calibracao", "nenhuma"), "limiar": float(p["limiar"]),
                          "limiar_regra": float(p.get("limiar_regra", "nan")),
                          "colunas": list(getattr(modelo, "feature_names_in_", [])),
                          "carregado_em": datetime.now().isoformat(timespec="seconds"),
                          "metricas": {k: m[k] for k in sorted(m) if k.startswith(("validacao_", "teste_"))
                                       and "no_limiar_todos" not in k}}
            log.info("Modelo carregado: %s v%s (%s)", nome, v.version, self._info["algoritmo"])
            return
        if self._modelo is None:
            raise RuntimeError("Nenhum modelo registrado no MLflow. Rode o treino (CC5) ou a DAG. " + "; ".join(erros))

    # ----------------------------------------------------------------- previsão
    def faixa(self, p: float, limiar: float) -> str:
        if p < limiar:
            return "nenhum"
        return "periferico_perceptivel" if p >= limiar * self.fator_perceptivel else "periferico_leve"

    def traduzir(self, payload: dict) -> dict:
        f = self.dados()
        return traduzir(payload, f["usuarios"].usuario_id.tolist())

    def prever(self, encontro: dict) -> dict:
        """`encontro` já traduzido (usuario_a/usuario_b reais, inicio/fim como Timestamp)."""
        t0 = time.time()
        f = self.dados()
        modelo, info = self.modelo()
        a, b = encontro["usuario_a"], encontro["usuario_b"]
        conhecidos = set(f["usuarios"].usuario_id)
        if a not in conhecidos or b not in conhecidos:
            raise ValueError("usuário fora da base")
        if a == b:
            raise ValueError("usuario_a igual a usuario_b")
        # só os encontros dos dois entram no cálculo: as features de encontro são do par e de cada usuário
        e = f["encontros"]
        g = dict(f, encontros=e[e.usuario_a.isin((a, b)) | e.usuario_b.isin((a, b))])
        colunas = info["colunas"] or self._colunas_padrao()
        X, df = matriz(encontro, g, self.cfg_p, colunas)
        p = float(modelo.predict_proba(X[colunas])[:, 1][0])
        linha = df.iloc[0]
        midias = motivos_do_sinal(linha, self.cfg_p)
        afinidade = {m: round(float(linha[f"compat_score_{m}"]), 4) for m in MIDIAS}
        nivel = self.faixa(p, info["limiar"])
        return {
            "encontro_id": encontro["encontro_id"], "pulseira_id": encontro.get("pulseira_id"),
            "usuario_a": a, "usuario_b": b, "inicio": pd.Timestamp(encontro["inicio"]).isoformat(),
            "fim": pd.Timestamp(encontro["fim"]).isoformat(), "local": encontro["local"],
            "duracao_s": int(encontro["duracao_s"]), "rssi_medio": float(encontro["rssi_medio"]),
            "probabilidade": round(p, 4), "limiar": round(info["limiar"], 4), "sinal": p >= info["limiar"],
            "nivel": nivel, "midias_aviso": midias, "afinidade_por_midia": afinidade,
            "compat_score": round(float(linha.compat_score), 4), "algum_segue": int(linha.algum_segue),
            "encontros_anteriores_par": int(linha.n_encontros_anteriores_par),
            "modelo": f"{info['nome']} v{info['versao']}", "algoritmo": info["algoritmo"],
            "latencia_ms": round(1000 * (time.time() - t0), 1),
        }

    def prever_payload(self, payload: dict) -> dict:
        """Mensagem `encontro` do contrato MQTT (IDs anônimos e rotativos)."""
        enc = self.traduzir(payload)
        f = self.dados()
        dono = f["usuarios"].set_index("usuario_id").pulseira_id
        enc["pulseira_id"] = payload.get("pulseira_id") or dono.get(enc["usuario_a"])
        return self.prever(enc)

    def _colunas_padrao(self) -> list[str]:
        import json
        from opinion.modelo import PASTA_PROCESSADOS
        return json.loads((PASTA_PROCESSADOS / "metadados.json").read_text())["colunas_features"]


def mensagem_feedback(r: dict) -> dict:
    """Sinal enviado à pulseira depois do encontro (tópico feedback)."""
    padrao = {"periferico_leve": "luz_suave", "periferico_perceptivel": "luz_pulsante"}.get(r["nivel"], "nenhum")
    return {"v": C.VERSAO, "tipo": "previsao_48h", "encontro_id": r["encontro_id"], "nivel": r["nivel"],
            "padrao": padrao, "probabilidade": r["probabilidade"], "midias": r["midias_aviso"],
            "mensagem": "Vocês têm boa chance de se conectar no OpiniOn. Que tal ver o perfil?",
            "ts": datetime.now().isoformat(timespec="seconds")}
