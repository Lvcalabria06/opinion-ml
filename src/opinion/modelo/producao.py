"""
Modelo de produção (CC7/CC8): o que a API e o consumidor MQTT usam para estimar a chance de amizade.

Reajusta a configuração escolhida no CC5/CC6 (`config/modelo.yaml → producao`) com o dataset mais
recente do pipeline, calibra as probabilidades na validação (sigmoide, transformação monótona:
não muda a ordenação nem a PR-AUC) e escolhe o limiar do sinal (F0,5) na validação. A nova versão
só recebe o alias `@producao` se superar a regra simples na validação; caso contrário a versão
anterior continua servindo. É a última task da DAG do Airflow.

    python -m opinion.modelo.producao
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np

from opinion.modelo import PASTA_PROCESSADOS
from opinion.modelo.treinar import (_metricas_planas, _pipeline, avaliar_regra, carregar, chave_metrica, config,
                                    escolher_limiar, mascara_desconhecidos, medir_teste, metricas, no_limiar)


def calibrar(modelo, X_va, y_va, metodo: str):
    """Calibra as probabilidades na validação sem reajustar o modelo (FrozenEstimator)."""
    if metodo in (None, "nenhuma"):
        return modelo, "nenhuma"
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    return CalibratedClassifierCV(FrozenEstimator(modelo), method=metodo).fit(X_va, y_va), metodo


def treinar_producao(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True) -> dict:
    cfg = cfg or config()
    p = cfg["producao"]
    beta, chave = float(cfg["limiar"]["beta"]), chave_metrica(cfg)
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    d_va, d_te = mascara_desconhecidos(va, cfg).to_numpy(), mascara_desconhecidos(te, cfg).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cru = _pipeline(p["tipo"], p["params"]).fit(tr[cols], tr.conectou)
    modelo, metodo = calibrar(cru, va[cols], va.conectou, p.get("calibracao", "sigmoid"))
    p_va = modelo.predict_proba(va[cols])[:, 1]
    limiar, _ = escolher_limiar(va.conectou.to_numpy()[d_va], p_va[d_va], beta)
    met_va = metricas(va.conectou, p_va, d_va)
    met_va["no_limiar_desconhecidos"] = no_limiar(va.conectou.to_numpy()[d_va], p_va[d_va], limiar, beta)
    regra = avaliar_regra(va, te, d_va, d_te, beta)
    _, met_te = medir_teste({"modelo": modelo, "limiar": limiar}, te, cols, d_te, beta)
    supera = met_va[chave] > regra["validacao"][chave] and not np.isclose(met_va[chave], regra["validacao"][chave])
    r = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "familia": p["familia"], "tipo": p["tipo"],
         "params": p["params"], "calibracao": metodo, "limiar": limiar, "limiar_regra": regra["limiar"],
         "metrica": chave, "validacao": met_va, "teste": met_te, "regra_validacao": regra["validacao"],
         "supera_regra_validacao": bool(supera), "corte_dos_dados": meta.get("corte_dos_dados"),
         "n_treino": len(tr), "colunas": cols, "_modelo": modelo}
    if registrar:
        r.update(_registrar(r, cfg, tr[cols].head(5)))
    return r


def _registrar(r: dict, cfg: dict, exemplo) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    p = cfg["producao"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["modelo_registrado"])
    with mlflow.start_run(run_name=f"producao-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.log_params({"algoritmo": r["familia"], "modelo": r["familia"], "tipo": r["tipo"],
                           "calibracao": r["calibracao"], "limiar": r["limiar"], "limiar_regra": r["limiar_regra"],
                           "metrica_escolha": r["metrica"], "params_json": json.dumps(r["params"]),
                           "corte_dos_dados": r["corte_dos_dados"], "n_treino": r["n_treino"],
                           "supera_regra_validacao": r["supera_regra_validacao"]})
        mlflow.log_metrics(_metricas_planas("validacao", r["validacao"]))
        mlflow.log_metrics(_metricas_planas("teste", r["teste"]))
        mlflow.log_metrics(_metricas_planas("regra_validacao", r["regra_validacao"]))
        mlflow.log_dict({"colunas": r["colunas"]}, "colunas.json")
        modelo = r["_modelo"]
        info = mlflow.sklearn.log_model(
            sk_model=modelo, name="modelo", input_example=exemplo,
            signature=infer_signature(exemplo, modelo.predict_proba(exemplo)[:, 1]),
            registered_model_name=p["modelo_registrado"], serialization_format="cloudpickle")
        versao = str(info.registered_model_version)
        if r["supera_regra_validacao"]:
            MlflowClient().set_registered_model_alias(p["modelo_registrado"], p["alias"], versao)
    return {"mlflow_run": run.info.run_id, "mlflow_versao": versao, "promovido": r["supera_regra_validacao"],
            "mlflow_modelo": f"{p['modelo_registrado']}@{p['alias']}"}


def main():
    ap = argparse.ArgumentParser(description="Retreina e promove o modelo servido pela API (chance de amizade)")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    ap.add_argument("--sem-registro", action="store_true")
    a = ap.parse_args()
    r = treinar_producao(a.pasta, registrar=not a.sem_registro)
    k = r["metrica"]
    print(f"{r['familia']} {r['params']} (calibração {r['calibracao']}, limiar {r['limiar']:.4f})")
    print(f"{k}: validação {r['validacao'][k]:.3f} (regra {r['regra_validacao'][k]:.3f}) | teste {r['teste'][k]:.3f}"
          f" | Brier teste {r['teste']['brier']:.4f}")
    if r.get("mlflow_versao"):
        print(f"MLflow: {r['mlflow_modelo']} v{r['mlflow_versao']} — "
              + ("promovida" if r["promovido"] else "NÃO promovida (não superou a regra na validação)"))


if __name__ == "__main__":
    main()
