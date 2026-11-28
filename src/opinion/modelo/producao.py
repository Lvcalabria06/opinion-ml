"""
Modelo de produção (CC7/CC8): o que a API e o consumidor MQTT usam.

Reajusta a configuração otimizada no CC5 (família e hiperparâmetros lidos do MLflow) com o
dataset mais recente do pipeline, calibra as probabilidades e escolhe o limiar na validação.
A nova versão só recebe o alias `@producao` se superar a regra simples na validação; caso
contrário, a versão anterior continua servindo. É a última task da DAG do Airflow.

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
from opinion.modelo.otimizar import calibrar
from opinion.modelo.treinar import (_metricas_planas, _pipeline, avaliar_regra, carregar, config, escolher_limiar,
                                    mascara_desconhecidos, medir_teste, metricas, no_limiar)
from opinion.modelo.validacao_cruzada import modelo_cc5


def treinar_producao(pasta=PASTA_PROCESSADOS, cfg=None, registrar=True) -> dict:
    cfg = cfg or config()
    beta, chave = float(cfg["limiar"]["beta"]), cfg["metrica"]
    spec = modelo_cc5(cfg)
    dm, meta = carregar(pasta)
    cols = meta["colunas_features"]
    tr, va, te = (dm[dm.split == s].copy() for s in ("treino", "validacao", "teste"))
    desc_va, desc_te = mascara_desconhecidos(va).to_numpy(), mascara_desconhecidos(te).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cru = _pipeline(spec["tipo"], spec["params"]).fit(tr[cols], tr.conectou)
    modelo, metodo = calibrar(cru, va[cols], va.conectou, cfg["otimizacao"].get("calibracao", "sigmoid"))
    p_va = modelo.predict_proba(va[cols])[:, 1]
    limiar, _ = escolher_limiar(va.conectou[desc_va], p_va[desc_va], beta)
    met_va = metricas(va.conectou, p_va, desc_va)
    met_va["no_limiar_desconhecidos"] = no_limiar(va.conectou[desc_va], p_va[desc_va], limiar, beta)
    regra = avaliar_regra(va, te, desc_va, desc_te, beta)
    _, met_te = medir_teste({"modelo": modelo, "limiar": limiar}, te, cols, desc_te, beta)
    supera = met_va[chave] > regra["validacao"][chave] and not np.isclose(met_va[chave], regra["validacao"][chave])
    r = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "familia": spec["familia"], "tipo": spec["tipo"],
         "params": spec["params"], "origem": spec["origem"], "calibracao": metodo, "limiar": limiar,
         "limiar_regra": regra["limiar"], "validacao": met_va, "teste": met_te, "regra_validacao": regra["validacao"],
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
        mlflow.log_params({"algoritmo": r["familia"], "modelo": r["familia"], "tipo": r["tipo"], "origem": r["origem"],
                           "calibracao": r["calibracao"], "limiar": r["limiar"], "limiar_regra": r["limiar_regra"],
                           "params_json": json.dumps(r["params"]), "corte_dos_dados": r["corte_dos_dados"],
                           "supera_regra_validacao": r["supera_regra_validacao"], "n_treino": r["n_treino"]})
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
    return {"mlflow_run": run.info.run_id, "mlflow_versao": versao,
            "promovido": r["supera_regra_validacao"], "mlflow_modelo": f"{p['modelo_registrado']}@{p['alias']}"}


def main():
    ap = argparse.ArgumentParser(description="Retreina e promove o modelo servido pela API")
    ap.add_argument("--pasta", default=str(PASTA_PROCESSADOS))
    ap.add_argument("--sem-registro", action="store_true")
    a = ap.parse_args()
    r = treinar_producao(a.pasta, registrar=not a.sem_registro)
    va, te = r["validacao"], r["teste"]
    print(f"{r['familia']} {r['params']} (origem {r['origem']}, calibração {r['calibracao']}, limiar {r['limiar']:.4f})")
    print(f"PR-AUC desconhecidos: validação {va['pr_auc_desconhecidos']:.3f} (regra {r['regra_validacao']['pr_auc_desconhecidos']:.3f})"
          f" | teste {te['pr_auc_desconhecidos']:.3f} | Brier teste {te['brier']:.4f}")
    if r.get("mlflow_versao"):
        print(f"MLflow: {cfg_nome()} v{r['mlflow_versao']} — "
              + ("promovida para @producao" if r["promovido"] else "NÃO promovida (não superou a regra na validação)"))


def cfg_nome() -> str:
    return config()["producao"]["modelo_registrado"]


if __name__ == "__main__":
    main()
