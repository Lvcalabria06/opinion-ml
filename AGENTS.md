# AGENTS.md — OpiniOn (trilha de Machine Learning)

Modelo que responde: **dois usuários acabaram de se encontrar; qual a chance de interagirem no app em
até 48 h?** A resposta decide se a pulseira do OpiniOn dá um sinal. Projeto acadêmico entregue em
sprints (requisitos CC1, CC2, CC3…). Os dados são simulados (não há usuários reais nem pulseira) e
Usuários, obras e avaliações vêm da Amazon Reviews 2023; encontros e o label ainda são simulados.

## Antes de começar qualquer tarefa

1. Leia [`docs/status.md`](docs/status.md): o que está feito, em revisão e o que falta (frentes A e B).
2. Se o projeto for novo para você, leia [`docs/contexto.md`](docs/contexto.md) (produto, pergunta,
   fluxo dos dados, glossário).
3. Confira a branch. A Sprint 2 ainda está no PR #1 (`sprint-2-cc2`); trabalho que depende dela parte
   dessa branch, não da `main`.

## Regras do projeto (invariantes)

Estas regras protegem a validade do modelo. Uma mudança que quebre alguma está errada, mesmo que as
métricas melhorem.

- **Só o passado:** toda feature de um encontro usa apenas dados de antes do `inicio` dele. Feature
  nova precisa passar em `tests/test_pipeline.py::test_sem_vazamento_do_futuro`.
- **Split temporal:** treino semanas 1–8, validação 9–10, teste 11–12 (`config/pipeline.yaml`), com
  purga da fronteira. Validação cruzada é agrupada por `par_id`. O teste é medido uma vez, no fim.
- **Padronização dentro do pipeline do modelo**, ajustada só no treino. `dataset_modelo.parquet` sai
  sem padronização de propósito.
- **Verdade da simulação fica fora do modelo:** nada de `data/bootstrap/verdade_simulacao_NAO_USAR_COMO_FEATURE/`
  vira feature, nem indiretamente. Serve só para validar.
- **Resultado sempre comparado:** reporte PR-AUC geral e só entre desconhecidos (`algum_segue == 0`),
  ao lado da prevalência e da regra simples (`compat_score`).
- **Honestidade sobre os dados:** métricas medem a simulação. "Melhor algoritmo" e ranking de features
  não são descobertas (ver [`docs/validacao_simulacao.md`](docs/validacao_simulacao.md)).
- **Dados externos têm fonte:** todo dataset público usado é citado (origem, licença, filtro aplicado).

## Convenções

- Código, nomes, mensagens, docs e commits em **português** (sem acento em identificadores:
  `encontros`, `conectou`, `compat_score`).
- Premissas e parâmetros ficam em YAML em `config/`, nunca fixos no código.
- Cada etapa do pipeline é uma função em `opinion/pipeline/` chamada tanto pela DAG quanto por
  `python -m opinion.pipeline.executar`; mantenha as duas formas funcionando.
- Gerados não vão para o Git: `data/staging/`, `data/processed/`, `data/opinion.db`, `.env` (ver `.gitignore`).
- Commits por entrega, no formato `Sprint N (CCN): resumo` ou `README: resumo`.

## Verificar o trabalho

```bash
pip install -r requirements-analise.txt && cp .env.example .env    # uma vez
python -m pytest tests -q                                           # fontes, pipeline, baseline
python -m opinion.pipeline.executar --fonte arquivos                # se data/processed/ estiver vazio
python -m opinion.modelo.treinar                                    # CC3 → MLflow
python -m opinion.checkpoint2                                       # Checkpoint 2
python -m opinion.modelo.comparar                                   # CC4 → MLflow + docs/comparacao_modelos.md
```

Sem `data/reais/amazon2023/`: `python -m opinion.fontes.amazon2023 tudo` e depois `python -m opinion.gerar_bootstrap`.

O roteiro completo (MQTT, Checkpoint 1, Airflow, Docker) e as saídas esperadas estão no [README](README.md).

## Ao terminar uma tarefa

- Atualize [`docs/status.md`](docs/status.md) (estado e data) e o [README](README.md) se o jeito de rodar mudou.
- Decisão que muda o que se constrói vai para [`docs/decisoes.md`](docs/decisoes.md).
- Coluna ou arquivo de dados novo vai para [`docs/dicionario_dados.md`](docs/dicionario_dados.md).

## Mapa da documentação

| Arquivo | Leia quando |
|---|---|
| [`docs/status.md`](docs/status.md) | sempre, antes de começar |
| [`docs/contexto.md`](docs/contexto.md) | primeira vez no projeto |
| [`docs/decisoes.md`](docs/decisoes.md) | for mudar algo que parece arbitrário (split, métrica, padronização) |
| [`docs/dicionario_dados.md`](docs/dicionario_dados.md) | mexer em dados, features ou no dataset do modelo |
| [`docs/contrato_mqtt.md`](docs/contrato_mqtt.md) | mexer na pulseira virtual, na ingestão ou falar com o Design |
| [`docs/fontes_dados.md`](docs/fontes_dados.md) | mexer em usuários, obras, avaliações ou baixar dados externos |
| [`docs/validacao_simulacao.md`](docs/validacao_simulacao.md) | mudar premissas do simulador ou interpretar resultados |
