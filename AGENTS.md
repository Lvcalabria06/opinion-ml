# AGENTS.md — OpiniOn (trilha de Machine Learning)

Modelo que responde: **dois usuários acabaram de se encontrar; qual a chance de interagirem no app em
até 48 h?** A pulseira sinaliza a afinidade alta por mídia ao detectar proximidade. A previsão de 48 h é calculada separadamente após o encontro. Projeto acadêmico entregue em
sprints (requisitos CC1, CC2, CC3…). Usuários, obras e avaliações vêm da Amazon Reviews 2023; encontros e interações no app
ainda são simulados. A pulseira usada aqui é virtual.

## Antes de começar qualquer tarefa

1. Leia [`docs/status.md`](docs/status.md): o que está feito, em revisão e o que falta (frentes A e B).
2. Se o projeto for novo para você, leia [`docs/contexto.md`](docs/contexto.md) (produto, pergunta,
   fluxo dos dados, glossário).
3. Confira a branch e o estado do Git antes de editar.

## Regras do projeto (invariantes)

Estas regras protegem a validade do modelo. Uma mudança que quebre alguma está errada, mesmo que as
métricas melhorem.

- **Só o passado:** toda feature de um encontro usa apenas dados de antes do `inicio` dele. Feature
  nova precisa passar em `tests/test_pipeline.py::test_sem_vazamento_do_futuro`.
- **Split temporal:** treino semanas 1–8, validação 9–10, teste 11–12 (`config/pipeline.yaml`), com
  purga da fronteira. Quando houver validação cruzada, agrupe por `par_id`. O teste é medido uma vez, no fim.
- **Padronização dentro do pipeline do modelo**, ajustada só no treino. `dataset_modelo.parquet` sai
  sem padronização de propósito.
- **Verdade da simulação fica fora do modelo:** nada de `data/bootstrap/verdade_simulacao_NAO_USAR_COMO_FEATURE/`
  vira feature, nem indiretamente. Serve só para validar.
- **Resultado sempre comparado:** em cada recorte por mídia, reporte PR-AUC geral e só entre desconhecidos (`algum_segue == 0`),
  ao lado da prevalência e da regra simples do recorte (`compat_score_<midia>`).
- **Aviso por categoria:** a regra usa somente o histórico anterior ao início do encontro e o corte de cada mídia. A estimativa do modelo usa dados disponíveis ao fim.
- **Honestidade sobre os dados:** métricas medem a simulação. "Melhor algoritmo" e ranking de features
  não são descobertas (ver [`docs/validacao_simulacao.md`](docs/validacao_simulacao.md)).
- **Mesmas features no treino e na inferência:** a API e o consumidor MQTT usam `opinion.modelo.pontuar.matriz`
  (funções do pipeline); nunca reimplemente uma feature só para a inferência.
- **Dados externos têm fonte:** todo dataset público usado é citado (origem, licença, filtro aplicado).

## Convenções

- Código, nomes, mensagens e docs em **português**; mensagens de commit em inglês conforme o histórico recente (sem acento em identificadores:
  `encontros`, `conectou`, `compat_score`).
- Premissas e parâmetros ficam em YAML em `config/`, nunca fixos no código.
- Cada etapa do pipeline é uma função em `src/opinion/pipeline/` chamada tanto pela DAG quanto por
  `python -m opinion.pipeline.executar`; mantenha as duas formas funcionando.
- Gerados não vão para o Git: `data/staging/`, `data/processed/`, `data/opinion.db`, `.env` (ver `.gitignore`).
- Commits por entrega, no formato de prefixo curto seguido de resumo em inglês, como `CC4: add category-based affinity alerts`.

## Verificar o trabalho

```bash
pip install -r requirements-analise.txt && cp .env.example .env    # uma vez
python -m pytest tests -q                                           # fontes, pipeline, baseline
python -m opinion.pipeline.executar --fonte arquivos                # se data/processed/ estiver vazio
python -m opinion.modelo.comparar --midia musica                   # CC4 por mídia → MLflow
python -m opinion.modelo.otimizar --trials 5 --sem-registro        # CC5 rápido, sem gravar
python -m opinion.modelo.validacao_cruzada --sem-registro          # CC6 sem gravar
python -m opinion.checkpoint2 --usuarios usr-00694 usr-01029     # aviso musical e estimativa
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
| [`docs/otimizacao.md`](docs/otimizacao.md) · [`docs/validacao_cruzada.md`](docs/validacao_cruzada.md) | mexer no modelo de produção (CC5/CC6) |
| [`api/README.md`](api/README.md) · [`app/README.md`](app/README.md) | mexer na API, no consumidor de inferência ou no dashboard |
