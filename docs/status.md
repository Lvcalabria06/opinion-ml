# Status do projeto — onde estamos

> **Fonte única do progresso.** Atualize este arquivo sempre que uma entrega mudar de estado
> (feita, em revisão, bloqueada) ou quando uma decisão for tomada. Decisões com justificativa
> vão também para [`decisoes.md`](decisoes.md).

Última atualização: **27/11/2026**.

## Entregas e estado atual

| Entrega | Estado | Referência |
|---|---|---|
| CC1, pulseira virtual, MQTT, ingestão e Checkpoint 1 | Implementado; testado com PostgreSQL em 27/11 | [Contrato MQTT](contrato_mqtt.md) |
| CC2, pipeline temporal e dados preparados | Implementado; DAG testada no Airflow 3.1 em 27/11 | [Dicionário de dados](dicionario_dados.md) |
| CC3, baseline acadêmico e Checkpoint 2 | Implementado | `src/opinion/modelo/treinar.py`, `src/opinion/checkpoint2.py` |
| CC4, comparação de 7 famílias (inclui KNN, AdaBoost e MLP) | Implementado | [Comparação](comparacao_modelos.md) · [por mídia](match_por_midia.md) |
| CC5, otimização com Optuna e calibração | Implementado | [Otimização](otimizacao.md) |
| CC6, validação cruzada agrupada, estabilidade, janela temporal | Implementado | [Validação cruzada](validacao_cruzada.md) |
| CC7, consumidor MQTT, sinal de volta à pulseira, API, retreino na DAG | Implementado e testado de ponta a ponta | `src/opinion/inferencia*.py`, `api/` |
| CC8, Docker Compose completo | Escrito e validado com `docker compose config`; fluxo testado fora do Docker | `docker-compose.yml` |
| CC9, dashboard | Implementado | `app/` |
| Relatório SBC | Gerado | [`reports/relatorio_opinion.pdf`](../reports/relatorio_opinion.pdf) |
| README e Google Sites | Nome, membros e link preenchidos; faltam os usuários do GitHub de 3 membros | [README](../README.md) · [conteúdo do site](google_sites.md) |

## Resultados principais (PR-AUC entre desconhecidos)

| | Validação | Teste | k-fold (média ± desvio) |
|---|---|---|---|
| AdaBoost otimizado e calibrado (produção) | 0,150 | 0,188 (IC 0,146–0,248) | 0,139 ± 0,011 |
| Regra simples (afinidade de gosto) | 0,100 | 0,131 | 0,109 ± 0,021 |
| Acaso (prevalência) | — | 0,014 | 0,011 |

## O que ainda falta

### Para a equipe

- **Rodar `docker compose up --build` uma vez** numa máquina com Docker e registrar o resultado aqui.
  O ambiente de 27/11 não baixava imagens do Docker Hub; o mesmo fluxo foi testado com os serviços
  instalados localmente (PostgreSQL, Mosquitto, Airflow 3.1, servidor MLflow).
- **Usuários do GitHub** de João Pedro, Maria Júlia Dantas e Maria Luíza Dantas no topo do README.
- **Conferir o template do relatório** com o anexo da especificação ([`reports/README.md`](../reports/README.md)).
- **Google Sites:** publicar o conteúdo de [`google_sites.md`](google_sites.md).

### Para fechar com o Design

| Decisão | Proposta da equipe de ML | Impacto |
|---|---|---|
| A pulseira deve sinalizar quem eu já sigo? | política atual usa apenas afinidade e modo | decisão de produto |
| Duração mínima antes do sinal | aviso de afinidade no início; sinal do modelo no fim | política de sinal |
| Personas definitivas | substituir as provisórias de `config/simulacao.yaml` | simulador |
| Custo de um sinal errado | F0,5 (precisão vale o dobro) | define o limiar |
| Padrões físicos do tópico `feedback` | `luz_suave` (leve) e `luz_pulsante` (perceptível) | Design |

### Próximo passo de dados

Semana 16: testes do protótipo do Design com pessoas reais. Os encontros reais entram pelo mesmo
MQTT e pelo mesmo pipeline; comparar com a simulação (seção 4 de [`validacao_simulacao.md`](validacao_simulacao.md))
e refazer CC4–CC6.
