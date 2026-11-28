# Status do projeto — onde estamos

> **Fonte única do progresso.** Atualize este arquivo sempre que uma entrega mudar de estado
> (feita, em revisão, bloqueada) ou quando uma decisão for tomada. Decisões com justificativa
> vão também para [`decisoes.md`](decisoes.md).

Última atualização: **28/11/2026**. Versão atual: experimento Last.fm (rótulo = amizade real, [D-014](decisoes.md)).

## Entregas e estado atual

| Entrega | Estado | Referência |
|---|---|---|
| CC1, pulseira virtual, MQTT, ingestão e Checkpoint 1 | Implementado; replay testado com PostgreSQL em 28/11 | [Contrato MQTT](contrato_mqtt.md) |
| CC2, pipeline e dados preparados | Implementado; DAG testada no Airflow 3.1 em 28/11 | [Dicionário de dados](dicionario_dados.md) |
| CC3, baseline e Checkpoint 2 | Implementado | `src/opinion/modelo/treinar.py`, `src/opinion/checkpoint2.py` |
| CC4, comparação de 7 famílias (inclui KNN, AdaBoost e MLP) | Implementado | [Comparação](comparacao_modelos.md) |
| CC5, Optuna na LR e na RF | Implementado | [Otimização](otimizacao.md) |
| CC6, validação cruzada agrupada por par | Implementado | [Validação cruzada](validacao_cruzada.md) |
| CC7, consumidor MQTT, sinal de volta à pulseira, API, retreino na DAG | Implementado e testado de ponta a ponta | `src/opinion/inferencia*.py`, `api/` |
| CC8, Docker Compose completo | Escrito e validado com `docker compose config`; fluxo testado fora do Docker | `docker-compose.yml` |
| CC9, dashboard | Implementado | `app/` |
| Relatório SBC | Gerado | [`reports/relatorio_opinion.pdf`](../reports/relatorio_opinion.pdf) |
| README e Google Sites | Nome, membros e link preenchidos; faltam usuários do GitHub de 2 membros | [README](../README.md) · [conteúdo do site](google_sites.md) |

## Resultados principais (PR-AUC em todos os encontros, teste)

Teste: 24.497 encontros, 184 entre pares amigos (prevalência 0,0075).

| Modelo | Validação | Teste | Validação cruzada |
|---|---|---|---|
| Regressão logística do CC5, calibrada (**produção**) | 0,065 | 0,061 | 0,061 ± 0,008 |
| Random Forest (escolhida no CC4 e no CC5 pela validação) | 0,070 | 0,053 | 0,058 ± 0,007 |
| Regra simples (`compat_score`) | 0,027 | 0,027 | 0,027 ± 0,003 |
| Acaso (prevalência) | — | 0,008 | 0,008 |

No ponto de operação (teste), o modelo de produção acerta 11,0% dos sinais, contra 4,8% da regra, sinalizando ~1,2% dos encontros.

## O que ainda falta

### Para a equipe

- **Rodar `docker compose up --build` uma vez** numa máquina com Docker e registrar o resultado aqui.
  O mesmo fluxo foi testado em 28/11 com os serviços instalados localmente.
- **Usuários do GitHub** de João Pedro e Maria Júlia Dantas no topo do README.
- **Conferir o template do relatório** com o anexo da especificação ([`reports/README.md`](../reports/README.md)).
- **Google Sites:** publicar o conteúdo de [`google_sites.md`](google_sites.md).
- **Validação do professor** para a troca do problema (interação em 48 h → amizade).

### Para fechar com o Design

| Decisão | Proposta da equipe de ML | Impacto |
|---|---|---|
| A pulseira deve sinalizar quem já é amigo? | hoje sinaliza pela chance, sem olhar a amizade | decisão de produto |
| Duração mínima antes do sinal | aviso de afinidade no início; sinal do modelo no fim | política de sinal |
| Custo de um sinal errado | F0,5 (precisão vale o dobro) | define o limiar |
| Padrões físicos do tópico `feedback` | `luz_suave` (leve) e `luz_pulsante` (perceptível) | Design |

### Limitações conhecidas

Sem datas de amizade não há split temporal e pode haver causalidade reversa; os encontros são simulados
e independentes da amizade; ~1,9 mil usuários; licença não comercial do HetRec 2011.

### Próximo passo de dados

Semana 16: testes do protótipo do Design com pessoas reais. Os encontros reais entram pelo mesmo MQTT e
pelo mesmo pipeline; refazer CC4–CC6 com eles.
