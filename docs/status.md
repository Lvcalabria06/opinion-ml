# Status do projeto — onde estamos

> **Fonte única do progresso.** Atualize este arquivo sempre que uma entrega mudar de estado
> (feita, em revisão, bloqueada) ou quando uma decisão for tomada. Decisões com justificativa
> vão também para [`decisoes.md`](decisoes.md).

Última atualização: **22/11/2026**. Branch `experimento-lastfm` (experimento sobre a `main`).

## Experimento Last.fm (branch `experimento-lastfm`)

CC1 a CC4 refeitos com o HetRec 2011 Last.fm e a pergunta "dois usuários se encontraram; qual a chance
de serem amigos?" ([D-014](decisoes.md), [fonte](fontes_dados.md)). Teste: 24.497 encontros, 184 encontros
positivos (176 pares), prevalência 0,0075.

| Modelo | PR-AUC | ROC-AUC |
|---|---|---|
| Regressão logística (baseline CC3) | 0,063 | 0,837 |
| Regra simples (`compat_score`) | 0,027 | 0,724 |
| Jaccard de artistas sozinho (medição pontual da equipe, mesmo split de teste) | 0,057 | 0,824 |
| Random Forest (escolhido no CC4 pela validação) | 0,055 (IC 0,041–0,074) | 0,833 |

No teste a regressão logística ficou em primeiro; Random Forest contra a regra: +0,027 (IC +0,014 a +0,044).
Checkpoint 2 OK. MLflow: `opinion-cc3-lastfm`, `opinion-baseline-cc3-lastfm`, `opinion-cc4-lastfm`,
`opinion-cc4-lastfm@melhor`. Validação da simulação: 13 OK (modo amizade; cenários de sensibilidade não
se aplicam). Notebook reexecutado, 12 figuras.

**CC5 e CC6 (23/11/2026, [D-015](decisoes.md)):** Optuna na LR e na RF ([`otimizacao.md`](otimizacao.md));
CV estratificada agrupada por par, 5 × 3 ([`validacao_cruzada.md`](validacao_cruzada.md)). LR 0,061 ± 0,008 e
RF 0,058 ± 0,007 empatam; a LR é a recomendada (mais simples, sem sobreajuste).

Pendências do experimento: teste de MQTT e Docker de ponta a ponta (Docker desligado, não
verificado); validação do professor para o novo problema. Limitações: sem datas de amizade (sem split
temporal, possível causalidade reversa), encontros sem sinal real do label, ~1,9 mil usuários, licença não comercial.

Os números abaixo, até a próxima seção de pendências, descrevem a `main` (Amazon).

## Entregas e estado atual

| Entrega | Estado | Referência |
|---|---|---|
| CC1, pulseira virtual, MQTT, ingestão e Checkpoint 1 | Implementado | [Contrato MQTT](contrato_mqtt.md) |
| CC2, pipeline temporal e dados preparados | Implementado | [Dicionário de dados](dicionario_dados.md) |
| CC3, baseline acadêmico e Checkpoint 2 | Implementado | `src/opinion/modelo/treinar.py`, `src/opinion/checkpoint2.py` |
| CC4, quatro famílias de modelo | Implementado com recortes por mídia | [Resultados atuais](match_por_midia.md) |
| Aviso por categoria na proximidade | Implementado na ingestão e pulseira virtual | [Decisão D-013](decisoes.md) |
| API e dashboard | Pendentes | `api/README.md`, `app/README.md` |

O bootstrap atual contém 1.200 usuários da Amazon, 48.018 obras, 58.251 avaliações e 69.384 encontros simulados. Após a purga, o dataset contém 68.315 encontros, com 46.244 no treino, 10.101 na validação e 11.970 no teste. A taxa de `conectou` é 8,16% no dataset preparado. Os cinco modelos por mídia foram treinados no MLflow local.

O backend emite aviso quando pelo menos uma mídia tem afinidade alta. O corte inicial é 0,68. Em 6.972 encontros houve categoria sinalizada, 209 em mais de uma. O label de 48 h é único por encontro e inclui interações de fundo. A demonstração do Checkpoint 2 informa o aviso por categoria e estima a chance de interação com os modelos registrados. O baseline geral segue disponível com `--baseline`.

O notebook de EDA e [a comparação geral](comparacao_modelos.md) foram gerados em rodadas anteriores. Seus gráficos e números estão identificados como históricos. A [validação atual da simulação](validacao_simulacao.md) encontrou 15 checagens OK, uma atenção e uma falha: a taxa de interação entre desconhecidos foi de 1,2%, abaixo da faixa de 2%–15% definida para a checagem.

## O que ainda falta

### Para fechar com o Design

| Decisão | Proposta da equipe de ML | Impacto |
|---|---|---|
| A pulseira deve sinalizar quem eu já sigo? | política atual usa apenas afinidade e modo | decisão de produto |
| Duração mínima antes do sinal | aviso no início é a hipótese atual | política de sinal |
| Personas definitivas | substituir as provisórias de `config/simulacao.yaml` | simulador |
| Custo de um sinal errado | — | define a métrica final e o limiar do modelo |
| Padrões físicos do tópico `feedback` | `luz_suave` no protótipo | Design |

### Pendências de infraestrutura

- **`docker compose up --build` nunca foi testado** (o ambiente original não tinha Docker). Alguém
  com Docker precisa rodar uma vez antes da apresentação e registrar o resultado aqui.

---

## Próximos requisitos (para planejamento)

| Requisito | O que se sabe hoje |
|---|---|
| CC5 | Otimização por mídia, com validação temporal e comparação com a regra da categoria |
| CC6 | k-fold **agrupado por par** (`par_id`), estabilidade, overfitting/underfitting |
| CC7 | API de inferência e integração do modelo; a ingestão já envia `feedback` por categoria |
| Semana 16 | testes do protótipo do Design com usuários reais; comparar com a simulação (seção 4 da validação) |
