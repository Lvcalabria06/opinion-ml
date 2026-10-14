# Dicionário de dados — bootstrap dataset (CC1)

Pasta `data/bootstrap/`. Período simulado: **06/07/2026 a 27/09/2026 (12 semanas)**, mais 48 h de
interações depois do fim para não cortar o label dos últimos encontros.

```
data/bootstrap/
├── historico_app/        o que o APP guarda (entra no banco pela carga)
├── pulseira/             o que a PULSEIRA publica (entra no banco pelo MQTT)
├── rotulos_referencia.csv
├── relatorio_simulacao.md
└── verdade_simulacao_NAO_USAR_COMO_FEATURE/
```

## `historico_app/`

| Arquivo | Colunas | Observação |
|---|---|---|
| `usuarios.csv` | usuario_id, pulseira_id, data_cadastro | todo usuário tem uma pulseira |
| `obras.csv` | obra_id, titulo, midia, generos, popularidade | gêneros separados por `\|` |
| `avaliacoes.csv` | avaliacao_id, usuario_id, obra_id, nota (1–5), criado_em | inclui 26 semanas antes da simulação |
| `seguidores.csv` | seguidor, seguido, desde | quem segue quem e desde quando (inclui os novos) |
| `interacoes.csv` | interacao_id, origem, destino, tipo, obra_id, avaliacao_id, criado_em | tipos: `seguir`, `abrir_perfil`, `curtir_review`, `comentar_review`, `salvar_obra` |

As interações vêm de três fontes, e o modelo **não sabe qual**: conexões geradas por encontros,
interações entre amigos e interações vindas do feed.

## `pulseira/` — mesmas mensagens do contrato MQTT

| Arquivo | Colunas |
|---|---|
| `encontros.csv` | encontro_id, pulseira_id, usuario_a, usuario_b (tokens rotativos), inicio, duracao_s, rssi_medio, local |
| `interesses.csv` | encontro_id, pulseira_id, usuario (token), ts |
| `status.csv` | pulseira_id, usuario (token), modo, ts |

## `rotulos_referencia.csv`

`encontro_id, conectou`. Vale 1 se, em até 48 h após o fim do encontro, um dos dois seguiu, abriu
o perfil, curtiu ou comentou uma review, ou salvou uma obra avaliada pelo outro (seção 4.4).

É uma **referência para conferência**. No Sprint 2, o label deve ser recalculado pela DAG do
Airflow a partir das tabelas `encontros` e `interacoes` do banco.

## `verdade_simulacao_NAO_USAR_COMO_FEATURE/`

Variáveis ocultas, só para validar a simulação (Sprint 2) e nunca para treinar o modelo:

- `usuarios_ocultos.csv`: persona, abertura social, nível de atividade
- `encontros_ocultos.csv`: usuários reais, se eram amigos, compatibilidade "verdadeira", probabilidade latente de conexão e se o encontro de fato causou a conexão
- `interacoes_causadas_por_encontro.csv`: quais interações nasceram de qual encontro

## Cuidados para o Sprint 2 (seção 4.5)

- **Split temporal:** treino semanas 1–8 (até 30/08), validação 9–10 (até 13/09) e teste 11–12.
- **Sem informação do futuro:** compatibilidade, encontros anteriores e "já se seguem" devem ser
  calculados só com dados de **antes do início** do encontro. As tabelas trazem as datas para isso.
- **Agrupamento por par:** o mesmo par pode se encontrar várias vezes, por isso o k-fold deve ser agrupado por par.

---

## `data/processed/` — saída do pipeline (CC2)

Gerada pela DAG `opinion_pipeline` (ou `python -m opinion.pipeline.executar`). É a entrada do CC3.

`dataset_modelo.parquet`: uma linha por encontro rotulado e dentro do período do split.

| Coluna | Descrição |
|---|---|
| encontro_id, par_id | identificação; `par_id` = os dois usuários em ordem alfabética (grupo do k-fold) |
| usuario_a, usuario_b, inicio, fim, semana | contexto do encontro (não usar como feature) |
| split | `treino` (semanas 1–8), `validacao` (9–10), `teste` (11–12) |
| local_original | local em texto (para análise; a versão codificada está nas features) |
| **features** (33) | listadas em `metadados.json → colunas_features`, agrupadas em `grupos_features` |
| conectou | label: 1 se houve interação entre os dois em até 48 h após o fim |

Features, todas calculadas com dados de **antes do início** do encontro:

| Grupo | Features |
|---|---|
| Encontro | log_duracao, rssi_medio, n_encontros_anteriores_par, horas_desde_ultimo_encontro_par, primeiro_encontro_par |
| Contexto | local_* (one-hot), periodo_* (one-hot), dia_semana, hora, fim_de_semana, algum_modo_discreto |
| Compatibilidade | sim_generos, generos_top3_comum, n_obras_comuns, sim_notas, tem_obras_comuns, favoritas_comuns, compat_score |
| Perfil | atividade_min/max (avaliações em 28 dias), historico_min/max, conexoes_min/max |
| Relação | algum_segue, seguem_mutuamente, interacoes_anteriores_par, interagiram_antes |

`_min`/`_max`: como "usuário A" e "usuário B" é uma ordem arbitrária, guardamos o menor e o maior valor dos dois.
`compat_score` é a pontuação usada pela regra simples de referência (pesos em `config/pipeline.yaml`).

`features_encontros.parquet`: os mesmos dados para **todos** os encontros, inclusive os que ainda aguardam
as 48 h (`label_disponivel = 0`) e os removidos na fronteira (`split = purgado`).
