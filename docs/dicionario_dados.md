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
