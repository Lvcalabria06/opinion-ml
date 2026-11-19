# app/ — dashboard (CC9)

Ainda não implementado. Entra no **CC9** (Sprint 8): dashboard interativo com encontros em tempo real,
probabilidades, sinais por faixa e indicadores de imersão.

O que a especificação pede que ele mostre:

- dados da infraestrutura de aquisição (MQTT e/ou API) e o funcionamento da solução em tempo de execução;
- métricas do modelo e as previsões;
- os indicadores da seção 3.2 do documento do projeto: taxa de conversão de encontros, precisão dos
  sinais da pulseira e taxa de silenciamento (fadiga de notificação).

Enquanto ele não existe, o Checkpoint 2 grava avisos por categoria e estimativas em `data/processed/avisos_categoria.csv`. A previsão do baseline geral opcional fica em `previsoes.csv`.
