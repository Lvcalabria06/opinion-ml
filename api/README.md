# api/ — API de inferência (CC7)

Ainda não implementada. Entra no **CC7** (Sprint 7): integração do modelo ao pipeline, com consumo
automático de encontros novos e inferências disponíveis por API.

O que já existe para ela usar:

- `src/opinion/modelo/pontuar.py`: recebe um encontro no formato do contrato MQTT e reproduz
  as features do treino (`carregar_modelo`, `traduzir`, `matriz`). O `pontuar` serve ao
  baseline geral opcional;
- os modelos por mídia registrados no MLflow como `opinion-cc4-aviso-categoria-<midia>@melhor`;
- o aviso por mídia que a ingestão já envia em `opinion/pulseira/{id}/feedback` ao receber `proximidade`.
