# api/ — API de inferência (CC7)

Ainda não implementada. Entra no **CC7** (Sprint 7): integração do modelo ao pipeline, com consumo
automático de encontros novos e inferências disponíveis por API.

O que já existe para ela usar:

- `src/opinion/modelo/pontuar.py`: recebe um encontro no formato do contrato MQTT, calcula as features
  só com o passado e devolve a probabilidade e o sinal (`carregar_modelo`, `traduzir`, `pontuar`);
- o modelo registrado no MLflow (`opinion-baseline-cc3@champion` hoje; o do CC5 depois);
- o tópico `opinion/pulseira/{id}/feedback` do contrato MQTT, para devolver o sinal à pulseira.
