# app/ — dashboard (CC9)

Streamlit em `app/dashboard.py`. Lê tudo da API de inferência (`OPINION_API_URL`, padrão
`http://localhost:8000`) e se atualiza sozinho (intervalo ajustável na barra lateral).

```bash
streamlit run app/dashboard.py            # http://localhost:8501
```

| Aba | Conteúdo |
|---|---|
| Ao vivo | indicadores no topo; últimas inferências com a chance e o nível do sinal; distribuição dos sinais; avisos de afinidade por mídia; últimos encontros recebidos pela ingestão (MQTT) |
| Indicadores | conversão por local e por dia, encontros e avisos por dia, fadiga de notificação, definições |
| Modelo | modelo servido (MLflow), limiar, calibração e métricas de validação e teste |
| Simular encontro | pontua um par na hora pela API e mostra a afinidade por mídia |

Indicadores da seção 3.2 (calculados em `src/opinion/indicadores.py`):

- **taxa de conversão de encontros:** encontros entre pessoas que são amigas na rede (o rótulo do
  experimento Last.fm; estático, porque as amizades não têm data);
- **precisão dos sinais da pulseira:** entre os encontros com sinal, a fração em que o par é amigo (aviso
  de afinidade e sinal do modelo, separados);
- **taxa de silenciamento:** sinais seguidos de troca da pulseira para `pausado` em até 24 h.
