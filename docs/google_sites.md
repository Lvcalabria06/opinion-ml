# Conteúdo do Google Sites — OpiniON

Site: https://sites.google.com/cesar.school/opinion

Texto pronto para colar, organizado por página. Imagens sugeridas entre colchetes (estão no repositório).

---

## Página 1 — Início

**OpiniON**
*Encontre quem gosta do que você gosta, no lugar onde você já está.*

O OpiniON é uma plataforma social em que as pessoas avaliam filmes, séries, livros, jogos e músicas.
Uma pulseira detecta quando duas pessoas estão perto uma da outra (no cinema, na livraria, no campus
ou numa convenção de jogos) e, se vale a pena, emite um sinal discreto de luz.

Por trás do sinal, um modelo de aprendizado de máquina responde uma pergunta:
**dois usuários acabaram de se encontrar. Qual a chance de eles se conectarem no app nas próximas 48 horas?**

[Botões: Relatório (PDF) · Código no GitHub · Painel ao vivo (demonstração)]

---

## Página 2 — Equipe

**Desenvolvimento (Ciência da Computação)**
- João Pedro — jpsa@cesar.school
- Lucas Calabria — lvc@cesar.school
- Maria Júlia Dantas — mjdma@cesar.school
- Maria Luíza Dantas — mldt@cesar.school

**Design**
- Fábio André Oliveira Lins — faol@cesar.school
- Marcus Vinicius Menezes Barbosa — mvmb@cesar.school
- Pedro Mota — pom2@cesar.school
- Rodrigo Quirino de Mello — rqm@cesar.school

---

## Página 3 — Como funciona

1. **A pulseira percebe a proximidade.** Ela anuncia um identificador anônimo que muda todo dia;
   só o servidor sabe quem é quem.
2. **Aviso de afinidade.** No início do encontro, o servidor compara as avaliações das duas pessoas
   por mídia. Se a afinidade em alguma passa do corte, a pulseira recebe: "alguém por perto tem
   alta afinidade com você em música".
3. **Previsão de conexão.** No fim do encontro, com a duração e a intensidade do sinal Bluetooth, o
   modelo estima a chance de interação em 48 h. Acima do limiar, a pulseira acende (leve ou
   perceptível, conforme a chance). Quem está com a pulseira em pausa não recebe nada.
4. **Aprendizado contínuo.** Todo dia o pipeline no Airflow recalcula os dados, retreina o modelo e
   só troca a versão em uso se a nova for melhor que a regra de afinidade.

[Imagem: diagrama da arquitetura — Figura 1 do relatório]

Tecnologias: MQTT (Mosquitto), PostgreSQL, Apache Airflow, scikit-learn, Optuna, MLflow, FastAPI,
Streamlit e Docker Compose.

---

## Página 4 — Dados

- **Gostos reais:** 1.200 pessoas da *Amazon Reviews 2023* que avaliaram livros, filmes, séries,
  jogos e música (58.251 avaliações de 48.018 obras).
- **Encontros simulados:** 12 semanas, 69.384 encontros em cinema, livraria, campus e convenções de
  jogos. Ainda não há pulseira física nem usuários reais do app; os testes do protótipo com pessoas
  (semana 16) vão substituir essa parte.
- **Sem olhar para o futuro:** toda informação usada numa previsão é anterior ao encontro; um teste
  automático garante isso.

---

## Página 5 — Modelo e resultados

O que importa é acertar entre **desconhecidos**, pessoas que ainda não se seguem. Nesse grupo, só
cerca de 1 em cada 70 encontros vira conexão, então medimos a qualidade pela PR-AUC
(precisão × revocação).

| | PR-AUC entre desconhecidos (teste) |
|---|---|
| Sorteio (acaso) | 0,014 |
| Regra de afinidade de gosto | 0,131 |
| **Modelo OpiniON (AdaBoost otimizado)** | **0,188** |

- Comparamos 7 tipos de modelo (regressão logística, árvore, Random Forest, Gradient Boosting, KNN,
  AdaBoost e rede neural).
- Ajustamos o melhor com otimização bayesiana (Optuna) e calibramos as probabilidades.
- Na validação cruzada, o resultado variou pouco entre partições (0,139 ± 0,011) e se manteve ao
  prever semanas que o modelo nunca viu.
- No ponto de operação, 1 em cada 4 sinais vira conexão, contra cerca de 1 em 5 da regra de afinidade.

[Imagens: docs/img/cc4/pr_auc_ic.png e docs/img/cc6/folds.png]

---

## Página 6 — Painel

O painel mostra, em tempo real, os encontros que chegam da pulseira, a chance estimada de cada um,
os sinais emitidos e três indicadores de produto:

- **Taxa de conversão:** encontros que viraram interação no app (8,1%).
- **Precisão dos sinais:** quantos encontros sinalizados viraram conexão.
- **Taxa de silenciamento:** quantos sinais foram seguidos de pausa da pulseira em até 24 h
  (fadiga de notificação).

[Imagem: captura do dashboard em http://localhost:8501 depois do `docker compose up`]

---

## Página 7 — Links

- Código: https://github.com/Lvcalabria06/opinion-ml
- Relatório (PDF, template SBC): `reports/relatorio_opinion.pdf` no repositório
- Como rodar: `cp .env.example .env && docker compose up --build`
