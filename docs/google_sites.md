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
**dois usuários acabaram de se encontrar. Qual a chance de eles serem amigos?**

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
3. **Previsão de amizade.** No fim do encontro, com a duração e a intensidade do sinal Bluetooth, o
   modelo estima a chance de os dois serem amigos. Acima do limiar, a pulseira acende (leve ou
   perceptível, conforme a chance). Quem está com a pulseira em pausa não recebe nada.
4. **Aprendizado contínuo.** Todo dia o pipeline no Airflow recalcula os dados, retreina o modelo e
   só troca a versão em uso se a nova for melhor que a regra de afinidade.

[Imagem: diagrama da arquitetura — Figura 1 do relatório]

Tecnologias: MQTT (Mosquitto), PostgreSQL, Apache Airflow, scikit-learn, Optuna, MLflow, FastAPI,
Streamlit e Docker Compose.

---

## Página 4 — Dados

- **Gostos e amizades reais:** 1.892 pessoas do Last.fm (HetRec 2011), com os artistas que mais
  escutam (92.834 pares pessoa–artista) e 12.717 amizades reais.
- **Encontros simulados:** 12 semanas, 162.897 encontros em cinema, livraria, campus e convenções de
  jogos. Ainda não há pulseira física; os testes do protótipo com pessoas (semana 16) vão substituir
  essa parte. Os encontros simulados não sabem quem é amigo de quem.
- **Sem olhar para o futuro:** toda informação usada numa previsão é anterior ao encontro; um teste
  automático garante isso. A rede de amizades em si fica fora do modelo, porque seria a resposta.

---

## Página 5 — Modelo e resultados

Só 3 em cada 400 encontros simulados são entre amigos, então medimos a qualidade pela PR-AUC
(precisão × revocação), que valoriza acertar os casos raros.

| | PR-AUC (teste) |
|---|---|
| Sorteio (acaso) | 0,008 |
| Regra de afinidade de gosto | 0,027 |
| **Modelo OpiniON (regressão logística)** | **0,061** |

- Comparamos 7 tipos de modelo (regressão logística, árvore, Random Forest, Gradient Boosting, KNN,
  AdaBoost e rede neural). Os mais complexos não superaram a regressão logística.
- Ajustamos os melhores com otimização bayesiana (Optuna) e conferimos com validação cruzada: o
  resultado é estável (0,061 ± 0,008) e a regressão logística não sobreajusta.
- No ponto de operação, 11% dos sinais vão para pares que são amigos, mais que o dobro da regra de
  afinidade (4,8%).

[Imagens: docs/img/cc4/pr_auc_ic.png e docs/img/cc6/estabilidade.png]

---

## Página 6 — Painel

O painel mostra, em tempo real, os encontros que chegam da pulseira, a chance de amizade de cada um,
os sinais emitidos e três indicadores:

- **Taxa de conversão:** encontros entre pessoas que são amigas (0,8%).
- **Precisão dos sinais:** quantos encontros sinalizados eram entre amigos.
- **Taxa de silenciamento:** quantos sinais foram seguidos de pausa da pulseira em até 24 h
  (fadiga de notificação).

[Imagem: captura do dashboard em http://localhost:8501 depois do `docker compose up`]

---

## Página 7 — Links

- Código: https://github.com/Lvcalabria06/opinion-ml
- Relatório (PDF, template SBC): `reports/relatorio_opinion.pdf` no repositório
- Como rodar: `cp .env.example .env && docker compose up --build`
