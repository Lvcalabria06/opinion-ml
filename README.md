# OpiniOn — Ciência da Computação · Sprint 1 (CC1)

**Pergunta do modelo (seção 3.4):** dado que dois usuários acabaram de se encontrar, qual a chance
de eles interagirem no OpiniOn nas próximas 48 horas?

Para chegar lá, o primeiro passo (CC1) é ter **dados** e o **caminho por onde eles chegam**.
Esta pasta entrega exatamente isso:

| Requisito CC1 (seção 4.6) | Onde está |
|---|---|
| Pulseira virtual publicando encontros, interesses e status via MQTT | `opinion/pulseira_virtual.py` + contrato em `docs/contrato_mqtt.md` |
| Serviço de ingestão e persistência em banco | `opinion/ingestao.py` → tabelas no banco |
| Dataset simulado de 12 semanas (usuários, encontros e interações) | `opinion/gerar_bootstrap.py` → `data/bootstrap/` |
| **Checkpoint 1**: encontro publicado → broker → banco → aparece numa consulta | `opinion/checkpoint1.py` |

Ficam de fora a pulseira física, o firmware e a interface, que são do Design. O modelo, o
Airflow e o MLflow ficam para os próximos sprints.

---

## Como as peças se encaixam

```
                          ┌──────────────────────────────┐
  gerar_bootstrap.py ───► │ data/bootstrap/              │
  (simula 12 semanas)     │  historico_app/  pulseira/   │
                          └──────┬──────────────┬────────┘
                                 │              │
         carregar_historico_app  │              │  pulseira_virtual.py replay
                                 │              ▼
                                 │       ┌─────────────┐
                                 │       │ broker MQTT │ ◄── pulseira_virtual.py ao-vivo
                                 │       └──────┬──────┘     (encontros novos, agora)
                                 │              ▼
                                 │        ingestao.py  (valida, traduz IDs, grava)
                                 ▼              ▼
                          ┌──────────────────────────────┐
                          │ BANCO                        │
                          │ usuarios, avaliacoes,        │  ◄── Sprint 2: Airflow lê daqui
                          │ seguidores, interacoes,      │
                          │ encontros, interesses, status│
                          └──────────────────────────────┘
```

Há duas origens de dados, como no sistema real:

- **Histórico do app** (avaliações, quem segue quem, interações): o app já guardaria isso no banco dele. Aqui vem do simulador e é carregado direto.
- **Pulseira** (encontros, toques, modo): chega **sempre pelo MQTT**. Até as 12 semanas do bootstrap passam pelo broker e pela ingestão (modo *replay*), então existe um único pipeline de dados.

## O que o simulador gera

São 1.200 usuários e 12 semanas (06/07 a 27/09/2026), com estes volumes:

| | |
|---|---|
| Encontros | 55.363 |
| Interações no app | 43.070 |
| Avaliações | 60.787 |
| Toques de interesse | 6.765 |
| Trocas de modo | 2.396 |

Como o mundo simulado funciona:

- **Personas:** os usuários seguem 5 personas provisórias (gamer social, cinéfilo, leitor reservado, estudante casual e maratonista de séries). **Devem ser trocadas pelas personas do Design** em `config/simulacao.yaml`.
- **Lugares:** as pessoas vão a cinema, livraria, campus e, nas semanas 3, 7 e 11, a uma convenção de games. Quem está no mesmo lugar e turno pode se encontrar, e metade dos usuários tem um grupo de amigos que sai junto.
- **Modo pausado:** pulseiras pausadas não registram encontros.
- **Label (`conectou`):** um encontro "vira conexão" por sorteio, com chance que depende de variáveis **ocultas** (abertura social e compatibilidade verdadeira de gostos), duração, local, encontros anteriores e ruído. Não é uma regra que o modelo possa simplesmente redescobrir (seção 4.3).
- **Resultado:** **13,7% dos encontros** têm `conectou = 1`, um problema desbalanceado. Amigos interagem de qualquer jeito (28% de positivos), desconhecidos bem menos (8%), e 3,5% das conexões só acontecem depois das 48 h.

Num teste rápido, uma regressão logística simples já separa razoavelmente as classes
(ROC-AUC ≈ 0,78 nas semanas 9–12), mas está longe de acertar tudo. É um ponto de partida
realista para o baseline do CC3.

Todos os parâmetros são **premissas da equipe**, comentadas em `config/simulacao.yaml`. A
pesquisa de monetização não é usada como alvo. Os números completos estão em
`data/bootstrap/relatorio_simulacao.md`, e as colunas de cada arquivo em `docs/dicionario_dados.md`.

---

## Como testar

### Opção A — com Docker (um comando)

```bash
cp .env.example .env        # uma vez: credenciais e segredo locais (o .env não vai para o Git)
docker compose up --build
```

Isso sobe o broker e o PostgreSQL, carrega o histórico e liga a ingestão. Em seguida faz o
replay das 12 semanas (cerca de 2 minutos) e deixa a pulseira ao vivo gerando encontros. Em
outro terminal:

```bash
docker compose exec ingestao python -m opinion.checkpoint1        # Checkpoint 1
docker compose exec ingestao python -m opinion.verificar_replay   # chegou tudo?
docker compose exec ingestao python -m opinion.consultar          # o que tem no banco
```

O banco também pode ser visto no navegador em **http://localhost:8080**. Use Sistema
*PostgreSQL*, servidor `postgres` e `opinion` como usuário, senha e base.

### Opção B — sem Docker (Python + SQLite)

Precisa de Python 3.10+ e de um broker MQTT. A forma mais simples de ter o broker:

```bash
docker run -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
```

Também dá para instalar o Mosquitto pelo site mosquitto.org (Windows) ou com `brew install mosquitto` (Mac).

```bash
pip install -r requirements.txt
cp .env.example .env                            # 0. configuração local (uma vez)

python -m opinion.carregar_historico_app        # 1. cria data/opinion.db com o histórico do app
python -m opinion.ingestao                      # 2. TERMINAL 1 — fica escutando o MQTT
python -m opinion.pulseira_virtual replay       # 3. TERMINAL 2 — publica as 12 semanas (~2 min)
python -m opinion.verificar_replay              # 4. confere se chegou tudo
python -m opinion.checkpoint1                   # 5. Checkpoint 1
python -m opinion.consultar                     # 6. resumo do banco
python -m opinion.pulseira_virtual ao-vivo      # 7. (opcional) encontros novos em tempo real
```

O dataset já vem pronto. Só rode `python -m opinion.gerar_bootstrap` se mudar as premissas.

### O que vocês devem ver

**Passo 4** (`verificar_replay`):
```
Arquivo do bootstrap  x  banco
  encontros    55,363   55,363   OK
  interesses    6,765    6,765   OK
  status        2,396    2,396   OK

IDs rotativos traduzidos corretamente: 55,363 de 55,363 encontros
RESULTADO: tudo certo ✔
```

**Passo 5** (`checkpoint1`):
```
1) Publicando encontro de teste
   tópico : opinion/pulseira/PUL-0001/encontro
2) Broker confirmou o recebimento (PUBACK, QoS 1): sim
3) Procurando o encontro no banco
4) Encontrado com a consulta: SELECT * FROM encontros WHERE encontro_id = 'e-checkpoint-...'
   usuario_a    usr-00001
   ...
CHECKPOINT 1 OK ✔
```

**Robustez:** rodem `python -m opinion.pulseira_virtual ao-vivo --duplicar 0.05 --invalidas 0.05`.
A pulseira passa a mandar mensagens repetidas e com defeito de propósito. No `consultar`, as
repetidas aparecem como ignoradas e as defeituosas como rejeitadas, com o motivo de cada uma.

---

## Estrutura

```
config/simulacao.yaml        premissas do simulador (personas, locais, label)
opinion/contrato.py          tópicos, validação e IDs rotativos (contrato MQTT)
opinion/gerar_bootstrap.py   simulador das 12 semanas
opinion/pulseira_virtual.py  pulseira virtual: replay e ao vivo
opinion/ingestao.py          serviço de ingestão MQTT -> banco
opinion/db.py                esquema do banco (PostgreSQL ou SQLite)
opinion/carregar_historico_app.py, consultar.py, checkpoint1.py, verificar_replay.py
docs/contrato_mqtt.md        contrato para combinar com o Design
docs/dicionario_dados.md     arquivos, colunas e cuidados para o Sprint 2
data/bootstrap/              dataset gerado
docker-compose.yml           tudo com um comando
```

## Testado / não testado

- **Testado:** opção B completa, com SQLite e broker MQTT local. Inclui replay das 12 semanas sem perdas, tradução correta de 100% dos IDs, Checkpoint 1, modo ao vivo e injeção de falhas.
- **Não testado:** `docker compose` e PostgreSQL, porque o ambiente onde isso foi escrito não tinha Docker. Rodem uma vez antes de apresentar.

## Pendências com o Design (seção 3.5)

- Personas definitivas para substituir as provisórias.
- Fechar o tópico `feedback` (padrões de luz, calor e vibração).
- Conferir se o protótipo físico consegue publicar no mesmo contrato.
