# Fontes de dados reais

O professor avisou que **dataset inventado perde pontos**. A pulseira e o OpiniOn ainda não
têm usuários, então o gosto, o catálogo e as avaliações passam a vir de **datasets públicos
de outras empresas**, com a fonte citada. Encontros, interações no app e o label `conectou`
continuam simulados: nenhum dataset público tem "dois usuários do OpiniOn se encontraram
e depois interagiram no app".

## Decisão: um dataset unificado, Amazon Reviews 2023

**Fonte:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab
(UC San Diego). Arquivos 5-core (`benchmark/5core/rating_only/*.csv.gz`) e metadados
(`raw/meta_categories/meta_*.jsonl.gz`).

**Citação:** Hou, Y.; Li, J.; He, Z.; Yan, A.; Chen, X.; McAuley, J. *Bridging Language
and Items for Retrieval and Recommendation*. arXiv:2403.03952, 2024.

**Licença:** dataset acadêmico publicado pelo McAuley Lab para pesquisa. Uso neste projeto
é acadêmico (disciplina de CC). Não redistribuímos os arquivos brutos (~5 GB); o repositório
só versiona a amostra filtrada em `data/reais/amazon2023/` (cerca de 1.200 usuários).

### Por que um só, e por que este

O Diego deixou a critério da equipe juntar várias fontes ou tratar cada mídia separada.
Juntar Spotify + Netflix + Steam **inventa** que pessoas diferentes são a mesma. A Amazon
é o único conjunto grande em que o **mesmo usuário real** avalia livro, filme/série, jogo
e música, com nota 1–5 e data. Isso é o perfil de gosto que o OpiniOn precisa.

| Categoria 5-core | Mídia no OpiniOn | Notas (5-core) |
|---|---|---|
| Books | `livro` | 9,5 M |
| Movies_and_TV | `filme` ou `serie` | 7,4 M |
| Video_Games | `jogo` (só a categoria Games; acessórios fora) | 0,8 M |
| CDs_and_Vinyl | `musica` | 1,6 M |

Há 14.589 usuários elegíveis na janela (ativos, ≥2 mídias). Desses, **1.200** entram no
projeto (semente 42), com cota por mídia dominante. Amostra versionada: 48.018 obras e
58.251 avaliações reais (`data/reais/amazon2023/fonte.json`).

Cerca de 1/3 das obras cai em `outros_<midia>` (a Amazon não tinha categoria que
mapeasse para os gêneros do OpiniOn). A compatibilidade ainda usa a mídia e as que
têm gênero reconhecido.

### O que é real e o que é transformação nossa

| Campo | Origem |
|---|---|
| usuário, obra, nota, data da avaliação | real (Amazon) |
| título e categorias da obra | real (metadados Amazon) |
| mídia (`livro`/`filme`/`serie`/`jogo`/`musica`) | mapeamento nosso, regras em `config/fontes.yaml` |
| gêneros do OpiniOn | mapeamento das categorias Amazon → lista em `config/fontes.yaml` |
| calendário 2026 | deslocamento de **semanas inteiras** (261 semanas: 2021-07-05 → 2026-07-06). Dia da semana, hora e a ordem entre as pessoas são os reais |
| encontros, seguidores, interações, label | **ainda simulados** (`opinion/gerar_bootstrap.py`) |

### Janela

Notas entre 2021-01-04 e 2021-09-27 (26 semanas de histórico + 12 de simulação). Essa janela
tem gente ativa o bastante depois do pico de 2014–2016, e ainda inclui as quatro mídias.

## Como (re)gerar

```bash
python -m opinion.fontes.amazon2023 tudo     # baixa ~5 GB para data/externo/ (fora do Git) e gera data/reais/amazon2023/
python -m opinion.gerar_bootstrap            # usa os reais (config/simulacao.yaml → usuarios.fonte)
```

`data/externo/` está no `.gitignore`. `data/reais/amazon2023/` vai para o Git (amostra pequena,
com `fonte.json` descrevendo o filtro). Sem a amostra, o simulador recusa rodar.

## Candidatos considerados e por que não

| Dataset | Motivo de não usar como fonte principal |
|---|---|
| MovieLens | só filme |
| Goodreads / UCSD Book Graph | só livro |
| Steam (UCSD) | só jogo; tempo de jogo, não nota 1–5 |
| Last.fm | só música; plays, não nota |
| Netflix Prize | só filme/série; usuários não cruzam com outras mídias |
| Copenhagen Networks Study | proximidade Bluetooth real — candidato para **encontros** no futuro, não para gosto |

## Limitação

Quem avalia na Amazon não é exatamente o público do OpiniOn. O gosto é real; a chance de
dois se encontrarem e conectarem no app continua sendo a simulação. Qualquer métrica do
modelo ainda mede essa mistura, não usuários reais da pulseira.
