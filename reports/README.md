# reports/ — relatório do projeto

- **[`relatorio_opinion.pdf`](relatorio_opinion.pdf)**: relatório final, no template da SBC
  para artigos (`sbc-template.sty`, versão de dezembro de 2017), sem alteração de margens, fontes,
  espaçamento ou número de colunas.
- `latex/`: fonte do relatório (`relatorio_opinion.tex`, `referencias.bib`, figuras), com o
  `sbc-template.sty` e o `sbc.bst` originais.

Para gerar de novo:

```bash
cd reports/latex
pdflatex relatorio_opinion && bibtex relatorio_opinion && pdflatex relatorio_opinion && pdflatex relatorio_opinion
mv relatorio_opinion.pdf ..
```

> **Confiram o template.** A especificação exige o template SBC **anexo a ela**. O usado aqui é o
> template SBC clássico (o mesmo do Overleaf "SBC Conferences Template"). Se o anexo for outra
> versão (por exemplo, o modelo SBC 2024 da SOL), troquem o `sbc-template.sty` pelo anexo e
> recompilem: o texto não depende de nada específico desta versão.

Os números do relatório vêm de `docs/comparacao_modelos.md`, `docs/otimizacao.md` e
`docs/validacao_cruzada.md` (rodada de 28/11/2026).
