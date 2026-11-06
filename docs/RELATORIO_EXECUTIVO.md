# Relatório executivo: IA para apontar perturbações na Amazônia sem precisar de exemplos

**Grupo 1 · IBM8924 Projeto de Deep Learning · Gabriel Pow, Pedro Pintor · novembro de 2026**

---

## O que foi feito?

Construímos um sistema que analisa imagens de satélite da Amazônia a cada três meses e dá uma "nota de estranheza"
para cada quadrado de 320 m. O sistema aprende só como a floresta normalmente se comporta, sem receber exemplos de
desmatamento, e foi comparado a métodos simples que já existem, numa área de teste no Tapajós (PA).

## Por que importa?

- **Os alertas oficiais chegam tarde e deixam passar eventos pequenos.** O PRODES (INPE) é anual. O DETER é mais
  rápido, mas não registra boa parte dos cortes pequenos: no Tapajós, só 37% das áreas desmatadas no PRODES
  também tinham alerta DETER.
- **Novos tipos de dano escapam de sistemas treinados com exemplos.** Um sistema que aprende "o que é normal" pode,
  em princípio, apontar garimpo, fogo ou degradação de formas ainda não catalogadas.
- **Detectar mais cedo reduz o custo da fiscalização.** Cada trimestre de antecedência é tempo para agir antes de o
  dano se consolidar.
- **Custo baixo.** Tudo roda num notebook comum, sem placa de vídeo, com imagens públicas e gratuitas
  (Sentinel, da Agência Espacial Europeia) e um modelo de IA aberto (TerraMind, da IBM e ESA).

## Quais foram os resultados?

> **Na área de teste, a IA ainda não superou o método simples.** O resultado definitivo depende de processar as
> áreas completas, o que ainda não foi feito.

| | Método simples (queda do verde) | Nossa IA |
|---|---|---|
| **Capacidade de separar eventos de áreas normais** (0,5 = acaso; 1 = perfeito) | **0,77** | 0,67 |
| Fogo | **0,90** | 0,77 |
| Garimpo | **0,65** | 0,49 |
| Eventos detectados com 1 alarme falso a cada 100 áreas | 0 de 29 | 0 de 29 |

O que isso quer dizer:

- **A IA funciona melhor que o acaso**, mas, com poucos alarmes falsos, ainda não detecta os eventos.
- **A diferença não é conclusiva.** Só havia 29 eventos para testar, e as margens de erro dos dois métodos se
  sobrepõem.
- **Garimpo é o ponto fraco de todos.** O método simples cai para 0,65 porque, no garimpo, o verde nem sempre
  diminui. Num teste interno com mais eventos, combinar a IA com indicadores clássicos melhorou a detecção de
  garimpo, mas isso não se repetiu na área de teste.
- **Parte dos "erros" da IA são acertos:** ela marcou garimpos visíveis nas imagens que nenhum mapa oficial tinha
  registrado.
- **Testamos só 1,3% da área planejada** (256 de 19.445 quadrados de 2,24 km), por limites de tempo e de
  infraestrutura.

O teste foi feito com regras definidas **antes** de ver os resultados, em áreas e períodos que a IA nunca viu, e
na proporção real de eventos (cerca de 5%). Por isso os números são menos favoráveis, mas confiáveis.

## O que recomendamos?

1. **Processar as áreas completas** (Tapajós e Rondônia). O código já está pronto; leva cerca de 3 dias de
   processamento. Esse é o teste que decide se a IA vale a pena.
2. **Não colocar a IA em operação ainda.** Hoje, o método simples (queda do verde) é igual ou melhor e muito mais
   barato.
3. **Se a combinação confirmar a vantagem em garimpo nas áreas completas, usá-la junto com o método simples**, não no
   lugar dele: cada um acerta tipos de evento diferentes.
4. **Medir a antecedência em relação ao DETER**, que é o ganho prático mais relevante para a fiscalização.
5. **Revisar os "alarmes falsos" com especialistas.** Se parte deles for garimpo sem registro, a IA já tem valor como
   ferramenta de triagem.

---

Detalhes técnicos: [Relatório técnico](RELATORIO_TECNICO.md) · [README](../README.md)
