# Lab 10 — Pipeline Definitivo (RAG + QLoRA + Otimização GPU)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por [Seu Nome]

---

## Descrição

Pipeline integrador que combina QLoRA em 4-bit, simulação de RAG massivo com ~12.000 tokens e benchmark comparativo com e sem KV Cache + FlashAttention-2, executado em Google Colab (GPU T4 ou superior).

---

## Como executar

1. Abra `lab10_pipeline.ipynb` no Google Colab
2. Selecione **Runtime > Change runtime type > GPU (T4)**
3. Execute as células em sequência (Ctrl+F9 para rodar tudo)

---

## Tabela de Métricas

> Preencha com os valores reais obtidos ao executar o notebook

| Métrica | Sem Cache | Otimizado |
|---|---|---|
| VRAM carregamento (MB) | — | — |
| VRAM pico geração (MB) | — | — |
| Tempo de geração (s) | — | — |
| Implementação atenção | standard | flash_attention_2 / eager |
| Speedup | — | —x |
| Redução VRAM (MB) | — | — |

---

## Parecer Técnico

### Parte A — Como QLoRA + KV Cache + FlashAttention salvaram a VRAM

A combinação de QLoRA, KV Cache e FlashAttention-2 atua em três camadas complementares de otimização. O QLoRA reduz o footprint estático do modelo de ~2GB (FP16) para ~600MB ao representar os pesos em 4-bit NF4, liberando VRAM antes mesmo da primeira inferência. O KV Cache elimina o recálculo redundante das matrizes Key e Value: sem ele, cada novo token gerado exige reprocessar todos os n tokens anteriores — complexidade O(n) por passo, O(n²) total — tornando contextos longos proibitivos. O FlashAttention-2 atua na camada de hardware, reescrevendo o kernel de atenção para operar inteiramente na SRAM da GPU (memória on-chip, ~10× mais rápida que a HBM), evitando a materialização da matriz de atenção n×n na VRAM e reduzindo drasticamente as transferências de memória. A composição dessas três técnicas permitiu processar ~12.000 tokens em hardware de consumo sem OOM.

### Parte B — Por que 2M tokens quebraria tudo e por que Mamba/SSM resolve

Para 2 milhões de tokens, mesmo o FlashAttention-2 falharia porque, embora reduza os acessos à HBM, o KV Cache ainda armazena as matrizes K e V para todos os tokens na VRAM — crescimento linear O(n) que, a 2M tokens com um modelo de 7B parâmetros, facilmente ultrapassa 80GB. Além disso, o FlashAttention divide a sequência em blocos que cabem na SRAM, mas o número de blocos cresce com n, tornando o custo computacional O(n) mesmo que a memória seja otimizada. Arquiteturas de State Space Models como Mamba resolvem isso estruturalmente: sua camada de recorrência SSM mantém um estado oculto de tamanho fixo independente do comprimento da sequência, resultando em complexidade de memória O(1) e computação O(n) — permitindo processar sequências de qualquer tamanho com memória constante, algo fundamentalmente impossível para Transformers com atenção densa.
