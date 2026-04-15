# Arquitetura do Viés Detector

## Visão Geral

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PIPELINE PRINCIPAL                           │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────┐  │
│  │  CAMADA 1   │──▶│   CAMADA 2   │──▶│  CAMADA 3   │──▶│CAM. 4 │  │
│  │  collector  │   │  classifier  │   │ aggregation │   │ideolog.│  │
│  └─────────────┘   └──────────────┘   └─────────────┘   └────────┘  │
│     RSS + SHA256     BERTimbau           BiasScore [0,2]  [-1,+1]   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   SQLite / MySQL   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │    REST API Flask  │
                          └───────────────────┘
```

## Camada 1 — Collector

**Responsabilidade:** Coletar notícias via RSS, deduplicar e pré-processar.

- `sources.py` — catálogo de feeds RSS por veículo
- `deduplicator.py` — hash SHA-256 para identificação única de artigos
- `preprocessor.py` — limpeza de HTML, remoção de URLs, tokenização NLTK
- `rss_fetcher.py` — orquestra a coleta com tratamento de erros e retry

**Conformidade LGPD:**
- Armazena apenas snippet ≤ 500 caracteres (nunca o artigo completo)
- Hash SHA-256 derivado de URL público, sem dados pessoais

## Camada 2 — Classifier

**Responsabilidade:** Classificar cada sentença em factual / enviesada / fortemente enviesada.

- `model_loader.py` — carrega BERTimbau fine-tuned (singleton com `@lru_cache`)
- `sentence_classifier.py` — inferência em batch com softmax
- `train.py` — fine-tuning sobre FactNews com MLflow + early stopping

**Modelo:** `neuralmind/bert-base-portuguese-cased` (SOUZA et al., 2020)  
**Dados:** FactNews, 6.191 sentenças anotadas (VARGAS et al., 2023)  
**Métrica principal:** Macro-F1 (trata classes com igual peso)

## Camada 3 — Aggregation

**Responsabilidade:** Agregar classificações sentence-level em índice por artigo e por veículo.

- `bias_score.py` — fórmula: `(1×n_env + 2×n_fort) / n_total` → [0, 2]
- `window_aggregator.py` — média, mediana, desvio padrão por veículo em janela temporal

**Faixas interpretativas:**

| Faixa      | Interpretação                    |
|------------|----------------------------------|
| 0.0 – 0.4  | Predominantemente factual        |
| 0.4 – 0.8  | Viés moderado                    |
| 0.8 – 1.4  | Viés elevado                     |
| 1.4 – 2.0  | Linguagem fortemente enviesada   |

## Camada 4 — Ideological

**Responsabilidade:** Contextualizar o BiasScore com posicionamento ideológico acadêmico.

- `reference_map.py` — carrega `ideological_references.json` (cacheado)
- `spectrum.py` — gera `IdeologicalContext` com texto interpretativo e caveat ético

**Escala:** [-1.0, +1.0]  
**Fontes:** Manchetômetro (FERES JÚNIOR et al., 2013-), GPOPAI/USP (ORTELLADO; RIBEIRO, 2018)

## Banco de Dados

| Tabela              | Conteúdo                                           |
|---------------------|----------------------------------------------------|
| `articles`          | Metadados + snippet + BiasScore por artigo         |
| `sentences`         | Classificação de cada sentença (label + scores)    |
| `vehicle_indices`   | Índice editorial por veículo (atualizado por run)  |

## API REST

| Método | Endpoint                      | Descrição                          |
|--------|-------------------------------|------------------------------------|
| GET    | `/api/health`                 | Status da API                      |
| GET    | `/api/vehicles`               | Todos os índices editoriais        |
| GET    | `/api/vehicles/<id>`          | Índice de um veículo               |
| GET    | `/api/spectrum`               | Espectro ideológico ordenado       |
| GET    | `/api/articles?source=<id>`   | Artigos recentes por veículo       |

## Limitações Documentadas

1. **Defasagem temporal:** FactNews cobre 2014–2022
2. **Shortcut learning:** risco de associação espúria léxico-rótulo (GEIRHOS et al., 2020)
3. **Subjetividade da anotação:** modelo aprende padrões dos anotadores, não "o viés real"
4. **Cobertura de veículos:** apenas grandes jornais impressos nacionais no corpus de treino
5. **Posicionamento ideológico estático:** mapa de referências não é dinâmico
