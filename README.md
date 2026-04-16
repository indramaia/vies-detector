# 🗞️ Viés Detector — Detecção Automatizada de Viés Editorial em Notícias Brasileiras

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-BERTimbau-orange)](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

Pipeline end-to-end para detecção e comunicação de viés editorial em veículos jornalísticos brasileiros, usando **BERTimbau** + **FactNews** + análise de espectro ideológico.

> Trabalho de Conclusão de Curso — Indra Seixas Neiva — USP (2026)

---

## 📐 Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────┐
│  Camada 1 · collector     RSS → metadados + SHA-256     │
│  Camada 2 · classifier    BERTimbau fine-tuned → rótulo │
│  Camada 3 · aggregation   sentenças → BiasScore [0,2]   │
│  Camada 4 · ideological   BiasScore → espectro [-1,+1]  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura do Projeto

```
vies_detector/
├── collector/          # Camada 1 – Coleta e pré-processamento RSS
│   ├── __init__.py
│   ├── rss_fetcher.py
│   ├── deduplicator.py
│   ├── preprocessor.py
│   └── sources.py
├── classifier/         # Camada 2 – Classificação sentence-level
│   ├── __init__.py
│   ├── sentence_classifier.py
│   ├── model_loader.py
│   └── train.py
├── aggregation/        # Camada 3 – BiasScore por artigo e veículo
│   ├── __init__.py
│   ├── bias_score.py
│   └── window_aggregator.py
├── ideological/        # Camada 4 – Contextualização ideológica
│   ├── __init__.py
│   ├── spectrum.py
│   ├── reference_map.py
│   └── data/
│       └── ideological_references.json
├── pipeline/           # Orquestração (Prefect)
│   ├── __init__.py
│   └── main_flow.py
├── api/                # REST API Flask
│   ├── __init__.py
│   └── app.py
├── tests/              # Testes unitários
│   ├── test_collector.py
│   ├── test_classifier.py
│   ├── test_aggregation.py
│   └── test_ideological.py
├── scripts/
│   ├── setup_db.py
│   └── run_pipeline.py
├── docs/
│   └── architecture.md
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Clone e instale dependências
git clone https://github.com/<seu-usuario>/vies-detector.git
cd vies-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab

# 2. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações

# 3. Inicialize o banco de dados
python scripts/setup_db.py

# 4. Execute o pipeline manualmente
python scripts/run_pipeline.py

# 5. Suba a API
python api/app.py
```

---

## 🤖 Treinamento do Classificador

```bash
# Faça download do FactNews https://github.com/franciellevargas/FactNews (VARGAS et al., 2023) e salve em data/factnews.csv

python classifier/train.py --data data/factnews.csv --output models/bertimbau-bias
```

Remapeamento para treino usando Pytorch:
| FactNews | Significado          | Índice PyTorch  | 
|----------|----------------------|-----------------|
|   – 1    | fortemente enviesada |      2          |
|     0    | factual              |      0          |
|     1    | enviesada            |      1          |

> O modelo treinado deve ser salvo em `models/bertimbau-bias/` e referenciado no `.env`.


---

## 📊 BiasScore

O BiasScore mede a **intensidade média de viés** de um artigo, ponderando cada sentença pelo grau de enviesamento detectado pelo classificador:
Sentenças factuais têm peso 0 e não elevam o score. O resultado varia de **0** (totalmente factual) a **2** (totalmente fortemente enviesado).

### Exemplo — artigo com 10 sentenças

| Composição                                      | Cálculo             | BiasScore |
|-------------------------------------------------|---------------------|-----------|
| 10 factuais                                     | (0 + 0) / 10        | 0.0       |
| 8 factuais · 2 enviesadas                       | (2×1 + 0) / 10      | 0.2       |
| 6 factuais · 4 enviesadas                       | (4×1 + 0) / 10      | 0.4       |
| 4 factuais · 4 enviesadas · 2 fort. enviesadas  | (4×1 + 2×2) / 10    | 0.8       |
| 10 fortemente enviesadas                        | (0 + 10×2) / 10     | 2.0       |

### Interpretação das faixas

| Faixa      | Interpretação                                                                 |
|------------|-------------------------------------------------------------------------------|
| 0.0 – 0.4  | Predominantemente factual — poucas sentenças enviesadas, intensidade leve     |
| 0.4 – 0.8  | Viés moderado — presença notável de sentenças enviesadas                      |
| 0.8 – 1.4  | Viés elevado — mistura significativa de enviesadas e fortemente enviesadas    |
| 1.4 – 2.0  | Linguagem fortemente enviesada — alta concentração de conteúdo carregado      |

> **Nota:** o BiasScore é uma estimativa probabilística baseada nos padrões aprendidos pelo modelo sobre o FactNews. Não representa um julgamento objetivo sobre a qualidade ou a veracidade do veículo.
 
---

## 🧪 Experimento de Treinamento

### Dataset e Divisão

O classificador foi treinado sobre o **FactNews** (Vargas et al., RANLP 2023),
composto por 6.191 sentenças de notícias brasileiras anotadas por especialistas.

| Conjunto   | Proporção | Sentenças | Papel                                              |
|------------|-----------|-----------|----------------------------------------------------|
| Treino     | 80%       | 4.952     | O modelo aprende os padrões linguísticos           |
| Validação  | 10%       | 619       | Monitora overfitting e decide quando parar         |
| Teste      | 10%       | 620       | Avaliação final — dados nunca vistos pelo modelo   |

A divisão é **estratificada por rótulo**: cada conjunto mantém a mesma
proporção de classes do dataset original, evitando avaliação enviesada.

| Classe               | Total | % do dataset |
|----------------------|-------|--------------|
| Factual              | 4.242 | 68,5%        |
| Fortemente enviesada | 1.391 | 22,5%        |
| Enviesada            | 558   | 9,0%         |

### Resultados

Modelo base: `neuralmind/bert-base-portuguese-cased` (BERTimbau)  
Tempo de treinamento: ~1h28min (CPU) · 5 épocas · batch efetivo 32

| Época | Macro-F1 (val) | Eval Loss |
|-------|---------------|-----------|
| 1     | 0.677         | 0.359     |
| 2     | 0.787         | **0.322** |
| 3     | 0.794         | 0.361     |
| 4     | 0.799         | 0.425     |
| 5     | **0.801**     | 0.449     |

### Desempenho no Teste (Macro-F1: **0.80**)

| Classe               | Precision | Recall | F1   | Suporte |
|----------------------|-----------|--------|------|---------|
| Factual              | 0.92      | 0.95   | 0.94 | 425     |
| Enviesada            | 0.57      | 0.50   | 0.53 | 56      |
| Fortemente enviesada | 0.95      | 0.91   | 0.93 | 139     |
| **Macro avg**        | **0.82**  | **0.79**| **0.80** | 620 |

> **Interpretação:** o modelo performa bem nas classes extremas (factual e
> fortemente enviesada), mas tem dificuldade com a classe intermediária
> "enviesada" (F1=0.53) — reflexo direto do desequilíbrio do dataset, onde
> essa classe representa apenas 9% das sentenças. Esse comportamento é
> esperado e está documentado como limitação do sistema.

---



## ⚠️ Limitações e Uso Responsável

- O BiasScore é uma **estimativa probabilística**, não um julgamento objetivo.
- O modelo aprende padrões dos anotadores do FactNews, que possuem perspectivas teóricas próprias.
- Risco de *shortcut learning*: menção de certos termos políticos pode inflar o score.
- Não utilize esta ferramenta para deslegitimar veículos de comunicação.

---

## 📜 Referências Principais

- SOUZA et al. BERTimbau (2020)
- VARGAS et al. FactNews (2023)
- ENTMAN, R. Framing Theory (1993)
- GEIRHOS et al. Shortcut Learning (2020)

---

## Licença

© Indra Seixas Neiva — USP 2026
"# vies-detector" 
"# vies-detector" 
