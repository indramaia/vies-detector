# 🗞️ Viés Detector — Detecção Automatizada de Viés Editorial em Notícias Brasileiras

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace](https://img.shields.io/badge/🤗-IndraSeixas/bertimbau--bias-orange)](https://huggingface.co/IndraSeixas/bertimbau-bias)

Pipeline end-to-end para detecção e comunicação de viés editorial em veículos jornalísticos brasileiros, usando **BERTimbau** fine-tuned no **FactNews** + espectro ideológico curado + API pública.

> Trabalho de Conclusão de Curso — Indra Seixas Neiva — USP (2026)

Demo: [biasradar.lovable.app](https://biasradar.lovable.app) · API: [vies-detector.onrender.com](https://vies-detector.onrender.com)

---

## 📐 Arquitetura em Camadas

```
┌──────────────────────────────────────────────────────────────────────┐
│  Camada 1 · collector     RSS + scraping HTTP → metadados + SHA-256  │
│  Camada 2 · classifier    BERTimbau fine-tuned → rótulo por sentença │
│  Camada 3 · aggregation   sentenças → BiasScore [0,2] + rs_factor    │
│  Camada 4 · ideological   mapa curado → IdeologyScore [−1,+1]        │
│  Camada 5 · api           Flask + cache SWR → endpoints REST         │
└──────────────────────────────────────────────────────────────────────┘
```

**Pipeline PLN:**
`Coleta RSS/scraping → segmentação NLTK Punkt → classificação BERTimbau fine-tuned → persistência seletiva (Neon PostgreSQL)`

---

## 📁 Estrutura do Projeto

```
vies_detector/
├── collector/                  # Camada 1 – Coleta e pré-processamento
│   ├── __init__.py
│   ├── sources.py              # Catálogo de veículos (RSS + homepage-based)
│   ├── rss_fetcher.py          # Coleta RSS + scraping paralelo por artigo
│   ├── article_scraper.py      # Scraping HTTP (requests + BeautifulSoup)
│   ├── deduplicator.py         # Deduplicação por hash SHA-256
│   └── preprocessor.py         # NLTK Punkt + limpeza de boilerplate
├── classifier/                 # Camada 2 – Classificação sentence-level
│   ├── __init__.py
│   ├── sentence_classifier.py  # Inferência BERTimbau + rs_factor
│   ├── model_loader.py         # Carregamento do modelo (HuggingFace Hub)
│   └── train.py                # Fine-tuning no FactNews
├── aggregation/                # Camada 3 – BiasScore por artigo e veículo
│   ├── __init__.py
│   ├── bias_score.py           # Fórmula BiasScore + detecção discurso reportado
│   ├── window_aggregator.py    # Média móvel por veículo (janela 30 dias)
│   └── topic_clusterer.py      # Agrupamento TF-IDF para /api/stories e /api/topics
├── ideological/                # Camada 4 – Contextualização ideológica
│   ├── __init__.py
│   ├── spectrum.py
│   ├── reference_map.py
│   └── data/
│       └── ideological_references.json   # Mapa curado por veículo
├── pipeline/                   # Orquestração do pipeline
│   ├── __init__.py
│   └── main_flow.py            # Fluxo completo: coleta → classifica → persiste → agrega
├── api/                        # REST API Flask
│   ├── __init__.py
│   └── app.py                  # Endpoints + cache TTL + SWR + pre-warm
├── scripts/
│   ├── setup_db.py             # Criação das tabelas (SQLAlchemy + Neon)
│   ├── run_pipeline.py         # Entrada do pipeline (GitHub Actions)
│   └── setup_cronjob.py        # Registro do keep-alive no cron-job.org
├── .github/
│   └── workflows/
│       ├── pipeline.yml        # Execução automática 2× ao dia
│       └── keepalive.yml       # Ping /api/warmup a cada 5 min
├── tests/
│   ├── test_collector.py
│   ├── test_classifier.py
│   ├── test_aggregation.py
│   └── test_ideological.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Clone e instale dependências
git clone https://github.com/IndraSeixas/vies-detector.git
cd vies-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab

# 2. Configure variáveis de ambiente
cp .env.example .env
# Edite .env: DATABASE_URL (Neon PostgreSQL)

# 3. Inicialize o banco de dados
python scripts/setup_db.py

# 4. Execute o pipeline manualmente
python scripts/run_pipeline.py

# 5. Suba a API localmente
python api/app.py
```

**Produção:** o pipeline roda automaticamente via GitHub Actions (`pipeline.yml`) 2× ao dia. O keep-alive (`keepalive.yml`) faz ping em `/api/warmup` a cada 5 minutos para evitar cold start no Render e autosuspend no Neon.

---

## 🤖 Treinamento do Classificador

```bash
# Faça download do FactNews (VARGAS et al., 2023)
# https://github.com/franciellevargas/FactNews
# Salve em data/factnews.csv

python classifier/train.py --data data/factnews.csv --output models/bertimbau-bias
```

O modelo treinado é publicado no HuggingFace Hub (`IndraSeixas/bertimbau-bias`) e baixado automaticamente pelo pipeline via `huggingface_hub.snapshot_download`.

**Remapeamento de classes para treino (PyTorch):**

| FactNews | Significado          | label_id |
|:--------:|----------------------|:--------:|
|   −1     | fortemente enviesada |    2     |
|    0     | factual              |    0     |
|    1     | enviesada            |    1     |

---

## 🌐 API REST

Hospedada em [vies-detector.onrender.com](https://vies-detector.onrender.com).

| Endpoint | Descrição |
|---|---|
| `GET /api/health` | Status da API + diagnóstico de cache |
| `GET /api/warmup` | Keep-alive: recarrega caches expirados ou faz SELECT 1 no Neon |
| `GET /api/stats` | Totais globais (artigos, sentenças, veículos) |
| `GET /api/vehicles` | Índice editorial de todos os veículos |
| `GET /api/vehicles/<ideology_id>` | Índice de um veículo específico |
| `GET /api/spectrum` | Veículos ordenados no espectro ideológico |
| `GET /api/articles?source=<id>` | Artigos recentes de um veículo |
| `GET /api/stories` | Stories multi-veículo agrupadas por TF-IDF |
| `GET /api/topics/<slug>` | Stories filtradas por tópico curado (mapa de sinônimos) |
| `GET /api/articles/<url_hash>/similar` | Artigos similares por TF-IDF cosine |

**Tópicos disponíveis em `/api/topics/<slug>`:**
`eleicoes-2026` · `stf` · `lula` · `bolsonaro` · `reforma-tributaria` · `petrobras` · `banco-central` · `camara` · `amazonia` · `pib` · `copa-mundo` · `seguranca-publica`

**Resiliência:** cache TTL in-memory (15 min veículos/espectro, 5 min stories) + Stale-While-Revalidate + pre-warm no startup + fallback em `ideological_references.json`.

---

## 📊 BiasScore

O BiasScore mede a **intensidade média de viés** de um artigo. Cada sentença contribui com peso proporcional à classe detectada, **atenuado pelo fator de discurso reportado** (`rs_factor`):

```
BiasScore = Σ(CLASS_WEIGHT[label] × rs_factor) / n_sentenças

CLASS_WEIGHTS: factual = 0.0 · enviesada = 1.0 · fortemente_enviesada = 2.0
```

**Detecção de discurso reportado (`rs_factor`):**
Sentenças atribuídas a fontes externas (aspas balanceadas, verbos de atribuição como *disse/afirmou*, locuções como *segundo/de acordo com*) recebem `rs_factor = 0.4` — contribuem apenas 40% ao BiasScore, pois o viés pertence à fonte citada, não ao veículo.

**Primeiras 100 sentenças de cada artigo são classificadas** — o jornalismo concentra enquadramento no lide.

### Exemplo — artigo com 10 sentenças (sem discurso reportado)

| Composição | Cálculo | BiasScore |
|---|---|:---:|
| 10 factuais | (10×0) / 10 | 0.00 |
| 8 factuais · 2 enviesadas | (2×1.0) / 10 | 0.20 |
| 6 factuais · 4 enviesadas | (4×1.0) / 10 | 0.40 |
| 4 factuais · 4 enviesadas · 2 fort. enviesadas | (4×1.0 + 2×2.0) / 10 | 0.80 |
| 10 fortemente enviesadas | (10×2.0) / 10 | 2.00 |

### Faixas interpretativas

| Faixa | BiasScore | % normalizado | Interpretação |
|---|:---:|:---:|---|
| Nível de viés baixo | 0.00 – 0.33 | 0 – 17% | Quase todas as sentenças factuais |
| Nível de viés moderado | 0.33 – 0.67 | 17 – 33% | Linguagem opinativa presente, factual ainda domina |
| Nível de viés alto | 0.67 – 1.33 | 33 – 67% | Linguagem enviesada predomina |
| Nível de viés muito alto | 1.33 – 2.00 | 67 – 100% | Maioria das sentenças com viés forte |

> **Nota:** o BiasScore é uma estimativa probabilística baseada nos padrões aprendidos no FactNews. Não representa julgamento sobre a veracidade ou qualidade do veículo.

---

## 🗺️ Veículos Monitorados e Posicionamento Ideológico

O sistema monitora **18 veículos** ativos. O IdeologyScore é definido por **mapa de referência curado** (`ideological_references.json`), fundamentado em literatura acadêmica sobre posicionamento editorial da imprensa brasileira — não é calculado pelo modelo em tempo real.

![Espectro Ideológico dos Veículos Monitorados](docs/spectrum.png)

> Para regenerar o gráfico: `python scripts/generate_spectrum_chart.py`

**Esquerda** (score < −0.55)

| Veículo | Score¹ | Incerteza | Base | Fontes |
|---|:---:|:---:|:---:|---|
| [Brasil de Fato](https://www.brasildefato.com.br) | −0.75 | ±0.10 | Direta | Ortellado & Ribeiro (2018); Intervozes (2017) |
| [Outras Palavras](https://outraspalavras.net) | −0.75 | ±0.15 | Direta | Ortellado & Ribeiro (2018); Intervozes (2017) |
| [Carta Capital](https://www.cartacapital.com.br) | −0.75 | ±0.10 | Direta | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |
| [Le Monde Diplomatique Brasil](https://diplomatique.org.br) | −0.65 | ±0.15 | Direta | Ortellado & Ribeiro (2018); Feres Júnior et al. (2013) |
| [The Intercept Brasil](https://theintercept.com/brasil) | −0.60 | ±0.20 | Direta | Ortellado & Ribeiro (2018) |

**Centro-esquerda** (−0.55 ≤ score < −0.10)

| Veículo | Score¹ | Incerteza | Base | Fontes |
|---|:---:|:---:|:---:|---|
| [Agência Pública](https://apublica.org) | −0.45 | ±0.25 | Direta | Ortellado & Ribeiro (2018) |
| [Agência Brasil](https://agenciabrasil.ebc.com.br) | −0.10 | ±0.25 | Estrutural | Intervozes (2017) |

**Centro** (−0.10 ≤ score ≤ +0.25)

| Veículo | Score¹ | Incerteza | Base | Fontes |
|---|:---:|:---:|:---:|---|
| [O Globo](https://oglobo.globo.com) | 0.00 | ±0.15 | Direta | Feres Júnior et al. (2013); Intervozes (2017) |
| [G1](https://g1.globo.com) | 0.00 | ±0.30 | Estrutural | Intervozes (2017) |
| [CNN Brasil](https://www.cnnbrasil.com.br) | 0.00 | ±0.35 | ⚠️ Inferência | — |
| [Metrópoles](https://www.metropoles.com) | 0.00 | ±0.35 | ⚠️ Inferência | — |
| [UOL Notícias](https://noticias.uol.com.br) | +0.10 | ±0.35 | ⚠️ Inferência | — |
| [Folha de S.Paulo](https://www.folha.uol.com.br) | +0.20 | ±0.20 | Direta | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |

**Centro-direita** (+0.25 < score ≤ +0.55)

| Veículo | Score¹ | Incerteza | Base | Fontes |
|---|:---:|:---:|:---:|---|
| [O Estado de S. Paulo](https://www.estadao.com.br) | +0.40 | ±0.10 | Direta | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |
| [R7](https://noticias.r7.com)² | +0.40 | ±0.15 | Direta | Ortellado & Ribeiro (2018); Intervozes (2017) |
| [Veja](https://veja.abril.com.br) | +0.55 | ±0.20 | Direta | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |

**Direita** (score > +0.55)

| Veículo | Score¹ | Incerteza | Base | Fontes |
|---|:---:|:---:|:---:|---|
| [Gazeta do Povo](https://www.gazetadopovo.com.br) | +0.70 | ±0.20 | Direta | Ortellado & Ribeiro (2018) |
| [Jovem Pan News](https://jovempan.com.br) | +0.70 | ±0.20 | Direta | Ortellado & Ribeiro (2018) |

> ¹ Escala de −1.0 (progressista) a +1.0 (conservador). **Incerteza** (±) expressa a divergência entre as fontes. **Base**: *Direta* = classificação explícita na fonte; *Estrutural* = inferida por propriedade/estrutura; *Inferência* = sem cobertura nas fontes primárias.
>
> ² R7 coleta via scraping da homepage (`noticias.r7.com`) — feed RSS descontinuado pelo veículo.
>
> El País Brasil removido da lista de monitorados: edição brasileira encerrada em dezembro de 2021.

---

### Base Literária e Metodologia de Agregação

#### Fontes primárias utilizadas

| Fonte | Método | Peso | Cobertura |
|---|---|:---:|---|
| [**Manchetômetro** — Feres Júnior et al. (2013–)](http://www.manchetometro.com.br) | Análise de cobertura eleitoral; classificação por parcialidade editorial (IESP/UERJ) | 1.0 | Grandes jornais e revistas nacionais |
| [**Ortellado & Ribeiro (2018)**](https://gpopai.usp.br) — GPOPAI/USP | Análise de audiência cruzada com conteúdo editorial; survey de leitores | 1.0 | Jornais, revistas e portais digitais |
| [**Donos da Mídia** — Intervozes (2017)](https://donosdamidia.com.br) | Mapeamento de propriedade e concentração dos grupos de comunicação | 0.8 | TV, rádio, jornais e portais |

> O peso menor do Intervozes (0.8) deve-se ao seu foco em estrutura de propriedade — e não em análise direta de conteúdo editorial.

#### Fórmula de cálculo

```
Score = Σ(wᵢ × sᵢ) / Σwᵢ
Incerteza = desvio-padrão(sᵢ)
```

**Tabela de normalização:**

| Classificação original | Score normalizado |
|---|:---:|
| Esquerda | −0.75 |
| Centro-esquerda | −0.35 |
| Centro | 0.00 |
| Centro-direita | +0.40 |
| Direita | +0.70 |

#### Referências completas

- **FERES JÚNIOR, J. et al.** Manchetômetro. IESP/UERJ, 2013–. [manchetometro.com.br](http://www.manchetometro.com.br)
- **ORTELLADO, P.; RIBEIRO, M. M.** Estamos todos desinformados? GPOPAI/USP, 2018. [gpopai.usp.br](https://gpopai.usp.br)
- **INTERVOZES.** Donos da Mídia. São Paulo: Intervozes, 2017. [donosdamidia.com.br](https://donosdamidia.com.br)
- **GROSECLOSE, T.; MILYO, J.** A Measure of Media Bias. *QJE*, v. 120, n. 4, 2005. DOI: [10.1162/003355305775097542](https://doi.org/10.1162/003355305775097542)
- **DENZIN, N. K.** *The Research Act.* 2. ed. New York: McGraw-Hill, 1978.

---

## ⚠️ Limitações e Uso Responsável

- O BiasScore é uma **estimativa probabilística**, não um julgamento objetivo.
- O modelo aprende padrões dos anotadores do FactNews, que possuem perspectivas teóricas próprias.
- Risco de *shortcut learning*: menção de certos termos políticos pode inflar o score.
- Sátira, coluna de opinião e reportagem investigativa têm naturezas distintas — o modelo não distingue gênero jornalístico.
- Não utilize esta ferramenta para deslegitimar veículos de comunicação.

---

## 📜 Referências Principais

- SOUZA et al. BERTimbau (2020)
- VARGAS et al. FactNews (2023)
- ENTMAN, R. Framing Theory (1993)
- PANG, B.; LEE, L. Opinion Mining and Sentiment Analysis (2008)
- GEIRHOS et al. Shortcut Learning (2020)

---

## Licença

© Indra Seixas Neiva — USP 2026 · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
