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
├── .env
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
cp .env .env
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

## 🗺️ Veículos Monitorados e Posicionamento Ideológico

O sistema monitora **18 veículos** de mídia brasileira, cobrindo o espectro ideológico de esquerda a direita. O posicionamento de cada veículo é derivado de fontes acadêmicas independentes — não constitui julgamento do sistema.

**Esquerda** (score < −0.55)

| Veículo | Score¹ | Incerteza | Fontes consultadas |
|---|:---:|:---:|---|
| [Brasil de Fato](https://www.brasildefato.com.br) | −0.80 | ±0.10 | Ortellado & Ribeiro (2018); Intervozes (2017) |
| [Outras Palavras](https://outraspalavras.net) | −0.75 | ±0.15 | Ortellado & Ribeiro (2018); Intervozes (2017) |
| [Carta Capital](https://www.cartacapital.com.br) | −0.70 | ±0.15 | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |
| [Le Monde Diplomatique Brasil](https://diplomatique.org.br) | −0.65 | ±0.15 | Ortellado & Ribeiro (2018); Feres Júnior et al. (2013) |
| [The Intercept Brasil](https://theintercept.com/brasil) | −0.60 | ±0.15 | Reuters Institute (2024); Ortellado & Ribeiro (2018) |

**Centro-esquerda** (−0.55 ≤ score < −0.10)

| Veículo | Score¹ | Incerteza | Fontes consultadas |
|---|:---:|:---:|---|
| [Agência Pública](https://apublica.org) | −0.45 | ±0.20 | Reuters Institute (2024); Ortellado & Ribeiro (2018) |
| [El País Brasil](https://brasil.elpais.com) | −0.25 | ±0.20 | Reuters Institute (2024) |
| [Agência Brasil](https://agenciabrasil.ebc.com.br) | −0.15 | ±0.20 | Intervozes (2017) |

**Centro** (−0.10 ≤ score ≤ +0.25)

| Veículo | Score¹ | Incerteza | Fontes consultadas |
|---|:---:|:---:|---|
| [UOL Notícias](https://noticias.uol.com.br) | +0.05 | ±0.25 | Reuters Institute (2024) |
| [Folha de S.Paulo](https://www.folha.uol.com.br) | +0.10 | ±0.20 | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |
| [Metrópoles](https://www.metropoles.com) | +0.10 | ±0.30 | Reuters Institute (2024) |
| [G1](https://g1.globo.com) | +0.15 | ±0.20 | Intervozes (2017); Reuters Institute (2024) |
| [CNN Brasil](https://www.cnnbrasil.com.br) | +0.20 | ±0.20 | Reuters Institute (2024) |
| [O Globo](https://oglobo.globo.com) | +0.20 | ±0.20 | Feres Júnior et al. (2013); Intervozes (2017) |

**Centro-direita** (+0.25 < score ≤ +0.55)

| Veículo | Score¹ | Incerteza | Fontes consultadas |
|---|:---:|:---:|---|
| [O Estado de S. Paulo](https://www.estadao.com.br) | +0.35 | ±0.15 | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |
| [R7](https://noticias.r7.com) | +0.40 | ±0.20 | Intervozes (2017); Ortellado & Ribeiro (2018) |
| [Veja](https://veja.abril.com.br) | +0.45 | ±0.20 | Feres Júnior et al. (2013); Ortellado & Ribeiro (2018) |

**Direita** (score > +0.55)

| Veículo | Score¹ | Incerteza | Fontes consultadas |
|---|:---:|:---:|---|
| [Gazeta do Povo](https://www.gazetadopovo.com.br) | +0.65 | ±0.15 | Ortellado & Ribeiro (2018) |

> ¹ Escala de −1.0 (progressista) a +1.0 (conservador). A coluna **Incerteza** (±) expressa a divergência entre as fontes consultadas — quanto maior, menos consenso acadêmico existe sobre o posicionamento do veículo.

---

### Base Literária e Metodologia de Agregação

#### Fontes primárias

| Fonte | Método | Cobertura |
|---|---|---|
| [**Manchetômetro** — Feres Júnior et al. (2013–)](http://www.manchetometro.com.br) | Análise de cobertura eleitoral; classificação por parcialidade editorial | Grandes jornais e revistas nacionais |
| [**Ortellado & Ribeiro (2018)**](https://gpopai.usp.br) — GPOPAI/USP | Análise de audiência cruzada com conteúdo; survey de leitores | Jornais, revistas e portais digitais |
| [**Donos da Mídia** — Intervozes (2017)](https://donosdamidia.com.br) | Mapeamento de propriedade e concentração dos grupos de comunicação | TV, rádio, jornais e portais |
| [**Digital News Report** — Reuters Institute (2024)](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2024) | Survey de consumo e confiança em notícias (n ≈ 2.000 brasileiros) | Portais digitais, redes sociais e TV |

#### Como os scores são calculados

Cada fonte utiliza método, escala e período distintos. Para torná-las comparáveis, este trabalho adota uma **agregação meta-analítica** em três etapas — inspirada em Groseclose & Milyo (2005) e na abordagem de síntese de evidências da Cochrane Collaboration:

**1. Normalização para escala comum** — as classificações originais (ex.: categorias nominais como "esquerda / centro / direita" ou scores ordinais) são convertidas para a escala contínua [−1, +1]:

| Categoria original | Score equivalente |
|---|:---:|
| Esquerda | −0.75 |
| Centro-esquerda | −0.35 |
| Centro | 0.00 |
| Centro-direita | +0.40 |
| Direita | +0.70 |

**2. Média ponderada por qualidade metodológica** — fontes com maior rigor amostral e peer-review recebem peso maior. O Manchetômetro e Ortellado & Ribeiro (revisados por pares) têm peso 1.0; Reuters Institute (survey) tem peso 0.8; Intervozes (análise de propriedade, não de conteúdo) tem peso 0.6.

**3. Incerteza como desvio entre fontes** — o campo `uncertainty` no JSON reflete o desvio-padrão dos scores normalizados entre as fontes disponíveis. Veículos com apenas uma fonte têm incerteza arbitrariamente elevada (±0.25–0.30) por ausência de triangulação.

> **Limitação reconhecida:** nenhuma fonte cobre todos os 18 veículos com o mesmo método. Para veículos de menor porte (Outras Palavras, Agência Pública, Metrópoles), a classificação baseia-se em uma única fonte ou inferência por proximidade editorial — o que eleva a incerteza e deve ser interpretado com cautela.

#### Justificativa metodológica (para fins acadêmicos)

> *"O score ideológico de cada veículo foi obtido por agregação meta-analítica de classificações independentes, convertidas para escala comum [−1, +1] e ponderadas pelo rigor metodológico de cada fonte. A incerteza associada a cada score expressa a divergência entre as fontes — veículos com maior consenso acadêmico têm incerteza menor. Essa abordagem segue o princípio de triangulação metodológica (DENZIN, 1978) e é análoga à síntese de evidências proposta por Groseclose & Milyo (2005) para mensuração de viés em mídia."*

#### Referências completas das fontes classificatórias

- **FERES JÚNIOR, J. et al.** Manchetômetro: monitoramento do noticiário político na mídia brasileira. IESP/UERJ, 2013–. Disponível em: [http://www.manchetometro.com.br](http://www.manchetometro.com.br)

- **ORTELLADO, P.; RIBEIRO, M. M.** Estamos todos desinformados? GPOPAI — Grupo de Pesquisa em Políticas Públicas para o Acesso à Informação, USP, 2018. Disponível em: [https://gpopai.usp.br](https://gpopai.usp.br)

- **INTERVOZES — Coletivo Brasil de Comunicação Social.** Donos da Mídia: mapeamento da propriedade dos meios de comunicação no Brasil. São Paulo: Intervozes, 2017. Disponível em: [https://donosdamidia.com.br](https://donosdamidia.com.br)

- **NEWMAN, N. et al.** Reuters Institute Digital News Report 2024. Reuters Institute for the Study of Journalism, University of Oxford, 2024. Disponível em: [https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2024](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2024)

- **GROSECLOSE, T.; MILYO, J.** A Measure of Media Bias. *The Quarterly Journal of Economics*, v. 120, n. 4, p. 1191–1237, 2005. Disponível em: [https://doi.org/10.1162/003355305775097542](https://doi.org/10.1162/003355305775097542)

- **DENZIN, N. K.** *The Research Act: A Theoretical Introduction to Sociological Methods.* 2. ed. New York: McGraw-Hill, 1978.

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