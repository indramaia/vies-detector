"""
scripts/benchmark_llm_baseline.py
──────────────────────────────────
Benchmark zero-shot de um LLM (Gemini) contra o BERTimbau fine-tuned
(classifier/train.py), para responder "comparado a quê?" na defesa.

Roda o MESMO split de teste do FactNews (620 sentenças, estratificado,
seed=42) usado para avaliar o BERTimbau — reaproveita classifier.train.
load_factnews para garantir que é exatamente o mesmo conjunto, sem
vazamento de dados nem viés de seleção entre os dois experimentos.

O LLM nunca vê um exemplo rotulado (zero-shot): só a definição das três
classes no prompt. Isso testa diretamente a alegação de que um encoder
fine-tuned com poucos milhares de exemplos anotados tende a empatar ou
superar um LLM generativo usado sem fine-tuning em classificação.

Uso:
    GEMINI_API_KEY=<chave> python scripts/benchmark_llm_baseline.py \
        --data "data/FactNews/dataset/factnews_dataset.csv"

    # Teste rápido/barato (menos chamadas de API):
    GEMINI_API_KEY=<chave> python scripts/benchmark_llm_baseline.py \
        --data "data/FactNews/dataset/factnews_dataset.csv" --limit 50

Sobre o modelo (--model, padrão gemini-3.5-flash-lite):
    Os modelos "flash" de topo de linha (ex. gemini-3.8-flash) têm cota free
    tier MUITO restrita (visto na prática: 20 requisições por DIA) — inviável
    para um benchmark de centenas de sentenças. Os modelos "flash-lite" são
    desenhados para alto volume/baixo custo e são suficientes para esta
    tarefa (classificação simples de 3 classes). Mesmo assim, confira sua
    cota atual em https://aistudio.google.com/rate-limit antes de rodar o
    conjunto de teste completo (620 sentenças) e ajuste --sleep conforme
    necessário.

Requer:
    pip install google-genai

Referências:
    VARGAS et al. FactNews (2023).
    BROWN et al. Language Models are Few-Shot Learners (2020) — zero-shot como baseline.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from sklearn.metrics import classification_report, f1_score

from classifier.model_loader import LABEL2ID
from classifier.train import load_factnews

# ── Prompt zero-shot ─────────────────────────────────────────────────────────
# Mesma taxonomia de 3 classes do FactNews/BERTimbau (VARGAS et al., 2023) —
# necessário para que a comparação seja sobre o mesmo problema, não sobre
# definições de classe diferentes.
PROMPT_TEMPLATE = """Você é um classificador de viés editorial em notícias jornalísticas brasileiras.
Classifique a sentença abaixo em EXATAMENTE uma das três categorias:

- factual: relata fatos de forma neutra, sem opinião nem linguagem carregada.
- enviesada: contém leve inclinação opinativa, adjetivos avaliativos ou enquadramento tendencioso, mas sem ataque direto.
- fortemente_enviesada: opinião explícita, linguagem emocionalmente carregada, acusações ou ataques diretos, sem atribuição a fonte externa.

Responda APENAS com uma destas três palavras, sem explicação, sem pontuação: factual, enviesada ou fortemente_enviesada.

Sentença: "{sentence}"
Classe:"""


def _parse_label(raw: str) -> int | None:
    """Extrai o label_id da resposta do LLM — tolera variações de formatação."""
    text = raw.strip().lower().replace(".", "").replace('"', "").replace("*", "")
    # ordem importa: "fortemente_enviesada" contém "enviesada" como substring
    for name in ("fortemente_enviesada", "fortemente enviesada", "enviesada", "factual"):
        if name in text:
            canonical = name.replace(" ", "_")
            return LABEL2ID[canonical]
    return None


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s|'retryDelay':\s*'(\d+)s'")


def _backoff_seconds(exc: Exception, attempt: int) -> float:
    """
    Decide quanto esperar antes da próxima tentativa.

    Erro 429 (RESOURCE_EXHAUSTED) traz o tempo de espera recomendado pelo
    próprio Google no corpo da resposta (ex. "Please retry in 21.8s" e/ou
    'retryDelay': '21s') — respeitar esse valor evita martelar a cota de
    novo e piorar o throttling. Outros erros (ex. 503 "high demand") usam
    backoff exponencial (5s, 10s, 20s, 40s…) por não trazerem essa dica.
    """
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        value = match.group(1) or match.group(2)
        return float(value) + 1.0  # margem de segurança sobre o valor sugerido
    return min(5 * 2 ** (attempt - 1), 40)


def classify_with_gemini(client, model: str, sentence: str, max_retries: int = 5) -> int | None:
    """Classifica uma sentença via Gemini, com retry adaptativo em caso de erro de API."""
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.0},  # determinismo — comparável entre rodadas
            )
            return _parse_label(response.text or "")
        except Exception as exc:
            logger.warning(f"Tentativa {attempt}/{max_retries} falhou: {exc}")
            if attempt < max_retries:
                time.sleep(_backoff_seconds(exc, attempt))
    return None


def run_benchmark(data_path: str, model: str, limit: int | None, sleep_s: float) -> None:
    try:
        from google import genai
    except ImportError:
        logger.error("Pacote 'google-genai' não instalado. Rode: pip install google-genai")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Variável GEMINI_API_KEY não definida (.env ou ambiente).")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    logger.info("Carregando o MESMO split de teste usado no BERTimbau (classifier.train.load_factnews)…")
    dataset = load_factnews(data_path)
    test = dataset["test"]
    if limit:
        test = test.select(range(min(limit, len(test))))
    logger.info(f"Sentenças de teste: {len(test)} | Modelo: {model} (zero-shot, temperature=0)")

    y_true: list[int] = []
    y_pred: list[int] = []
    unparsed = 0

    for i, row in enumerate(test):
        pred = classify_with_gemini(client, model, row["sentence"])
        if pred is None:
            unparsed += 1
            pred = LABEL2ID["factual"]  # fallback conservador — mantém a amostra na contagem
        y_true.append(row["label"])
        y_pred.append(pred)

        if (i + 1) % 25 == 0 or (i + 1) == len(test):
            logger.info(f"  {i + 1}/{len(test)} classificadas…")
        if sleep_s:
            time.sleep(sleep_s)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=["factual", "enviesada", "fortemente_enviesada"],
        zero_division=0,
    )

    logger.info(f"\n── {model} (zero-shot) — resultado no teste FactNews ──\n{report}")
    logger.info(f"Macro-F1 ({model}, zero-shot): {macro_f1:.4f}")
    if unparsed:
        logger.warning(
            f"{unparsed}/{len(test)} respostas não reconhecidas como uma das 3 classes "
            f"(contadas como 'factual' por fallback — ver prompt/parsing se a taxa for alta)."
        )
    logger.info(
        "Referência — BERTimbau fine-tuned, CE ponderada + threshold calibrado "
        "(classifier/train.py --loss ce): Macro-F1 = 0.82 "
        "(ver README.md, seção 'Desempenho no Teste')."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot de LLM (Gemini) vs. BERTimbau fine-tuned no teste do FactNews"
    )
    parser.add_argument("--data", required=True, help="Caminho para factnews_dataset.csv")
    parser.add_argument(
        "--model", default="gemini-3.5-flash-lite",
        help="Modelo Gemini (padrão: gemini-3.5-flash-lite — modelo 'flash' de alto volume/baixo "
             "custo; gemini-3.8-flash é o topo de linha e tem cota free tier de só 20 req/DIA, "
             "inviável para este benchmark)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limita nº de sentenças — teste rápido/barato")
    parser.add_argument(
        "--sleep", type=float, default=4.0,
        help="Segundos de espera entre chamadas — ajuste conforme sua cota em "
             "https://aistudio.google.com/rate-limit (RPM varia por modelo/conta)",
    )
    args = parser.parse_args()

    run_benchmark(args.data, args.model, args.limit, args.sleep)
