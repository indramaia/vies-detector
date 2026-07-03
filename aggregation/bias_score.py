"""
aggregation/bias_score.py
─────────────────────────
Cálculo do BiasScore por artigo a partir das classificações sentence-level.

Fórmula (definida no TCC):
    BiasScore = sum(CLASS_WEIGHTS[label] × reported_speech_factor) / n_total

Intervalo: [0.0, 2.0]
    0.0 → todas as sentenças factuais
    2.0 → todas as sentenças fortemente enviesadas

Tabela de referência (3 faixas — espelha as classes do modelo):
    [0.0, 0.67) → Factual
    [0.67, 1.33) → Enviesada
    [1.33, 2.0] → Fortemente enviesada

Discurso reportado (terceira pessoa atribuída):
    Frases onde o viés pertence a uma fonte citada, não ao veículo,
    recebem fator de atenuação REPORTED_SPEECH_FACTOR (padrão 0.4).
    Exemplo: "A diretora disse 'Isso é uma grande catástrofe'"
    → classificada como enviesada, mas contribui 40% ao BiasScore.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ── Detecção de discurso reportado ────────────────────────────────────────────
# Fator aplicado sobre o peso da classe quando a sentença atribui o conteúdo
# a uma fonte externa (discurso direto ou indireto). O modelo ainda classifica
# a frase normalmente; apenas a contribuição ao BiasScore é atenuada.
REPORTED_SPEECH_FACTOR = 0.4

_ATTRIBUTION_VERBS_RE = re.compile(
    r"\b(disse|afirmou|declarou|acrescentou|respondeu|admitiu|reconheceu|"
    r"alegou|ressaltou|apontou|destacou|revelou|contou|explicou|comentou|"
    r"informou|anunciou|confirmou|negou|defendeu|criticou|argumentou|"
    r"sustentou|ponderou|observou|notou|salientou|frisou|reiterou|"
    r"assegurou|garantiu|prometeu|advertiu|alertou|lembrou|reforçou|"
    r"sinalizou|pontuou|enfatizou|sublinhou)\b",
    re.IGNORECASE,
)

_FRAMING_RE = re.compile(
    r"\b(segundo|de acordo com|conforme|na visão de|na avaliação de|"
    r"na opinião de)\b",
    re.IGNORECASE,
)

# Aspas balanceadas com conteúdo ≥ 3 chars — cobre ASCII, tipográficas e guillemets.
# Detecta discurso direto mesmo sem verbo de atribuição explícito:
#   ex. '"Isso é uma catástrofe"' ou '«Precisamos agir»'.
# Exige pelo menos 3 caracteres entre as aspas para evitar falsos positivos
# em abreviações ou ênfase tipográfica curta (ex. o "boom", o "não").
_BALANCED_QUOTES_RE = re.compile(
    r'(?:"[^"]{3,}"'       # "texto longo"  — ASCII duplas
    r'|“[^”]{3,}”'  # "texto"  — tipográficas (Word/Mac)
    r"|'[^']{3,}'"         # 'texto longo'  — ASCII simples
    r"|‘[^’]{3,}’"  # 'texto'  — tipográficas simples
    r"|«[^»]{3,}»)"        # «texto»        — guillemets (jornais BR)
)


def reported_speech_factor(sentence: str) -> float:
    """
    Retorna REPORTED_SPEECH_FACTOR se a sentença é discurso atribuído a uma
    fonte (terceira pessoa), 1.0 caso contrário.

    Três sinais detectados:
      1. Verbo de atribuição jornalística (disse, afirmou, declarou…)
      2. Locução enquadradora (segundo, de acordo com, conforme…)
      3. Aspas balanceadas com conteúdo ≥ 3 chars (discurso direto sem verbo)

    O label do classificador é mantido; só o peso no BiasScore é reduzido.
    """
    if (
        _ATTRIBUTION_VERBS_RE.search(sentence)
        or _FRAMING_RE.search(sentence)
        or _BALANCED_QUOTES_RE.search(sentence)
    ):
        return REPORTED_SPEECH_FACTOR
    return 1.0


@dataclass
class SentenceResult:
    """Resultado da classificação de uma única sentença."""
    sentence: str
    label: str
    label_id: int
    confidence: float
    scores: dict[str, float]
    # 1.0 = discurso direto do veículo; REPORTED_SPEECH_FACTOR = atribuído a fonte
    rs_factor: float = field(default=1.0)


# ── Pesos por classe ──────────────────────────────────────────────────────────
CLASS_WEIGHTS: dict[str, float] = {
    "factual": 0.0,
    "enviesada": 1.0,
    "fortemente_enviesada": 2.0,
}

# ── Faixas interpretativas (3 classes — espelha o classificador) ──────────────
BIAS_BANDS: list[tuple[float, float, str]] = [
    (0.0,  0.67, "Factual"),
    (0.67, 1.33, "Enviesada"),
    (1.33, 2.01, "Fortemente enviesada"),
]


@dataclass
class ArticleBiasResult:
    """Resultado do BiasScore para um artigo."""
    url_hash: str
    source_name: str
    ideology_id: str
    bias_score: float            # [0.0, 2.0]
    interpretation: str          # Faixa interpretativa
    sentence_count: int
    n_factual: int
    n_biased: int
    n_strongly_biased: int
    sentence_results: list[SentenceResult]


def _interpret(score: float) -> str:
    for low, high, label in BIAS_BANDS:
        if low <= score < high:
            return label
    return "Linguagem fortemente enviesada"


def compute_article_bias(
    url_hash: str,
    source_name: str,
    ideology_id: str,
    sentence_results: list[SentenceResult],
) -> ArticleBiasResult:
    """
    Calcula o BiasScore de um artigo.

    Args:
        url_hash         : hash SHA-256 do artigo (identificador único)
        source_name      : nome canônico do veículo
        ideology_id      : chave do mapeamento ideológico
        sentence_results : saída do SentenceClassifier para cada sentença

    Returns:
        ArticleBiasResult com score, interpretação e estatísticas por classe.
    """
    if not sentence_results:
        return ArticleBiasResult(
            url_hash=url_hash,
            source_name=source_name,
            ideology_id=ideology_id,
            bias_score=0.0,
            interpretation="Sem sentenças classificáveis",
            sentence_count=0,
            n_factual=0,
            n_biased=0,
            n_strongly_biased=0,
            sentence_results=[],
        )

    n_total = len(sentence_results)
    n_factual = sum(1 for r in sentence_results if r.label == "factual")
    n_biased = sum(1 for r in sentence_results if r.label == "enviesada")
    n_strongly = sum(1 for r in sentence_results if r.label == "fortemente_enviesada")

    # Fórmula do BiasScore ponderada por discurso reportado.
    # Sentenças atribuídas a fontes (rs_factor < 1.0) contribuem menos,
    # pois o viés é da fonte citada, não do veículo.
    weighted_sum = sum(
        CLASS_WEIGHTS[r.label] * r.rs_factor for r in sentence_results
    )
    bias_score = round(weighted_sum / n_total, 4)

    return ArticleBiasResult(
        url_hash=url_hash,
        source_name=source_name,
        ideology_id=ideology_id,
        bias_score=bias_score,
        interpretation=_interpret(bias_score),
        sentence_count=n_total,
        n_factual=n_factual,
        n_biased=n_biased,
        n_strongly_biased=n_strongly,
        sentence_results=sentence_results,
    )
