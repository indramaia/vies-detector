"""
scripts/generate_spectrum_chart.py
───────────────────────────────────
Gera o gráfico do espectro ideológico dos veículos monitorados
e salva em docs/spectrum.png para inclusão no README.

Uso:
    python scripts/generate_spectrum_chart.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / "ideological" / "data" / "ideological_references.json"
OUT_PATH = ROOT / "docs" / "spectrum.png"
OUT_PATH.parent.mkdir(exist_ok=True)

# ── Cores por posição ─────────────────────────────────────────────────────────
COLOR_MAP = {
    "Esquerda":       "#c0392b",
    "Centro-esquerda":"#e67e22",
    "Centro":         "#7f8c8d",
    "Centro-direita": "#2980b9",
    "Direita":        "#1a5276",
}

MARKER_MAP = {
    "direto":              "o",
    "estrutural":          "s",
    "inferencia_editorial":"^",
}

# ── Carrega dados ─────────────────────────────────────────────────────────────
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
vehicles = data["vehicles"]

entries = sorted(
    [
        {
            "name":      v["name"],
            "score":     v["ideology_score"],
            "uncertainty": v["uncertainty"],
            "label":     v["position_label"],
            "basis":     v.get("classification_basis", "direto"),
        }
        for v in vehicles.values()
    ],
    key=lambda x: x["score"],
)

names   = [e["name"]  for e in entries]
scores  = [e["score"] for e in entries]
errors  = [e["uncertainty"] for e in entries]
colors  = [COLOR_MAP.get(e["label"], "#95a5a6") for e in entries]
markers = [MARKER_MAP.get(e["basis"], "o") for e in entries]

# ── Figura ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

y = np.arange(len(names))

# Faixas de fundo
ax.axvspan(-1.0, -0.55, alpha=0.08, color="#c0392b")
ax.axvspan(-0.55, -0.10, alpha=0.08, color="#e67e22")
ax.axvspan(-0.10,  0.25, alpha=0.08, color="#7f8c8d")
ax.axvspan( 0.25,  0.55, alpha=0.08, color="#2980b9")
ax.axvspan( 0.55,  1.00, alpha=0.08, color="#1a5276")

# Linha central
ax.axvline(0, color="#bdc3c7", linewidth=1, linestyle="--")

# Barras de erro + pontos
for i, (score, err, color, marker) in enumerate(zip(scores, errors, colors, markers)):
    ax.errorbar(
        score, y[i],
        xerr=err,
        fmt="none",
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        alpha=0.6,
    )
    ax.scatter(score, y[i], color=color, marker=marker, s=90, zorder=5)

# Rótulos dos veículos
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)

# Rótulo do score ao lado de cada ponto
for i, (score, name) in enumerate(zip(scores, names)):
    ha = "left" if score < 0 else "right"
    offset = 0.03 if score < 0 else -0.03
    ax.text(score + offset, y[i], f"{score:+.2f}", va="center", ha=ha,
            fontsize=7.5, color="#555555")

# Eixo X
ax.set_xlim(-1.05, 1.05)
ax.set_xlabel("Score ideológico  [ −1 = Esquerda · 0 = Centro · +1 = Direita ]",
              fontsize=10, labelpad=10)

# Espaço extra no topo para os rótulos de faixa não colidirem com o título
ax.set_ylim(-0.8, len(names) + 1.5)

# Rótulos das faixas no topo
faixas = [
    (-0.775, "Esquerda"),
    (-0.325, "Centro-esquerda"),
    ( 0.075, "Centro"),
    ( 0.40,  "Centro-direita"),
    ( 0.775, "Direita"),
]
for xpos, label in faixas:
    ax.text(xpos, len(names) + 0.8, label, ha="center", va="bottom",
            fontsize=8, color="#555555", style="italic")

# Título
ax.set_title(
    "Espectro Ideológico dos Veículos Monitorados- Viés Detector · Indra Seixas Neiva · USP 2026",
    fontsize=10, fontweight="bold", pad=14,
)

# Legenda — cores (posição)
legend_pos = [
    mpatches.Patch(color=c, label=l)
    for l, c in COLOR_MAP.items()
]
# Legenda — marcadores (base de classificação)
from matplotlib.lines import Line2D
legend_basis = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=8, label="Classificação direta"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#555", markersize=8, label="Estrutural (propriedade)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#555", markersize=8, label="Inferência editorial"),
]

leg1 = ax.legend(
    handles=legend_pos,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.0),
    fontsize=8, title="Posição", title_fontsize=8,
    framealpha=0.9,
)
ax.add_artist(leg1)
ax.legend(
    handles=legend_basis,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.22),
    fontsize=8, title="Base de classificação", title_fontsize=8,
    framealpha=0.9,
)

ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em: {OUT_PATH}")
