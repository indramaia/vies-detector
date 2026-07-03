"""
classifier/train.py
───────────────────
Fine-tuning do BERTimbau sobre o dataset FactNews para classificação
de viés editorial em três classes.

Uso:
    python classifier/train.py --data data/factnews.csv --output models/bertimbau-bias
    python classifier/train.py --data data/factnews.csv --output models/bertimbau-bias --seeds 42 123 456

Referências:
    DEVLIN et al. BERT (2019).
    SOUZA et al. BERTimbau (2020).
    VARGAS et al. FactNews (2023).
    LOSHCHILOV; HUTTER. AdamW (2019).
    HOWARD; RUDER. ULMFiT (2018).
    JOHNSON; KHOSHGOFTAAR. Survey on deep learning with class imbalance (2019).
    SOKOLOVA; LAPALME. A systematic analysis of performance measures (2009).
    LIN et al. Focal Loss for Dense Object Detection (2017).
    MÜLLER et al. When does label smoothing help? (2019).
    GURURANGAN et al. Don't Stop Pretraining (2020).

Fine-Tuning do BERTimbau — Do Zero ao Classificador de Viés Editorial comentado para leitores sem experiência prévia em NLP:
 O BERTimbau já "sabe português" — foi treinado em bilhões de palavras da Wikipedia e da web brasileira.
 Mas ele não sabe o que é viés editorial. O fine-tuning é como uma especialização:
 você pega esse modelo que já entende português e mostra para ele 4.952 sentenças de notícias que especialistas já classificaram.
 Ele aprende a associar padrões linguísticos (palavras carregadas, tom emocional, ausência de atribuição) com os rótulos:
 factual / enviesada / fortemente enviesada. Depois de treinado, ele consegue classificar sentenças novas que nunca viu.

"""

from __future__ import annotations
import argparse
import os
from itertools import product as itertools_product
from pathlib import Path
# import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from .model_loader import LABEL2ID, ID2LABEL, NUM_LABELS, MAX_LENGTH


# ── Hiperparâmetros (VARGAS et al., 2023; DEVLIN et al., 2019) ───────────────
BASE_MODEL                  = "neuralmind/bert-base-portuguese-cased"
# LEARNING_RATE             = 2e-5  # original — convergência rápida, overfitting
LEARNING_RATE               = 1e-5  # reduzido: aprende mais devagar, generalizou melhor
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4     # batch efetivo = 32
NUM_EPOCHS                  = 5
WARMUP_RATIO                = 0.1
# WEIGHT_DECAY              = 0.01  # original — regularização L2 fraca
WEIGHT_DECAY                = 0.1   # aumentado 10x: penaliza pesos grandes, reduz memorização
# EARLY_STOPPING_PATIENCE   = 2     # original — parava cedo demais na época 2
EARLY_STOPPING_PATIENCE     = 3     # aumentado: dá mais chance ao modelo antes de parar

# ── Pesos de classe — Camada 1: loss ponderada por frequência inversa ─────────
# Distribuição FactNews (VARGAS et al., 2023): factual=4242, enviesada=558, fortemente=1391
# Peso = N_total / n_c  →  cada exemplo da minoritária contribui ~11x mais para o gradiente.
# Justificativa: sem pesos, 4.242 "factuais" dominam o gradiente e o modelo aprende a
# "preferir" essa classe. Com pesos inversos, o modelo é forçado a prestar atenção na
# classe "enviesada" (9% do corpus). Ganho típico: F1 minoritária 0.64 → 0.68-0.72.
# Ref: JOHNSON; KHOSHGOFTAAR (2019)
_N_TOTAL = 6191
CLASS_WEIGHTS = torch.tensor([
    _N_TOTAL / 4242,  # factual              → peso ≈ 1.46
    _N_TOTAL / 558,   # enviesada            → peso ≈ 11.10  (minoritária)
    _N_TOTAL / 1391,  # fortemente_enviesada → peso ≈ 4.45
]).float()

# Variante suave (sqrt) — penaliza menos, preserva melhor as classes majoritárias
# CLASS_WEIGHTS = torch.sqrt(CLASS_WEIGHTS)
# ── Focal Loss — Camada 2 (alternativa à loss ponderada; comentada) ───────────
# Desenhada para classes desbalanceadas em detecção (LIN et al., 2017).
# Exemplos que o modelo já acerta com alta confiança contribuem pouco para a loss;
# exemplos difíceis (minoritária "enviesada") contribuem muito.
# Testar γ ∈ {1, 2, 3}. Combinada com class_weights, tende a ganhar +1-2pp sobre
# loss ponderada pura.
#
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=CLASS_WEIGHTS, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, logits, targets):
#         alpha = self.alpha.to(logits.device)
#         ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
#         pt = torch.exp(-ce)
#         return ((1 - pt) ** self.gamma * ce).mean()


# ── Trainer com loss ponderada — Camada 1 ────────────────────────────────────
class WeightedTrainer(Trainer):
    """
    Subclasse do HuggingFace Trainer que injeta CrossEntropyLoss ponderada
    por classe com label smoothing combinados.

    Por que subclasse e não label_smoothing_factor do TrainingArguments?
    TrainingArguments.label_smoothing_factor não aceita class_weights — precisam
    ser combinados manualmente na loss. (MÜLLER et al., 2019)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=CLASS_WEIGHTS.to(logits.device),
            label_smoothing=0.1,  # preserva o label smoothing original
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    # Alternativa com Focal Loss — substituir o compute_loss acima por:
    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     focal = FocalLoss(alpha=CLASS_WEIGHTS, gamma=2.0)
    #     loss = focal(logits, labels)
    #     return (loss, outputs) if return_outputs else loss


# ── Calibração de threshold — Camada 1 ───────────────────────────────────────
def tune_thresholds(
    logits_all: np.ndarray,
    labels_all: np.ndarray,
    n_classes: int = 3,
) -> tuple[np.ndarray, float]:
    """
    Varre thresholds no conjunto de validação para maximizar Macro-F1 sem retreinar.
    Ref: LIPTON et al. (2014); SOKOLOVA; LAPALME (2009).

    Estratégia: ajusta o softmax de cada classe por um threshold t_c antes do argmax.
    Grid 0.15–0.85 passo 0.05 → 15^3 = 3.375 combinações (< 1s de CPU).

    Retorna (best_thresholds, best_macro_f1).
    """
    probs = torch.softmax(torch.tensor(logits_all), dim=-1).numpy()
    best_f1 = 0.0
    best_thresholds = np.array([1 / n_classes] * n_classes)

    grid = np.arange(0.15, 0.90, 0.05)
    for thresholds in itertools_product(grid, repeat=n_classes):
        t = np.array(thresholds)
        preds = np.argmax(probs / t, axis=-1)
        f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = t

    return best_thresholds, best_f1


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Calcula Macro-F1 e acurácia.
    Macro-F1 é adotado como métrica principal por tratar todas as classes
    com igual peso, independentemente do desbalanceamento do FactNews.
    (SOKOLOVA; LAPALME, 2009)
    """

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": float(accuracy)}


# ── Carregamento e divisão do corpus ─────────────────────────────────────────

def load_factnews(csv_path: str) -> DatasetDict:
    """
    Carrega o FactNews a partir do CSV e divide em treino/validação/teste.

    Esquema esperado do CSV:
        sentence : str  — sentença anotada
        label    : int  — -1 (fort. enviesada), 0 (factual), 1 (enviesada)

    Remapeamento para índices não-negativos:
        -1 → 2  (fortemente_enviesada)
         0 → 0  (factual)
         1 → 1  (enviesada)

    Divisão: 80/10/10 estratificada por rótulo.
    """
    df = pd.read_csv(csv_path)

    # Adaptar colunas do FactNews (sentences/classe → sentence/label)
    df = df.rename(columns={'sentences': 'sentence', 'classe': 'label'})

    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("O CSV deve conter colunas 'sentence' e 'label'.")

    # Remapeamento
    label_map = {-1: 2, 0: 0, 1: 1}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["sentence", "label"])
    df["label"] = df["label"].astype(int)

    logger.info(f"FactNews carregado: {len(df)} sentenças.")
    logger.info(f"Distribuição:\n{df['label'].value_counts().to_string()}")

    # Divisão 80/10/10 estratificada
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    return DatasetDict({
        "train": Dataset.from_pandas(train_df[["sentence", "label"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df[["sentence", "label"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df[["sentence", "label"]].reset_index(drop=True)),
    })


# ── Tokenização ───────────────────────────────────────────────────────────────

def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    return dataset.map(tokenize_fn, batched=True, remove_columns=["sentence"])


# ── Treinamento principal ─────────────────────────────────────────────────────

def train(data_path: str, output_dir: str, seeds: list[int] | None = None) -> None:
    """
    seeds — Camada 1: média de múltiplas seeds (HENDERSON et al., 2018).
        Ex: [42, 123, 456] roda 3 treinos independentes e faz média dos logits.
        Ganho típico: +0.5-1.5pp no macro + desvio-padrão reportável na defesa
        (ex.: "macro-F1 = 0.87 ± 0.01", mais robusto que um número cru).
        Se None, usa apenas seed=42 (comportamento original).

    Trabalhos futuros — Camada 3 (não implementados por limitação de tempo):
        - Domain-adaptive continued pretraining em corpus jornalístico BR
          (GURURANGAN et al., 2020): continuar MLM em ~50MB de notícias coletadas
          via RSS antes do fine-tuning. Ganho esperado: +2-5pp.
        - Troca de backbone: Albertina-PTBR (PORTULAN, 2023) ou BERTimbau-large
          — +1-3pp em benchmarks PT-BR, custo ~2x tempo de treino.
        - Aumento de dados LLM: gerar ~300-500 exemplos da classe "enviesada"
          via Claude/GPT-4 com prompt controlado + revisão humana. Dobra o corpus
          minoritário sem custo de re-anotação.
        - Pseudo-labeling semi-supervisionado: rotular matérias RSS com confiança
          > 0.85, adicionar ao treino. Expansão do corpus sem anotação humana.
    """
    if seeds is None:
        seeds = [42]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Carregando tokenizador: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    logger.info("Carregando e preparando FactNews…")
    dataset = load_factnews(data_path)
    tokenized = tokenize_dataset(dataset, tokenizer)
    tokenized = tokenized.rename_column("label", "labels")

    logger.info("=" * 60)
    logger.info("HIPERPARÂMETROS DE FINE-TUNING (melhorias anti-overfitting)")
    logger.info(f"  [1] Learning Rate        : {LEARNING_RATE}")  
    logger.info(f"  [2] Hidden Dropout       : 0.2")             
    logger.info(f"  [4] Attention Dropout    : 0.2")
    logger.info(f"  [5] Weight Decay (L2)    : {WEIGHT_DECAY}")
    logger.info(f"  [6] Early Stop Patience  : {EARLY_STOPPING_PATIENCE} épocas")
    logger.info(f"  [7] Loss ponderada       : pesos {CLASS_WEIGHTS.tolist()}")
    logger.info(f"  [8] Threshold tuning     : ativado pós-treino na validação")
    logger.info(f"  [9] Multi-seed averaging : seeds={seeds}")
    logger.info(f"  Batch efetivo            : {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Máx. épocas              : {NUM_EPOCHS}")
    logger.info(f"  Warmup ratio             : {WARMUP_RATIO}")
    logger.info("=" * 60)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    all_test_logits: list[np.ndarray] = []
    all_val_logits: list[np.ndarray] = []
    val_labels: np.ndarray | None = None
    test_labels: np.ndarray | None = None

    for seed in seeds:
        logger.info(f"── Seed {seed} ({seeds.index(seed) + 1}/{len(seeds)}) ──")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        # Dropout aumentado para reduzir memorização (padrão BERT: 0.1)
        model.config.hidden_dropout_prob = 0.2
        model.config.attention_probs_dropout_prob = 0.2

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=50,
            # label_smoothing_factor=0.1,  # movido para WeightedTrainer.compute_loss
            #                              # (não pode coexistir com class_weights no Trainer padrão)
            label_smoothing_factor=0.0,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=seed,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )

        logger.info("Iniciando fine-tuning…")
        trainer.train()

        # Coleta logits do teste e da validação para averaging e threshold tuning
        test_out = trainer.predict(tokenized["test"])
        val_out = trainer.predict(tokenized["validation"])
        all_test_logits.append(test_out.predictions)
        all_val_logits.append(val_out.predictions)

        if test_labels is None:
            test_labels = test_out.label_ids
        if val_labels is None:
            val_labels = val_out.label_ids

        # Salva o modelo da última seed
        if seed == seeds[-1]:
            logger.info(f"Salvando modelo (seed={seed}) em: {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

    # ── Averaging — Camada 1 ──────────────────────────────────────────────────
    mean_test_logits = np.mean(all_test_logits, axis=0)
    mean_val_logits  = np.mean(all_val_logits,  axis=0)

    # F1 por seed (para desvio-padrão reportável na defesa)
    per_seed_f1 = [
        f1_score(test_labels, np.argmax(logits, axis=-1), average="macro", zero_division=0)
        for logits in all_test_logits
    ]
    std_f1 = float(np.std(per_seed_f1)) if len(per_seed_f1) > 1 else 0.0

    # ── Threshold tuning — Camada 1 ───────────────────────────────────────────
    logger.info("Calibrando thresholds na validação…")
    best_thresholds, val_f1_tuned = tune_thresholds(mean_val_logits, val_labels)
    logger.info(f"Thresholds ótimos (val): {best_thresholds.round(2)} → Macro-F1 val={val_f1_tuned:.4f}")

    # Predições no teste: argmax padrão
    preds_default = np.argmax(mean_test_logits, axis=-1)
    macro_f1_default = f1_score(test_labels, preds_default, average="macro", zero_division=0)

    # Predições no teste: thresholds calibrados
    probs_test   = torch.softmax(torch.tensor(mean_test_logits), dim=-1).numpy()
    preds_tuned  = np.argmax(probs_test / best_thresholds, axis=-1)
    macro_f1_tuned = f1_score(test_labels, preds_tuned, average="macro", zero_division=0)

    # ── Relatório final ───────────────────────────────────────────────────────
    report_default = classification_report(
        test_labels, preds_default,
        target_names=["factual", "enviesada", "fortemente_enviesada"],
        zero_division=0,
    )
    report_tuned = classification_report(
        test_labels, preds_tuned,
        target_names=["factual", "enviesada", "fortemente_enviesada"],
        zero_division=0,
    )

    logger.info(f"\n── Resultado argmax padrão ──\n{report_default}")
    logger.info(f"\n── Resultado com threshold calibrado ──\n{report_tuned}")
    logger.info(
        f"✅ Fine-tuning concluído.\n"
        f"   Macro-F1 (média {len(seeds)} seed(s), argmax)     : {macro_f1_default:.4f} ± {std_f1:.4f}\n"
        f"   Macro-F1 (threshold calibrado no teste) : {macro_f1_tuned:.4f}\n"
        f"   F1 por seed: {[round(f, 4) for f in per_seed_f1]}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BERTimbau → FactNews")
    parser.add_argument("--data",   required=True, help="Caminho para factnews.csv")
    parser.add_argument("--output", default="models/bertimbau-bias", help="Diretório de saída do modelo")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Seeds para média de múltiplas rodadas (ex: --seeds 42 123 456). Camada 1.",
    )
    args = parser.parse_args()
    train(args.data, args.output, seeds=args.seeds)
