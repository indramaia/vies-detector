"""
classifier/train.py
───────────────────
Fine-tuning do BERTimbau sobre o dataset FactNews para classificação
de viés editorial em três classes.

Uso:
    python classifier/train.py --data data/factnews.csv --output models/bertimbau-bias

Referências:
    DEVLIN et al. BERT (2019).
    SOUZA et al. BERTimbau (2020).
    VARGAS et al. FactNews (2023).
    LOSHCHILOV; HUTTER. AdamW (2019).
    HOWARD; RUDER. ULMFiT (2018).
    
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
from pathlib import Path
# import mlflow
import numpy as np
import pandas as pd
import torch
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
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
LEARNING_RATE = 2e-5              # original — convergência rápida, risco de overfitting
#LEARNING_RATE = 1e-5             # reduzido: aprende mais devagar, generaliza melhor
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4   # batch efetivo = 32
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01               # original — regularização L2 fraca
#WEIGHT_DECAY = 0.1               # aumentado 10x: penaliza pesos grandes, reduz memorização
EARLY_STOPPING_PATIENCE = 2       # original — parava cedo demais na época 2
#EARLY_STOPPING_PATIENCE = 3      # aumentado: dá mais chance ao modelo antes de parar


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

def train(data_path: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Carregando tokenizador: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    logger.info("Carregando e preparando FactNews…")
    dataset = load_factnews(data_path)
    tokenized = tokenize_dataset(dataset, tokenizer)
    tokenized = tokenized.rename_column("label", "labels")

    logger.info(f"Inicializando modelo: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # Dropout aumentado para reduzir memorização (padrão BERT: 0.1)
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2

    logger.info("=" * 60)
    logger.info("HIPERPARÂMETROS DE FINE-TUNING (melhorias anti-overfitting)")
    logger.info(f"  [1] Learning Rate        : {LEARNING_RATE}  (original: 2e-5)")
    logger.info(f"  [2] Hidden Dropout       : {model.config.hidden_dropout_prob}             (original: 0.1)")
    logger.info(f"  [4] Attention Dropout    : {model.config.attention_probs_dropout_prob}    (original: 0.1)")
    logger.info(f"  [5] Weight Decay (L2)    : {WEIGHT_DECAY}    (original: 0.01)")
    logger.info(f"  [6] Early Stop Patience  : {EARLY_STOPPING_PATIENCE} épocas (original: 2)")
    logger.info(f"  Batch efetivo            : {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Máx. épocas              : {NUM_EPOCHS}")
    logger.info(f"  Warmup ratio             : {WARMUP_RATIO}")
    logger.info("=" * 60)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        # label_smoothing_factor=0.0,  # original — sem suavização, modelo fica overconfident
        label_smoothing_factor=0.1,   # penaliza confiança excessiva, melhora generalização
        fp16=torch.cuda.is_available(),
        report_to="none",  # Desativa integração com MLflow (ajuste conforme necessário
    )

    trainer = Trainer(
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
    
    logger.info("Avaliando no conjunto de teste…")
    test_results = trainer.predict(tokenized["test"])
    preds = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids
    
    report = classification_report(
        labels, preds,
        target_names=["factual", "enviesada", "fortemente_enviesada"],
        zero_division=0,
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    logger.info(f"\n{report}")
    logger.info(f"Salvando modelo em: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Fine-tuning concluído. Macro-F1 no teste: {macro_f1:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BERTimbau → FactNews")
    parser.add_argument("--data", required=True, help="Caminho para factnews.csv")
    parser.add_argument("--output", default="models/bertimbau-bias", help="Diretório de saída do modelo")
    args = parser.parse_args()
    train(args.data, args.output)
