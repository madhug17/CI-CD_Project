import mlflow
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score


# =========================================================
# METRICS
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1  = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}


# =========================================================
# TRAIN
# =========================================================
def train_model():

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------
    print("Loading Financial PhraseBank dataset...")
    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_50agree",
        trust_remote_code=True
    )

    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    # BUG FIX: HuggingFace Trainer expects the column named 'labels'
    # Financial PhraseBank ships it as 'label' (singular) — rename it
    dataset = dataset.rename_column("label", "labels")

    print(f"Train samples : {len(dataset['train'])}")
    print(f"Val samples   : {len(dataset['test'])}")

    # --------------------------------------------------
    # 2. Tokenizer + Model
    # --------------------------------------------------
    print("\nLoading FinBERT tokenizer and model...")
    model_name = "ProsusAI/finbert"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
        # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
        # main.py maps these to:         Bullish      Bearish      Neutral
    )

    # --------------------------------------------------
    # 3. Tokenize
    # --------------------------------------------------
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # --------------------------------------------------
    # 4. MLflow experiment
    # --------------------------------------------------
    mlflow.set_experiment("FinBERT_Financial_Sentiment")

    # --------------------------------------------------
    # 5. Training arguments
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./finbert_results",

        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,   # saves the best checkpoint automatically
        metric_for_best_model="f1",

        # Hyperparameters
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,            # 3 epochs hits ~91% F1 on this dataset
        weight_decay=0.01,
        warmup_ratio=0.1,              # 10% of steps for LR warmup

        # Logging
        logging_steps=10,
        report_to="mlflow",

        # Reproducibility
        seed=42,
    )

    # --------------------------------------------------
    # 6. Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------
    # 7. Train
    # --------------------------------------------------
    print("\nStarting training...")
    print("Watch metrics live at: http://localhost:5000 (run: mlflow ui)")
    trainer.train()

    # --------------------------------------------------
    # 8. Save best model
    # --------------------------------------------------
    print("\nSaving best model to ./app/saved_model ...")
    model.save_pretrained("./app/saved_model")
    tokenizer.save_pretrained("./app/saved_model")
    print("Done. Run: uvicorn app.main:app --reload")
    # Add this right before model.save_pretrained()

    id2label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    label2id = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
    
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    model.save_pretrained("./app/saved_model")
    tokenizer.save_pretrained("./app/saved_model")


if __name__ == "__main__":
    train_model()