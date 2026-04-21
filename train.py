import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import config
from data_extractor import download_dataset
from data_loader import prepare_data
from tokenizer import load_tokenizer

def train_model():
    # Ruta donde se guardará en tu Google Drive
    DRIVE_SAVE_PATH = "/content/drive/MyDrive/fin_sentiment_model"
    
    print("1. Descargando datos de Kaggle...")
    csv_path = download_dataset()

    print("2. Cargando tokenizador...")
    # Usamos tu nuevo archivo tokenizer.py
    tokenizer = load_tokenizer(config.MODEL_NAME)

    print("3. Preparando datos...")
    train_dataset, eval_dataset = prepare_data(csv_path, tokenizer, config.MAX_LENGTH)

    print("4. Cargando modelo base para entrenar...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )

    print("5. Configurando hiperparámetros para GPU...")
    args = TrainingArguments(
        output_dir=DRIVE_SAVE_PATH,     
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        load_best_model_at_end=True,
        fp16=True  # Magia para GPU

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print("6. ¡Iniciando el entrenamiento en GPU!")
    trainer.train()

    print("7. Guardando el modelo final y el tokenizador en Google Drive...")
    model.save_pretrained(DRIVE_SAVE_PATH)
    tokenizer.save_pretrained(DRIVE_SAVE_PATH)
    
    print(f"¡Éxito! Tu modelo ya vive en tu Google Drive: {DRIVE_SAVE_PATH}")

if __name__ == "__main__":
    train_model()