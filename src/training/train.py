import os
import sys

# 1. Forzar a Python a encontrar la raíz del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# 2. Importación absoluta para evitar conflictos con módulos del sistema
from src import config 
from src.training.data_extractor import download_dataset
from src.training.data_loader import prepare_data
from src.training.tokenizer import load_tokenizer

def train_model():
    # Ruta donde se guardará en Google Drive
    DRIVE_SAVE_PATH = "./fin_sentiment_model"
    
    print("1. Descargando datos de Kaggle...")
    csv_path = download_dataset()

    print("2. Cargando tokenizador...")
    # Usar archivo tokenizer.py
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
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=config.LEARNING_RATE,
        load_best_model_at_end=False,
        save_total_limit=1,
        fp16=True  # GPU
    )

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