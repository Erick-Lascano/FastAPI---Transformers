from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_trained_model(model_path: str):
    """Carga el modelo y el tokenizador desde una ruta local."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() # Importante: pone el modelo en modo inferencia
    return model, tokenizer