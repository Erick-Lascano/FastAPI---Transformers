# model_loader.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path):
    print(f"Cargando desde: {model_path}")
    
    # Cargar el modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Pasar a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer