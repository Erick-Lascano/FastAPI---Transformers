from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_trained_model(model_path: str):
    """Carga el modelo y fuerza el uso del tokenizador de Python (lento pero seguro)."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        local_files_only=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    model.eval() 
    return model, tokenizer