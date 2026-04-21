import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Importamos tu función personalizada
from tokenizer import tokenize 

class FinancialDataset(Dataset):
    """
    Clase que envuelve los datos para PyTorch. 
    Se encarga de tokenizar cada oración bajo demanda.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Utilizamos tu función del archivo tokenizer.py
        encoding = tokenize(text, self.tokenizer, self.max_length)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(dataset_path: str, tokenizer, max_length: int = 128):
    """
    Carga el CSV, mapea etiquetas a números y crea los datasets de entrenamiento y validación.
    """
    # 1. Localizar el archivo CSV
    if os.path.isdir(dataset_path):
        full_path = os.path.join(dataset_path, "data.csv")
    else:
        full_path = dataset_path
        
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No se encontró el archivo data.csv en {full_path}")

    df = pd.read_csv(full_path)
    
    # 2. Limpieza y Mapeo de Etiquetas
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    df.columns = [c.lower() for c in df.columns]
    
    if 'sentiment' not in df.columns or 'sentence' not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'sentence' y 'sentiment'")

    df['label'] = df['sentiment'].str.lower().map(sentiment_map)
    df = df.dropna(subset=['label', 'sentence'])
    
    # 3. División de datos (80% entrenamiento, 20% validación)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['sentence'].tolist(), 
        df['label'].astype(int).tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 4. Crear objetos Dataset de PyTorch
    train_dataset = FinancialDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = FinancialDataset(val_texts, val_labels, tokenizer, max_length)
    
    return train_dataset, val_dataset