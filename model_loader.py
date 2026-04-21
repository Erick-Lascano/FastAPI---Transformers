from transformers import AutoModelForSequenceClassification

def load_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model