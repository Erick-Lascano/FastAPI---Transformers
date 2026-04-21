import torch

def get_prediction(text: str, model, tokenizer):
    """Procesa el texto y devuelve la etiqueta de sentimiento."""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding='max_length'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs, dim=-1).item()
    
    labels = {0: "negative", 1: "neutral", 2: "positive"}
    return labels[label_id], torch.max(probs).item()