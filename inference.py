import torch
import torch.nn.functional as F
import math

def predict(text, model, tokenizer):
    """
    Realiza la predicción de sentimiento y devuelve un diccionario con la etiqueta 
    mapeada y la confianza protegida contra valores NaN.
    """
    # 1. Configurar el dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Tokenización
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(device)

    # 3. Inferencia (sin calcular gradientes para ahorrar memoria)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. Procesamiento de resultados
    logits = outputs.logits
    # Aplicamos Softmax para obtener probabilidades entre 0 y 1
    probs = F.softmax(logits, dim=-1)
    
    # Obtenemos la confianza más alta y el ID de la clase
    confidence_tensor, predicted_class_id = torch.max(probs, dim=-1)
    
    # Extraemos los valores de los tensores
    conf_val = confidence_tensor.item()
    class_id = int(predicted_class_id.item())

    # 5. Mapeo de Etiquetas (Ajustado al Financial PhraseBank)
    # 0: Negative, 1: Neutral, 2: Positive
    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    label_text = id2label.get(class_id, f"LABEL_{class_id}")

    # 6. Control de Seguridad para NaN (Evita el error en JSON)
    if math.isnan(conf_val) or math.isinf(conf_val):
        conf_val = 0.0

    return {
        "label": label_text,
        "confidence": round(float(conf_val), 4)
    }