import os

# Creamos una carpeta de cache en tu disco D:
os.environ['HF_HOME'] = r'D:\Az\FastAPI---Transformers\hf_cache'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config
import model_loader
import inference


app = FastAPI(
    title="MUTLF: Benchmark Service",
    description="Comparativa entre FinBERT (Experto) y Custom BERT (Kaggle)"
)

print("Cargando modelos en memoria...")

expert_model, expert_tok = model_loader.load_model_and_tokenizer(config.FINBERT_NAME)

custom_model = None
custom_tok = None
if os.path.exists(config.CUSTOM_MODEL_PATH):
    custom_model, custom_tok = model_loader.load_model_and_tokenizer(config.CUSTOM_MODEL_PATH)
    print("Modelo Custom detectado y cargado.")
else:
    print("Modelo Custom no encontrado. Solo FinBERT estará disponible.")

class SentimentRequest(BaseModel):
    sentence: str

@app.get("/")
def health_check():
    return {"status": "online", "custom_model_loaded": custom_model is not None}

@app.post("/compare")
def compare_sentiment(request: SentimentRequest):
    if not request.sentence or not request.sentence.strip():
        raise HTTPException(status_code=400, detail="Debes proporcionar una frase.")

    try:
        # Resultados del experto (FinBERT)
        res_expert = inference.predict(request.sentence, expert_model, expert_tok)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en FinBERT: {str(e)}")

    response = {
        "input": request.sentence,
        "benchmark_finbert": res_expert,  # ← clave consistente con test_api
        "custom_model": None
    }

    # Resultados modelo custom
    if custom_model:
        try:
            res_custom = inference.predict(request.sentence, custom_model, custom_tok)
            response["custom_model"] = res_custom
        except Exception as e:
            response["custom_model"] = {"error": str(e)}
    else:
        response["custom_model"] = {"label": "N/A", "confidence": 0.0, "info": "Modelo no entrenado aún."}

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)