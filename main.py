from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_trained_model
from inference import get_prediction

app = FastAPI(title="Financial Sentiment API")

# Cargamos el modelo una sola vez al iniciar la app
MODEL_PATH = "./fin_sentiment_model"
model, tokenizer = load_trained_model(MODEL_PATH)

class TextRequest(BaseModel):
    sentence: str

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    sentiment, confidence = get_prediction(request.sentence, model, tokenizer)
    return {
        "sentence": request.sentence,
        "sentiment": sentiment,
        "confidence": f"{confidence:.2%}"
    }