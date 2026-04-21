BASE_PATH = r"D:\Az\FastAPI---Transformers"
MODEL_NAME = "distilbert-base-uncased"  # Model
FINBERT_NAME = "ProsusAI/finbert"       # Benchmark
CUSTOM_MODEL_PATH = r"D:\Az\FastAPI---Transformers\my_fin_model"    

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 3  # Son 3 sentimientos (negativo, neutral, positivo)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3      