FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias de compilación para PyTorch/Transformers
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el proyecto
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Ejecutamos la API (no el entrenamiento, el entrenamiento se hace antes)
CMD ["python", "main.py"]