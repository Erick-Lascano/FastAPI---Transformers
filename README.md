# FastAPI---Transformers  

## Financial Sentiment Analysis – Cloud Computing Project  

Este proyecto implementa un sistema de análisis de sentimientos para textos financieros utilizando arquitecturas de **Transformers**. El sistema está diseñado de forma modular, permitiendo desde el entrenamiento de modelos personalizados hasta su exposición mediante una **API REST**.

---

## 🏗️ Arquitectura del Proyecto  

El sistema se organiza en capas siguiendo principios de ingeniería de software para facilitar el despliegue en entornos Cloud.

* **`src/api/`**: Contiene la lógica del servicio web (FastAPI), carga de modelos y funciones de inferencia.  
* **`src/training/`**: Scripts dedicados a la extracción de datos, tokenización y el pipeline de entrenamiento del Transformer.  
* **`fin_sentiment_model/`**: Carpeta donde se almacenan los pesos del modelo, la configuración y el vocabulario del tokenizador (Artifacts).  
* **`audit_api.py`**: Script de validación que garantiza que el modelo generalice correctamente sin *Data Leakage*.  

---

## 🐳 Contenerización con Docker  

Para asegurar la portabilidad del sistema, se utiliza **Docker**, permitiendo ejecutar la API en cualquier entorno sin conflictos de dependencias.

### Construcción de la imagen

```bash
docker build -t financial-sentiment-api:v1 .
```

Ejecución local del contenedor
docker run -p 8000:8000 financial-sentiment-api:v1


La API estará disponible en:

http://localhost:8000/docs

## Despliegue en Azure

El proyecto está diseñado para ser desplegado como un servicio web en la nube utilizando Microsoft Azure.

Infraestructura utilizada
Azure Container Registry (ACR): Almacenamiento y versionado de imágenes Docker
Azure App Service (Linux): Hosting del contenedor
Configuración clave:
Puerto: WEBSITES_PORT=8000
SKU recomendado: B1 (≥ 1.75 GB RAM para modelos Transformer)
Endpoint de producción

https://dockertransformer-amfzfrd0c4dzhhda.mexicocentral-01.azurewebsites.net/docs

## 🚀 Instrucciones de Ejecución



### 1. Preparación del Entorno

Es recomendable utilizar un entorno virtual para evitar conflictos de dependencias.

```bash

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

```



### 2. Entrenamiento del Modelo

Para entrenar el modelo desde cero o realizar un *fine-tuning* con los datos de Kaggle, ejecuta el script de entrenamiento. Este proceso descargará los datos, los procesará y guardará el "cerebro" resultante en la carpeta local.

```bash

python src/training/train.py

```

*Nota: Se recomienda ejecutar este paso en un entorno con GPU, esta implementación está diseñada para ser ejecutada en Google Colab.*



Después, la carpeta resultante `fin_sentiment_model` debe ser descargada a la raíz de este proyecto.*



### 3. Lanzamiento de la API (Servicio de Inferencia)

Una vez que el modelo esté presente en la carpeta `fin_sentiment_model/`, el script principal (`main.py`) orquestará el levantamiento del servidor local.

```bash

python main.py

```

La API estará disponible en `http://0.0.0.0:8000`. Puedes acceder a la documentación interactiva en `/docs`.



## 🧠 Componentes Principales



### Inferencia

El sistema utiliza un bloque `if __name__ == "__main__":` en `main.py` para que el servidor **Uvicorn** se inicie únicamente cuando el script es el punto de entrada, cargando el modelo en memoria una sola vez.



### Manejo de Datos

Se implementó una división de datos (`train_test_split`) con una semilla fija para asegurar la reproducibilidad de los resultados y evitar que el modelo "memorice" las respuestas del examen de auditoría.





## 🛠️ Tecnologías Utilizadas

* **Python**

* **Hugging Face Transformers**: Para el manejo del modelo pre-entrenado.

* **FastAPI / Uvicorn**: Para el servicio de API REST.

* **PyTorch**: Como motor de tensores y optimización.

* **Scikit-learn**: Para métricas y división de datasets.



---



### Equipo 1:



Erick Isaac Lascano Otañez - A00836571



Luis Fernando Alcazar Díaz - A00836287



Pedro Soto Juárez - A00837560



Alexei Carrillo Acosta - A01285424



Mateo Zepeda Pérez - A01722398



---
