import kagglehub
import pandas as pd
import requests
import os

def test_api_with_kaggle(n_samples):
    # 1. Descarga del Dataset de Kaggle
    print("--- Descargando datos de Kaggle ---")
    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
    csv_path = os.path.join(path, "data.csv")
    df = pd.read_csv(csv_path)

    # 2. Configuración de la URL de API 
    API_URL = "http://127.0.0.1:8000/predict"
    
    # 3. Selección de muestra aleatoria
    sample_df = df.sample(n_samples)
    
    print(f"--- Enviando {n_samples} frases a la API ---")
    comparison_results = []

    for _, row in sample_df.iterrows():
        sentence = row['Sentence']
        real_label = row['Sentiment']

        # 4. Petición HTTP POST a la API
        try:
            response = requests.post(API_URL, json={"sentence": sentence})
            
            if response.status_code == 200:
                api_response = response.json()
                comparison_results.append({
                    "Sentence": sentence[:60] + "...",
                    "Real (Kaggle)": real_label,
                    "API_Predict": api_response["sentiment"],
                    "Conf %": api_response["confidence"]
                })
            else:
                print(f" Error en API: Código {response.status_code}")
        except Exception as e:
            print(f" Error de conexión: {e}")
            print("Asegúrate de que uvicorn esté corriendo en otra terminal.")
            break

    # 5. Visualización de resultados
    if comparison_results:
        results_df = pd.DataFrame(comparison_results)
        print("\n" + "="*100)
        print(" RESULTADOS DE LA API vs KAGGLE")
        print("="*100)
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string(index=False))
        print("="*100)

if __name__ == "__main__":
    test_api_with_kaggle(n_samples=10)