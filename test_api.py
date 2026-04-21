import requests
import pandas as pd
import kagglehub
import os
import config

def test_api_hybrid(n_samples=10):
    # 1. Protección de Disco D:
    os.environ['KAGGLEHUB_CACHE'] = os.path.join(config.BASE_PATH, "kaggle_cache")

    # 2. Descarga del dataset
    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
    df = pd.read_csv(os.path.join(path, "data.csv")).sample(n_samples)

    comparison_results = []
    print(f"--- Comparando {n_samples} frases en la API ---")

    for _, row in df.iterrows():
        sentence = row['Sentence']

        try:
            # FIX 1: La clave debe ser "sentence" (coincide con SentimentRequest en main.py)
            response = requests.post(
                "http://127.0.0.1:8000/compare",
                json={"sentence": sentence},
                timeout=30
            )

            if response.status_code == 200:
                res = response.json()

                # FIX 2: Usar "benchmark_finbert" (nombre correcto de main.py)
                finbert = res.get("benchmark_finbert", {})

                # FIX 3: custom_model siempre es dict ahora, pero lo protegemos igual
                custom = res.get("custom_model", {})
                custom_label = custom.get("label", "N/A") if isinstance(custom, dict) else "N/A"
                custom_conf  = custom.get("confidence", 0.0) if isinstance(custom, dict) else 0.0

                comparison_results.append({
                    "Frase":          sentence[:50] + "...",
                    "Real (Kaggle)":  row['Sentiment'],
                    "TU MODELO":      custom_label,
                    "FINBERT":        finbert.get("label", "N/A"),
                    "Conf Custom %":  custom_conf,
                    "Conf FinBERT %": finbert.get("confidence", 0.0),
                })
            else:
                print(f"[HTTP {response.status_code}] {response.text} | Frase: {sentence[:60]}")

        except requests.exceptions.ConnectionError:
            print("Error de conexión: ¿Está corriendo el servidor en puerto 8000?")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")

    if comparison_results:
        results_df = pd.DataFrame(comparison_results)
        print("\n" + "=" * 110)
        print(results_df.to_string(index=False))
        print("=" * 110)
    else:
        print("No se obtuvieron resultados. Verifica que el servidor esté activo.")

if __name__ == "__main__":
    test_api_hybrid(10)