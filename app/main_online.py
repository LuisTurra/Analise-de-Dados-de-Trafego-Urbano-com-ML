import pandas as pd
from .dashboard import app, run_dashboard

app_server = app.server

try:
    df = pd.read_csv("static/real_predictions.csv")
    print(f"Carregadas {len(df)} predições REAIS do modelo treinado!")
except Exception as e:
    print("Erro ao carregar real_predictions.csv:", e)
    print("Usando mock como fallback...")
    import numpy as np
    np.random.seed(42)
    hours = np.tile(np.arange(1, 28), 30)
    df = pd.DataFrame({
        'Hour (Coded)': hours,
        'label': np.random.normal(50, 15, len(hours)).clip(0, 100),
        'prediction': np.random.normal(48, 16, len(hours)).clip(0, 100)
    })

run_dashboard(df)

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=False)