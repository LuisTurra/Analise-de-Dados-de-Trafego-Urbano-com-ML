import pandas as pd
import numpy as np
from .dashboard import app, run_dashboard

app_server = app.server 

if __name__ == "__main__":
    print("Iniciando versão ONLINE (sem PySpark — ideal pra portfólio)")
    
    
    np.random.seed(42)
    n_days = 30
    hours_per_day = 27  
    total = n_days * hours_per_day
    
    data = {
        'Hour (Coded)': np.tile(np.arange(1, 28), n_days),
        'label': np.abs(8 + 12 * np.random.beta(1.5, 4, total) + 
                       25 * (np.tile(np.arange(1, 15, 22), n_days)) +  
                       np.random.normal(0, 3, total)),
        'prediction': None
    }
    df = pd.DataFrame(data)
    df['prediction'] = df['label'] + np.random.normal(0, 2.5, total)  
    df['prediction'] = df['prediction'].clip(0, 100)
    
    print(f"Gerados {len(df)} registros mock — RMSE médio ~2.5")
    print("Iniciando dashboard...")

    run_dashboard(df)  