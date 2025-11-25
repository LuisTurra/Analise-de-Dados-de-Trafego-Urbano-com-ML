import pandas as pd
import numpy as np
from .dashboard import app, run_dashboard

app_server = app.server  

if __name__ == "__main__":
    print("Iniciando versão ONLINE — dashboard leve e lindo")

    
    np.random.seed(42)
    n_days = 30
    hours = np.tile(np.arange(1, 28), n_days)
    
    df = pd.DataFrame({
        'Hour (Coded)': hours,
        'label': np.abs(8 + 12*np.random.beta(1.5, 4, len(hours)) + 
                       25*(hours > 15) + np.random.normal(0, 3, len(hours))),
        'prediction': None
    })
    df['prediction'] = df['label'] + np.random.normal(0, 2.5, len(df))
    df['prediction'] = df['prediction'].clip(0, 100)

   
    run_dashboard(df)