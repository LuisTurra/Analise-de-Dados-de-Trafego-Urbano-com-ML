from .dashboard import app, run_dashboard
from .model import train_model
from etl import etl_process

app_server = None

if __name__ == "__main__":
    print("Iniciando ETL...")
    df = etl_process("data/traffic_data.csv")
    print("Treinando modelo...")
    model, predictions = train_model(df)
    print("Gerando explicabilidade SHAP...")
    print("Iniciando dashboard...")
    run_dashboard(predictions)

    from dashboard import app
    app_server = app.server  