from etl import etl_process, spark
from model import train_model, explain_model
from dashboard import run_dashboard

if __name__ == "__main__":
    file_path = r"C:\Users\Microsoft\Desktop\Projects\Python Projects\Analise-de-Dados-de-Trafego-Urbano-com-ML\data\traffic_data.csv"
    df = etl_process(file_path)
    model, predictions = train_model(df)
    explain_model(model, predictions)
    run_dashboard(predictions)  # Chama Dash dashboard diretamente
    spark.stop()