
import os
from app.etl import etl_process
from app.model import train_model
from app.dashboard import app, run_dashboard


os.environ['JAVA_HOME'] = r'C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot'
os.environ['HADOOP_HOME'] = r'C:\hadoop'

if __name__ == "__main__":
    print("Iniciando ETL com PySpark...")
    df = etl_process("data/traffic_data.csv")

    print("Treinando modelo Random Forest (30-60 segundos)...")
    best_model, predictions = train_model(df)

    print("Gerando dashboard com dados REAIS do modelo treinado...")
    run_dashboard(predictions)  

    print("Abrindo no navegador: http://127.0.0.1:8050")
    print("Feche com Ctrl+C")

   
    app.run(host="127.0.0.1", port=8050, debug=False)