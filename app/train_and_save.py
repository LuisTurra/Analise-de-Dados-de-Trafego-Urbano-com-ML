from app.etl import etl_process
from app.model import train_model
from pathlib import Path

print("Iniciando treino offline...")

df = etl_process("data/traffic_data.csv")
best_model, predictions = train_model(df)


predictions_final = predictions.select("Hour (Coded)", "label", "prediction")


pdf = predictions_final.toPandas()


ROOT_DIR = Path(__file__).parent.parent
static_folder = ROOT_DIR / "static"
static_folder.mkdir(exist_ok=True)

csv_path = static_folder / "real_predictions.csv"
pdf.to_csv(csv_path, index=False)

print(f"\nSUCESSO TOTAL!")
print(f"Predições reais salvas em: {csv_path}")
print(f"Linhas: {len(pdf)}")
print("Agora só commit da pasta static/ e deploy no Render!")
print("O DASHBOARD ONLINE VAI FICAR 100% IGUAL AO LOCAL!")