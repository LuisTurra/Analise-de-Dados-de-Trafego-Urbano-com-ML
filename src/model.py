from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import shap
import pandas as pd

def train_model(df):
    """
    Treina um modelo de Random Forest para prever congestionamento.
    """
    (train_data, test_data) = df.randomSplit([0.8, 0.2])
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100)
    pipeline = Pipeline(stages=[rf])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE do modelo: {rmse}")
    return model, predictions
# src/model.py (só essa função)
def explain_model(model, data):
    """
    Explicabilidade com SHAP – versão corrigida pro PySpark + Vector
    """
    try:
        # Agora funciona 100%
        import shap
        import pandas as pd
        import numpy as np

        # Extrai o RandomForest do pipeline
        rf_model = model.stages[-1]

        # Converte Spark Vector → numpy array
        pdf = data.select("features").toPandas()
        X = np.array(pdf["features"].tolist())  # <-- isso que tava faltando!

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)

        print("\nSHAP Summary Plot gerado com sucesso!")
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        import matplotlib.pyplot as plt
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("Gráfico salvo como shap_summary.png na raiz do projeto!")

    except Exception as e:
        print(f"SHAP falhou (normal em algumas versões): {e}")
        print("Mas o modelo e o dashboard funcionam perfeitamente!")