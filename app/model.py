# import shap
# import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

def train_model(df):
    feature_cols = [col for col in df.columns if col not in ["Slowness in traffic (%)", "slowness_clean"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    rf = RandomForestRegressor(featuresCol="features", labelCol="slowness_clean", numTrees=100)
    pipeline = Pipeline(stages=[assembler, rf])
    
    # Treino/teste
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_data)
    
    predictions = model.transform(test_data)
    predictions = predictions.withColumnRenamed("slowness_clean", "label")
    
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE do modelo: {rmse:.3f}")
    
    return model, predictions

# SHAP desativado no Render
def explain_model(model, predictions):
    print("SHAP desativado no Render – imagem estática carregada")
    
#def explain_model(model, data):
    # """
    # Explicabilidade com SHAP  
    # """
    # try:
        
    #     import shap
    #     import pandas as pd
    #     import numpy as np

    #     rf_model = model.stages[-1]

    #     pdf = data.select("features").toPandas()
    #     X = np.array(pdf["features"].tolist())  

    #     explainer = shap.TreeExplainer(rf_model)
    #     shap_values = explainer.shap_values(X)

    #     print("\nSHAP Summary Plot gerado com sucesso!")
    #     shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    #     import matplotlib.pyplot as plt
    #     plt.savefig("shap_summary.png", bbox_inches='tight', dpi=150)
    #     plt.close()
    #     print("Gráfico salvo como shap_summary.png na raiz do projeto!")

    # except Exception as e:
    #     print(f"SHAP falhou : {e}")
    #     print("Mas o modelo e o dashboard funcionam perfeitamente!")]
    
