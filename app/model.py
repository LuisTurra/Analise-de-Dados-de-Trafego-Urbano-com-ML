from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np

def train_model(df):
    """
    Recebe o DataFrame já com 'features' (do etl.py)
    Cria a coluna 'label' aqui (só aqui, pra evitar ambiguidade)
    Treina Random Forest e retorna modelo + predições
    """
    
   
    df = df.withColumn("label", df.slowness_clean)

    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

  
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=42
    )


    pipeline = Pipeline(stages=[rf])

 
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [8, 10]) \
        .build()

    
    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )

  
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42
    )

    print("Treinando modelo Random Forest com Cross-Validation...")
    cv_model = crossval.fit(train_df)

 
    best_model = cv_model.bestModel
    rf_model = best_model.stages[0]

    print(f"Melhores parâmetros:")
    print(f"  numTrees: {rf_model.getNumTrees}")
    print(f"  maxDepth: {rf_model.getMaxDepth()}")

    
    predictions = best_model.transform(test_df)

  
    rmse = evaluator.evaluate(predictions)
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    print(f"\nRESULTADOS DO MODELO:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")


    predictions = predictions.select(
        "Hour (Coded)", "label", "prediction"
    )

    return best_model, predictions