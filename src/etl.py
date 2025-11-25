# src/etl.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.ml.feature import VectorAssembler

# Força Windows (deixa pra sempre)
os.environ['JAVA_HOME'] = r'C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot'
os.environ['HADOOP_HOME'] = r'C:\hadoop'
os.environ['PATH'] = os.environ['JAVA_HOME'] + r'\bin;' + os.environ['HADOOP_HOME'] + r'\bin;' + os.environ['PATH']

spark = SparkSession.builder \
    .appName("TrafficETL") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

def etl_process(file_path):
    df = spark.read \
        .option("header", "true") \
        .option("delimiter", ";") \
        .option("inferSchema", "true") \
        .csv(file_path)

    print("Colunas encontradas no CSV:")
    df.printSchema()

    # CORRIGE A COLUNA DE LENTIDÃO: "12,5" → "12.5" → 12.5
    df = df.withColumn(
        "slowness_clean",
        regexp_replace(col("Slowness in traffic (%)"), ",", ".").cast("double")
    )

    # Remove linhas com valor nulo na lentidão (só pra garantir)
    df = df.na.drop(subset=["slowness_clean"])

    # Features exatas do seu CSV
    feature_cols = [
        "Hour (Coded)", "Immobilized bus", "Broken Truck", "Vehicle excess",
        "Accident victim", "Running over", "Fire vehicles",
        "Occurrence involving freight", "Incident involving dangerous freight",
        "Lack of electricity", "Fire", "Point of flooding", "Manifestations",
        "Defect in the network of trolleybuses", "Tree on the road",
        "Semaphore off", "Intermittent Semaphore"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_final = assembler.transform(df)

    # Label limpo e como double
    df_final = df_final.withColumn("label", col("slowness_clean"))

    return df_final.select("features", "label", "slowness_clean", "Hour (Coded)").cache()