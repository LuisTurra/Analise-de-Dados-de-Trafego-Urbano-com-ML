import sys
import os
import pyspark
from pyspark.sql import SparkSession


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from dashboard import app, run_dashboard
from etl import etl_process 
from model import train_model, explain_model


app_server = app.server

if __name__ == "__main__":
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java' 
    os.environ['HADOOP_HOME'] = '/opt/hadoop' 
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


    df = etl_process("data/traffic_data.csv")
    model, predictions = train_model(df)
    explain_model(model, predictions)
    run_dashboard(predictions)