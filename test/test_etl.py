import pytest
from src.etl import etl_process, spark

def test_etl_process():
    file_path = "../data/traffic_data.csv"
    df = etl_process(file_path)
    assert df.count() > 0, "Dataset vazio após ETL"
    assert "features" in df.columns, "Coluna de features ausente"
    assert "label" in df.columns, "Coluna de label ausente"

# Rode com: pytest tests/test_etl.py