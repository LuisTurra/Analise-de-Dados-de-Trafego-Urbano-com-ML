# Projeto: Análise de Tráfego Urbano com ML em Escala

## Descrição
Pipeline de ML com PySpark para prever congestionamentos em São Paulo. Inclui ETL, treinamento, explicabilidade e dashboard.

## Instalação
1. Clone o repo: `git clone https://github.com/luisturra/traffic-ml-bigdata.git`
2. Instale: `pip install -r requirements.txt`
3. Baixe dataset e coloque em `data/`.

## Como rodar

### Local (com PySpark + modelo real + SHAP)
```bash
pip install -r requirements.txt
python -m pip install pyspark==3.5.2  # só uma vez
python app/main.py

## Demo
-

## Dataset
Use: https://archive.ics.uci.edu/dataset/483/behavior+of+the+urban+traffic+of+the+city+of+sao+paulo+in+brazil

## Desafios Superados
- Processamento distribuído com PySpark para big data.
- Integração de mapa em Render.