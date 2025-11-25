# Projeto: Análise de Tráfego Urbano com ML em Escala

## Descrição
Pipeline de ML com PySpark para prever congestionamentos em São Paulo. Inclui ETL, treinamento, explicabilidade e dashboard.

## Instalação
1. Clone o repo: `git clone https://github.com/luisturra/traffic-ml-bigdata.git`
2. Instale: `pip install -r requirements.txt`
3. Baixe dataset e coloque em `data/`.

## Uso
- Rode pipeline: `python src/main.py`
- Dashboard: `streamlit run src/dashboard.py`
- Testes: `pytest tests/`

## Dataset
Use: https://archive.ics.uci.edu/dataset/483/behavior+of+the+urban+traffic+of+the+city+of+sao+paulo+in+brazil

## Resultados
RMSE exemplo: 3.070 (depende do treino). SHAP para explicações.


## Desafios Superados
- Processamento distribuído com PySpark para big data.
- Integração de mapa em Render.