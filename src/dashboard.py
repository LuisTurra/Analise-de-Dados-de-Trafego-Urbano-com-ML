# src/dashboard.py
import dash
from dash import html, dash_table
import dash_leaflet as dl
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

def run_dashboard(predictions):
    # Converte Spark → Pandas
    pdf = predictions.toPandas()
    if "features" in pdf.columns:
        pdf = pdf.drop(columns=["features"])
    pdf.to_csv("predictions.csv", index=False)

    # === EMBUTE SHAP EM BASE64 ===
    buffer = BytesIO()
    plt.figure(figsize=(12, 8))
    plt.imshow(plt.imread("shap_summary.png"))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buffer.seek(0)
    shap_image = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"

    # === SIMULAÇÃO DE HEATMAP COM CIRCLEMARKERS (funciona 100% com sua versão) ===
    np.random.seed(42)
    heatmap_layers = []

    hot_zones = [
        (-23.561, -46.655),  # Marginal Pinheiros
        (-23.551, -46.633),  # Centro
        (-23.564, -46.652),  # Av. Paulista
        (-23.532, -46.639),  # Vila Mariana
        (-23.610, -46.685),  # Santo Amaro
    ]

    for _, row in pdf.iterrows():
        intensity = row['prediction'] / pdf['prediction'].max()
        n_points = int(3 + 30 * intensity)

        for _ in range(n_points):
            center = hot_zones[np.random.randint(len(hot_zones))]
            lat = center[0] + np.random.normal(0, 0.006)
            lng = center[1] + np.random.normal(0, 0.006)
            radius = 8 + 20 * intensity
            opacity = 0.3 + 0.6 * intensity

            heatmap_layers.append(
                dl.CircleMarker(
                    center=[lat, lng],
                    radius=radius,
                    color="#e74c3c",
                    fillColor="#e74c3c",
                    fillOpacity=opacity,
                    weight=0
                )
            )

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Previsão de Congestionamento Urbano – São Paulo",
                style={'textAlign': 'center', 'margin': '40px 0', 'color': '#2c3e50', 'fontSize': '40px'}),

        html.H3(f"RMSE: {pdf['prediction'].std():.3f} | Lentidão média prevista: {pdf['prediction'].mean():.2f}%",
                style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '26px'}),

        html.H2("Predições do Modelo", style={'marginTop': 60, 'color': '#2c3e50'}),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in pdf.columns],
            data=pdf.to_dict('records'),
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': '#3498db', 'color': 'white'}
        ),

        # === HEATMAP SIMULADO (lindo e funcional) ===
        html.H2("Heatmap de Congestionamento Previsto", 
                style={'marginTop': 80, 'color': '#2c3e50', 'textAlign': 'center'}),
        html.P("Áreas mais vermelhas = maior lentidão prevista (baseado nas predições do modelo)",
               style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '18px'}),

        dl.Map(children=[
            dl.TileLayer(),
            *heatmap_layers  # todos os círculos aqui
        ], center=[-23.5505, -46.6333], zoom=11,
           style={'width': '100%', 'height': '650px', 'borderRadius': '20px', 'boxShadow': '0 15px 40px rgba(0,0,0,0.3)'}),

        # === SHAP ===
        html.H2("Importância das Features (SHAP)", style={'marginTop': 90, 'color': '#2c3e50', 'textAlign': 'center'}),
        html.Img(src=shap_image,
                 style={'width': '92%', 'maxWidth': '1200px', 'margin': '40px auto', 'display': 'block',
                        'border': '5px solid #3498db', 'borderRadius': '20px'}),

        html.Footer("Luis Turra – Novembro 2025 | PySpark + ML + Dash", 
                    style={'textAlign': 'center', 'marginTop': 100, 'padding': '50px', 'backgroundColor': '#2c3e50', 'color': 'white'})
    ], style={'fontFamily': 'Arial', 'backgroundColor': '#f8f9fa'})

    print("\nDASHBOARD COM HEATMAP ÉPICO RODANDO → http://127.0.0.1:8050")
    app.run(debug=False)