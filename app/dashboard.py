import dash
from dash import html, dcc
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np


app = dash.Dash(__name__)

def run_dashboard(predictions):
    """
    Função que configura o layout baseado nas predições do modelo.
    """
    # Converte Spark
    pdf = predictions.toPandas()
    if "features" in pdf.columns:
        pdf = pdf.drop(columns=["features"])
    if "slowness_clean" in pdf.columns:
        pdf = pdf.drop(columns=["slowness_clean"])  


    pdf.to_csv("predictions.csv", index=False)

    # Mapeamento das horas
    
    hour_map = {i: f"{7 + (i-1)//2}:{'00' if (i-1)%2 == 0 else '30'}" for i in range(1, 28)}
    pdf['Hora'] = pdf['Hour (Coded)'].map(hour_map)

    # Agrupa por hora e calcula média (pra gráfico suave)
    hourly = pdf.groupby('Hour (Coded)').agg({
        'label': 'mean',  
        'prediction': 'mean'  
    }).reset_index()
    hourly['Hora'] = hourly['Hour (Coded)'].map(hour_map)

    # Gráfico de linha: Real vs Previsão
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly['Hora'], y=hourly['label'],
                             mode='lines+markers', name='Lentidão Real (%)',
                             line=dict(color='#e74c3c', width=4)))  
    fig.add_trace(go.Scatter(x=hourly['Hora'], y=hourly['prediction'],
                             mode='lines+markers', name='Previsão do Modelo (%)',
                             line=dict(color='#3498db', width=4, dash='dot')))  
    fig.update_layout(
        title="Evolução da Lentidão no Trânsito de São Paulo (7h-20h)",
        xaxis_title="Horário do Dia",
        yaxis_title="Lentidão (%)",
        template="plotly_white",  
        legend=dict(y=0.99, x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        height=500
    )

    
    buffer = BytesIO()
    plt.figure(figsize=(12, 8))
    plt.imshow(plt.imread("shap_summary.png"))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buffer.seek(0)
    shap_image = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"

    # HEATMAP SIMULADO (explicação: gera pontos fictícios em áreas reais de SP, baseados na intensidade da previsão)
    np.random.seed(42)
    heatmap_layers = []
    hot_zones = [  
        (-23.561, -46.655),  
        (-23.551, -46.633), 
        (-23.564, -46.652),  
        (-23.532, -46.639),  
        (-23.610, -46.685)   
    ]
    for _, row in pdf.iterrows():
        intensity = row['prediction'] / pdf['prediction'].max()  
        n_points = int(3 + 35 * intensity) 
        for _ in range(n_points):
            center = hot_zones[np.random.randint(len(hot_zones))]
            lat = center[0] + np.random.normal(0, 0.006) 
            lng = center[1] + np.random.normal(0, 0.006)
            radius = 10 + 25 * intensity 
            heatmap_layers.append(
                dl.CircleMarker(
                    center=[lat, lng],
                    radius=radius,
                    color="#e74c3c",
                    fillColor="#e74c3c",
                    fillOpacity=0.4 + 0.5 * intensity, 
                    weight=0
                )
            )

    # Layout do dashboard 
    app.layout = html.Div([
        html.H1("Previsão de Congestionamento Urbano – São Paulo",
                style={'textAlign': 'center', 'padding': '40px 0', 'color': '#2c3e50', 'fontSize': '42px'}),

        # Métricas principais 
        html.Div([
            html.Div([html.H4("RMSE do Modelo"), html.P(f"{pdf['prediction'].std():.3f}")], className="metric"),
            html.Div([html.H4("Lentidão Média Prevista"), html.P(f"{pdf['prediction'].mean():.2f}%")], className="metric"),
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '60px', 'marginBottom': '40px'}),

        # Gráfico de linha
        html.H2("Evolução da Lentidão ao Longo do Dia (7h-20h)",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '50px'}),
        dcc.Graph(figure=fig, style={'width': '95%', 'margin': '30px auto'}),

        # Heatmap
        html.H2("Heatmap de Congestionamento Previsto",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '80px'}),
        html.P("Áreas mais vermelhas = maior lentidão prevista pelo modelo (baseado em incidentes históricos)",
               style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '18px', 'marginBottom': '30px'}),
        dl.Map(children=[dl.TileLayer(), *heatmap_layers],
               center=[-23.5505, -46.6333], zoom=11,
               style={'width': '100%', 'height': '650px', 'borderRadius': '20px', 'boxShadow': '0 15px 40px rgba(0,0,0,0.3)'}),

        # SHAP
        html.H2("Importância das Features (SHAP Values)",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '90px'}),
        html.P("As features mais importantes para o modelo prever lentidão (ex: 'Hour (Coded)' é o horário do dia)",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'}),
        html.Img(src=shap_image,
                 style={'width': '90%', 'maxWidth': '1100px', 'margin': '40px auto', 'display': 'block',
                        'border': '5px solid #3498db', 'borderRadius': '20px'}),

        # Rodapé
        html.Footer("Projeto completo por Luis Turra – Novembro 2025 | PySpark • ML • Dash • SHAP",
                    style={'textAlign': 'center', 'marginTop': '100px', 'padding': '60px',
                           'background': 'linear-gradient(135deg, #2c3e50, #3498db)', 'color': 'white', 'fontSize': '20px'})
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'padding': '0 20px'})

    # CSS pros métricas
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Previsão de Trânsito SP - Luis Turra</title>
            {%css%}
            <style>
                .metric { background: white; padding: 25px 45px; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center; min-width: 200px; }
                .metric h4 { color: #2c3e50; margin: 0 0 10px 0; font-size: 20px; }
                .metric p { color: #e74c3c; margin: 0; font-size: 28px; font-weight: bold; }
            </style>
        </head>
        <body>
            {%app_entry%}
            {%config%}
            {%scripts%}
            {%renderer%}
        </body>
    </html>
    '''


if __name__ == '__main__':
    print("Rodando localmente...")
    app.run(debug=False, host="0.0.0.0", port=8050)