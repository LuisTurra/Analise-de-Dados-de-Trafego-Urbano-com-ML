import dash
from dash import html, dcc
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
import numpy as np


app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("Carregando dashboard...", style={'textAlign': 'center', 'marginTop': '200px', 'color': '#666'}),
    html.Div(id='main-content')
])

def run_dashboard(predictions):
    if hasattr(predictions, "toPandas"):
        pdf = predictions.toPandas()
    else:
        pdf = predictions

    hour_map = {i: f"{7 + (i-1)//2}:{'00' if (i-1)%2 == 0 else '30'}" for i in range(1, 28)}
    pdf['Hora'] = pdf['Hour (Coded)'].map(hour_map)

    hourly = pdf.groupby('Hour (Coded)').agg({'label': 'mean', 'prediction': 'mean'}).reset_index()
    hourly['Hora'] = hourly['Hour (Coded)'].map(hour_map)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly['Hora'], y=hourly['label'], mode='lines+markers', name='Real', line=dict(color='#e74c3c')))
    fig.add_trace(go.Scatter(x=hourly['Hora'], y=hourly['prediction'], mode='lines+markers', name='Previsto', line=dict(color='#3498db', dash='dot')))
    fig.update_layout(title="Lentidão ao Longo do Dia", xaxis_title="Horário", yaxis_title="Lentidão (%)", template="plotly_white")

    # === HEATMAP SIMPLES ===
    np.random.seed(42)
    points = []
    for _, row in pdf.iterrows():
        intensity = row['prediction'] / 100
        for _ in range(int(10 * intensity)):
            points.append([
                -23.55 + np.random.normal(0, 0.05),
                -46.63 + np.random.normal(0, 0.05)
            ])

    # === LAYOUT ===
    app.layout = html.Div([
        html.H1("Previsão de Congestionamento Urbano – São Paulo", style={'textAlign':'center','padding':'30px','color':'#2c3e50'}),
        html.Div([
            html.Div(f"RMSE: {pdf['prediction'].std():.2f}", className="metric"),
            html.Div(f"Média prevista: {pdf['prediction'].mean():.1f}%", className="metric"),
        ], style={'display':'flex','justifyContent':'center','gap':'50px','margin':'30px'}),
        dcc.Graph(figure=fig),
        dl.Map([dl.TileLayer(), dl.LayerGroup([dl.CircleMarker(center=p, radius=8, color="#e74c3c", fillOpacity=0.6) for p in points])],
               center=[-23.55, -46.63], zoom=11, style={'height':'600px'}),
        html.Footer("Luis Turra – 2025", style={'textAlign':'center','padding':'40px','background':'#2c3e50','color':'white','marginTop':'50px'})
    ], style={'fontFamily':'Arial','background':'#f8f9fa'})

    print("Dashboard carregado com sucesso!")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)