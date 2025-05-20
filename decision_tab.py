from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from info_compartida import DATAFRAMES, PROCESS_DATASET

class DecisionTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        return html.Div([
            html.H2("Toma de Decisiones", style={"margin": "20px"}),

            html.H3("🎯 Objetivo del Sistema", style={"margin": "10px"}),
            html.P(
                "Este sistema permite analizar el comportamiento de reservas hoteleras para ayudar en la toma de decisiones estratégicas, "
                "como detectar cancelaciones probables, patrones de estacionalidad y segmentación de clientes.",
                style={"marginBottom": "30px"}
            ),

            html.H3("📊 Resultados del Análisis Clustering", style={"margin": "10px"}),

            html.Div([
                html.Div([
                    html.Label("Selecciona el eje X:"),
                    dcc.Dropdown(id='x-axis-dropdown-decision', placeholder="Selecciona variable X", style={'width': '300px'})
                ], style={'display': 'inline-block', 'marginRight': '40px'}),

                html.Div([
                    html.Label("Selecciona el eje Y:"),
                    dcc.Dropdown(id='y-axis-dropdown-decision', placeholder="Selecciona variable Y", style={'width': '300px'})
                ], style={'display': 'inline-block'}),
            ], style={'margin': '20px'}),

            dcc.Loading(
                id="loading-cluster-decision",
                type="circle",
                children=[dcc.Graph(id="cluster-decision-graph")]
            ),

            html.Div(id='decision-explanation', style={"margin": "20px", "fontStyle": "italic", "color": "#333"}),

            html.H4("🔍 Recomendaciones estratégicas basadas en clústeres", style={"marginTop": "30px"}),
            html.Ul([
                html.Li("🟩 El clúster más frecuente representa el comportamiento dominante. Considera adaptar promociones a este grupo."),
                html.Li("🟧 Identifica clústeres con baja frecuencia pero alto valor. Podrían representar clientes premium o leales."),
                html.Li("🟥 Si algún clúster tiene alta tasa de cancelaciones, analiza políticas de retención o comunicación proactiva."),
                html.Li("📅 Usa la segmentación temporal de clústeres para anticipar estacionalidades y optimizar recursos."),
            ], style={"margin": "20px"}),
        ])

    def register_callbacks(self):
        @self.app.callback(
            Output('x-axis-dropdown-decision', 'options'),
            Output('y-axis-dropdown-decision', 'options'),
            Input('x-axis-dropdown-decision', 'id')
        )
        def update_variable_options(_):
            if 'cluster_result' not in PROCESS_DATASET or PROCESS_DATASET['cluster_result'].empty:
                return [], []
            df = PROCESS_DATASET['cluster_result']
            numeric_cols = df.select_dtypes(include='number').columns
            options = [{'label': col, 'value': col} for col in numeric_cols if col != 'cluster']
            return options, options

        @self.app.callback(
            Output('cluster-decision-graph', 'figure'),
            Output('decision-explanation', 'children'),
            Input('x-axis-dropdown-decision', 'value'),
            Input('y-axis-dropdown-decision', 'value')
        )
        def update_cluster_graph(x_col, y_col):
            if not x_col or not y_col:
                raise PreventUpdate

            if 'cluster_result' not in PROCESS_DATASET:
                return px.scatter(), "No hay datos de clustering disponibles."

            df = PROCESS_DATASET['cluster_result']
            if df.empty or 'cluster' not in df.columns:
                return px.scatter(), "No hay datos válidos para graficar."

            filtered_df = df[[x_col, y_col, 'cluster']].dropna()
            if filtered_df.empty:
                return px.scatter(), "No hay suficientes datos válidos para graficar."

            # Colores manuales por clúster
            color_map = {
                '0': '#2ca02c',  # 🟩
                '1': '#ff7f0e',  # 🟧
                '2': '#d62728'   # 🟥
            }

            filtered_df['cluster'] = filtered_df['cluster'].astype(str)

            fig = px.scatter(
                filtered_df, x=x_col, y=y_col, color='cluster',
                color_discrete_map=color_map,
                title=f"Clustering de Clientes: {x_col} vs {y_col}",
                labels={'cluster': 'Clúster'},
                opacity=0.8
            )

            distribution = filtered_df['cluster'].value_counts().sort_index()
            centroides = filtered_df.groupby('cluster')[[x_col, y_col]].mean().round(2)

            explicacion = html.Div([
                html.P("Distribución de clústeres: " +
                       ", ".join([f"Clúster {i}: {n} registros" for i, n in distribution.items()])),
                html.Br(),
                html.P("📌 Interpretación de centroides:"),
                html.Ul([
                    html.Li(f"{['🟩', '🟧', '🟥'][int(i)]} Clúster {i}: media({x_col}) = {row[x_col]}, media({y_col}) = {row[y_col]}")
                    for i, row in centroides.iterrows()
                ])
            ])

            return fig, explicacion