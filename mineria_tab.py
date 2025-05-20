from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from dash.exceptions import PreventUpdate
from info_compartida import DATAFRAMES, PROCESS_DATASET

class MineriaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        return html.Div([
            html.H2("Minería de Datos", style={'margin': '20px'}),

            html.Label("Selecciona el número de clústeres (K):", style={'margin': '10px'}),
            dcc.Input(id='num-clusters', type='number', value=3, min=1, step=1),

            html.Div([
                html.Div([
                    html.Label("Selecciona el eje X:"),
                    dcc.Dropdown(id='x-axis-dropdown', placeholder="Variable eje X", style={'width': '300px'})
                ], style={'display': 'inline-block', 'marginRight': '40px'}),

                html.Div([
                    html.Label("Selecciona el eje Y:"),
                    dcc.Dropdown(id='y-axis-dropdown', placeholder="Variable eje Y", style={'width': '300px'})
                ], style={'display': 'inline-block'}),
            ], style={'margin': '20px'}),

            html.Button("Aplicar K-Means", id='apply-kmeans', n_clicks=0,
                        style={'margin': '10px', 'backgroundColor': '#007bff', 'color': 'white',
                               'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px'}),

            html.Div(id='cluster-explanation', style={'margin': '20px'}),
            dcc.Graph(id='kmeans-graph')
        ])

    def register_callbacks(self):
        @self.app.callback(
            Output('x-axis-dropdown', 'options'),
            Output('y-axis-dropdown', 'options'),
            Input('x-axis-dropdown', 'id')
        )
        def update_dropdown_options(_):
            if 'processed_data' not in DATAFRAMES or DATAFRAMES['processed_data'].empty:
                return [], []

            df = DATAFRAMES['processed_data']
            numeric_cols = df.select_dtypes(include='number').columns
            options = [{'label': col, 'value': col} for col in numeric_cols]
            return options, options

        @self.app.callback(
            Output('kmeans-graph', 'figure'),
            Output('cluster-explanation', 'children'),
            Input('apply-kmeans', 'n_clicks'),
            State('num-clusters', 'value'),
            State('x-axis-dropdown', 'value'),
            State('y-axis-dropdown', 'value')
        )
        def apply_kmeans(n_clicks, k, x_col, y_col):
            if n_clicks == 0:
                raise PreventUpdate

            if 'processed_data' not in DATAFRAMES or DATAFRAMES['processed_data'].empty:
                return px.scatter(), "No hay datos disponibles."

            df = DATAFRAMES['processed_data']

            if not x_col or not y_col:
                return px.scatter(), "Selecciona columnas válidas para X y Y."

            try:
                clustering_data = df[[x_col, y_col]].dropna()
                if clustering_data.empty:
                    return px.scatter(), "No hay suficientes datos sin nulos para aplicar K-Means."

                kmeans = KMeans(n_clusters=k, random_state=0)
                clustering_data['cluster'] = kmeans.fit_predict(clustering_data)
                clustering_data['cluster'] = clustering_data['cluster'].astype(str)

                PROCESS_DATASET['cluster_result'] = clustering_data  # NO se sobrescribe processed_data

                color_map = {
                    '0': '#2ca02c',
                    '1': '#ff7f0e',
                    '2': '#d62728'
                }

                fig = px.scatter(
                    clustering_data, x=x_col, y=y_col, color='cluster',
                    color_discrete_map=color_map,
                    title=f"Clustering de Clientes: {x_col} vs {y_col}",
                    labels={x_col: x_col, y_col: y_col, 'cluster': 'Clúster'},
                    opacity=0.7
                )

                count = clustering_data['cluster'].value_counts().sort_index().to_dict()
                dist_str = ", ".join([f"Clúster {i}: {n} registros" for i, n in count.items()])

                centroids = clustering_data.groupby('cluster')[[x_col, y_col]].mean().round(2)
                centroid_text = html.Ul([
                    html.Li(f"Clúster {i}: media({x_col}) = {row[x_col]}, media({y_col}) = {row[y_col]}")
                    for i, row in centroids.iterrows()
                ])

                explanation = html.Div([
                    html.P(f"Distribución de clústeres: {dist_str}.", style={'marginTop': '20px'}),
                    html.H5("📈 Interpretación de centroides (promedios por clúster):"),
                    centroid_text,
                    html.H5("🔍 Recomendaciones estratégicas basadas en clústeres:"),
                    html.Ul([
                        html.Li("🟩 El clúster más frecuente representa el comportamiento dominante. Considera adaptar promociones a este grupo."),
                        html.Li("🟧 Identifica clústeres con baja frecuencia pero alto valor. Podrían representar clientes premium o leales."),
                        html.Li("🟥 Si algún clúster tiene alta tasa de cancelaciones, analiza políticas de retención o comunicación proactiva."),
                        html.Li("📅 Usa la segmentación temporal de clústeres para anticipar estacionalidades y optimizar recursos."),
                    ])
                ])

                return fig, explanation

            except Exception as e:
                return px.scatter(), f"Error al aplicar K-Means: {str(e)}"