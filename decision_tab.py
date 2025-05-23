from dash import dcc, html, Input, Output
import dash
import pandas as pd
import plotly.graph_objs as go

from info_compartida import DATAFRAMES

class DecisionTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        if 'processed_data' not in DATAFRAMES:
            return html.Div([
                html.H2("Toma de decisiones", style={'margin': '20px'}),
                html.Div([
                    html.H3("No hay datos procesados disponibles para mostrar."),
                    html.P("Por favor, carga y procesa los datos primero.")
                ], style={'padding': '20px', 'background': '#f8f9fa', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'})
            ])

        df = DATAFRAMES['processed_data']

        hist_fig = go.Figure(data=[
            go.Histogram(x=df['total_of_special_requests'], nbinsx=20, marker_color='#17a2b8')
        ])
        hist_fig.update_layout(
            title="Distribución de predicciones del número de solicitudes especiales",
            xaxis_title="Número de solicitudes especiales",
            yaxis_title="Frecuencia"
        )

        trend_fig = go.Figure()
        if 'arrival_date_week_number' in df.columns:
            trend = df.groupby('arrival_date_week_number')['total_of_special_requests'].mean().reset_index()
            trend_fig = go.Figure(data=[
                go.Scatter(x=trend['arrival_date_week_number'], y=trend['total_of_special_requests'], mode='lines+markers')
            ])
            trend_fig.update_layout(
                title="Promedio de solicitudes especiales por semana",
                xaxis_title="Semana del año",
                yaxis_title="Promedio de solicitudes"
            )

        additional_section = html.Div([
            html.H3("🎯 Objetivo de la Toma de Decisiones"),
            html.P("Ayudar al hotel a optimizar sus recursos y mejorar la experiencia del cliente, anticipando solicitudes especiales y detectando clientes especiales antes de su llegada."),

            html.H3("📊 Visualizaciones clave para la toma de decisiones"),

            html.Div([
                dcc.Graph(figure=hist_fig),
                html.P("Esta gráfica muestra cuántos clientes suelen hacer muchas solicitudes especiales."),
                html.P("Decisión asociada: Si hay picos altos en ciertas épocas, el hotel puede aumentar staff o recursos esos días.")
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'}),

            html.Div([
                dcc.Graph(figure=trend_fig),
                html.P("Esta gráfica muestra la predicción agregada de solicitudes especiales por semana."),
                html.P("Decisión asociada: El hotel puede planear compras y personal según la demanda esperada.")
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'}),

            html.Div([
                html.H3("📌 Recomendaciones adicionales"),
                html.Ul([
                    html.Li("Anticipar la contratación de personal en semanas con alta demanda de solicitudes especiales."),
                    html.Li("Utilizar el modelo para identificar clientes especiales y preparar beneficios personalizados."),
                    html.Li("Monitorear los factores que influyen en la solicitud de servicios para ajustar campañas de marketing."),
                ])
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ffffff', 'border-radius': '8px', 'box-shadow': '0 2px 6px rgba(0,0,0,0.08)'})
        ])

        return html.Div([additional_section, self._render_original_decision_tab(df)])

    def _render_original_decision_tab(self, df):
        client_demading_counts = df['is_demanding_client'].value_counts()
        return html.Div([
            html.H3("Contenido original de las gráficas de decisión"),
            dcc.Graph(
                id='original-pie-chart',
                figure={
                    'data': [
                        {
                            'labels': client_demading_counts.index,
                            'values': client_demading_counts.values,
                            'type': 'pie',
                            'name': 'Clientes demandantes'
                        }
                    ],
                    'layout': {
                        'title': 'Clientes que hacen peticiones especiales'
                    }
                }
            )
        ])

    def register_callbacks(self):
        pass
