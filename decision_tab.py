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
                html.H2("Toma de decisiones",
                        style={'margin': '20px'}),
                html.Div([
                    html.H3("No hay datos procesados disponibles para mostrar."),
                    html.P("Por favor, carga y procesa los datos primero.")
                ], style={'padding': '20px', 'background': '#f8f9fa', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'})
            ])
        else:
            df = DATAFRAMES['processed_data']
            client_demading_counts = df['is_demanding_client'].value_counts()
            hotel_demands = df.groupby('hotel')['total_of_special_requests'].sum().reset_index()
            market_segment_demands = df.groupby('market_segment')['total_of_special_requests'].sum().reset_index()
            distribution_channel_demands = df.groupby('distribution_channel')['total_of_special_requests'].sum().reset_index()
            reserved_room_type_demands = df.groupby('reserved_room_type')['total_of_special_requests'].sum().reset_index()
            assigned_room_type_demands = df.groupby('assigned_room_type')['total_of_special_requests'].sum().reset_index()
            deposit_type_demands = df.groupby('deposit_type')['total_of_special_requests'].sum().reset_index()
            customer_type_demands = df.groupby('customer_type')['total_of_special_requests'].sum().reset_index()
            special_counts = df['total_of_special_requests'].apply(lambda x: str(x) if x < 3 else '3+').value_counts().sort_index()
            # Grafica de frecuencia de peticiones especiales por hotel
        return html.Div([
            html.H2("Toma de decisiones",
                    style={'margin': '20px'}),  
            # Grafica de pastel de clientes que hace petciones especiales
            html.H3("🎯 Objetivo de la Toma de Decisiones"),
            html.P("Ayudar al hotel a optimizar sus recursos y mejorar la experiencia del cliente, anticipando solicitudes especiales y detectando clientes especiales antes de su llegada."),
            html.P("Las peticiones especiales son un indicador de las necesidades de los clientes y pueden influir en la satisfacción del cliente."),
            html.H3("📊 Visualizaciones clave para la toma de decisiones"),
            html.Div([
                dcc.Graph(
                    id='special-requests-pie-chart',
                    figure={
                        'data': [
                            {
                                'labels': client_demading_counts.index,
                                'values': client_demading_counts.values,
                                'type': 'pie',
                                'name': 'Clientes demandantes',
                                'hoverinfo': 'label+percent+name',
                                'textinfo': 'percent',
                                'textfont_size': 20,
                                'marker': {'line': {'width': 2, 'color': '#ffffff'}}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Clientes que hacen peticiones especiales",
                                'font_size': 24
                            },
                            'showlegend': True,
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra la proporción de clientes que hacen peticiones especiales en comparación con aquellos que no lo hacen.", style={'margin-top': '10px'}),
                html.P("Esta información resulta útil para entender la magnitud del fenómeno y su impacto en la operación del hotel."),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '5px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de pastel de frecuencias de peticiones especiales
            html.Div([
                dcc.Graph(
                    id='requests-distribution-pie',
                    figure={
                        'data': [
                            {
                                'labels': special_counts.index,
                                'values': special_counts.values,
                                'type': 'pie',
                                'name': 'Distribución de peticiones',
                                'hoverinfo': 'label+percent+name',
                                'textinfo': 'percent',
                                'textfont_size': 20,
                                'marker': {'line': {'width': 2, 'color': '#ffffff'}}
                            }
                        ],
                        'layout': {
                        'title': {
                            'text': "Distribución de peticiones especiales por cliente (0, 1, 2, 3+)",
                            'font_size': 24
                        },
                        'showlegend': True,
                        'height': 400,
                        'width': 600
                    }
                }
            ),
                html.P("Esta gráfica muestra la distribución de peticiones especiales por cliente, categorizadas en 0, 1, 2 y 3+ o más peticiones especiales.", style={'margin-top': '10px'}),
                html.P("Esta información es útil para entender la frecuencia de las peticiones especiales y su impacto en la operación del hotel."),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '5px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0', 'margin-top': '20px'} ),
            # Grafica de peticiones especiales por hotel
            html.Div([
                dcc.Graph(
                    id='special-requests-hotel-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': hotel_demands['hotel'],
                                'y': hotel_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por hotel',
                                'marker': {'color': '#007bff'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por hotel",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Hotel'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por hotel."),
                html.P("Esta información es útil para identificar qué hotel tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
                html.P("En 'City Hotel' requiere más atención en comparación de Resort Hotel.", style={'margin-top': '10px'}),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de peticiones especiales por segmento de mercado
            html.Div([
                dcc.Graph(
                    id='special-requests-market-segment-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': market_segment_demands['market_segment'],
                                'y': market_segment_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por segmento de mercado',
                                'marker': {'color': '#28a745'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por segmento de mercado",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Segmento de mercado'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por segmento de mercado."),
                html.P("Esta información es útil para identificar qué segmento de mercado tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de peticiones especiales por canal de distribución
            html.Div([
                dcc.Graph(
                    id='special-requests-distribution-channel-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': distribution_channel_demands['distribution_channel'],
                                'y': distribution_channel_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por canal de distribución',
                                'marker': {'color': '#dc3545'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por canal de distribución",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Canal de distribución'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por canal de distribución."),
                html.P("Esta información es útil para identificar qué canal de distribución tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de peticiones especiales por tipo de habitación reservada
            html.Div([
                dcc.Graph(
                    id='special-requests-reserved-room-type-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': reserved_room_type_demands['reserved_room_type'],
                                'y': reserved_room_type_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por tipo de habitación reservada',
                                'marker': {'color': '#ffc107'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por tipo de habitación reservada",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Tipo de habitación reservada'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por tipo de habitación reservada."),
                html.P("Esta información es útil para identificar qué tipo de habitación reservada tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de peticiones especiales por tipo de habitación asignada
            html.Div([
                dcc.Graph(
                    id='special-requests-assigned-room-type-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': assigned_room_type_demands['assigned_room_type'],
                                'y': assigned_room_type_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por tipo de habitación asignada',
                                'marker': {'color': '#17a2b8'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por tipo de habitación asignada",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Tipo de habitación asignada'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por tipo de habitación asignada."),
                html.P("Esta información es útil para identificar qué tipo de habitación asignada tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de peticiones especiales por tipo de depósito
            html.Div([
                dcc.Graph(
                    id='special-requests-deposit-type-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': deposit_type_demands['deposit_type'],
                                'y': deposit_type_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por tipo de depósito',
                                'marker': {'color': '#6f42c1'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por tipo de depósito",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Tipo de depósito'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }
                ),
                html.P("Esta gráfica muestra el total de peticiones especiales por tipo de depósito."),
                html.P("Esta información es útil para identificar qué tipo de depósito tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Tipo de cliente
            html.Div([
                dcc.Graph(
                    id='special-requests-customer-type-bar-chart',
                    figure={
                        'data': [
                            {
                                'x': customer_type_demands['customer_type'],
                                'y': customer_type_demands['total_of_special_requests'],
                                'type': 'bar',
                                'name': 'Peticiones especiales por tipo de cliente',
                                'marker': {'color': '#20c997'}
                            }
                        ],
                        'layout': {
                            'title': {
                                'text': "Peticiones especiales por tipo de cliente",
                                'font_size': 24
                            },
                            'xaxis': {'title': 'Tipo de cliente'},
                            'yaxis': {'title': 'Total de peticiones especiales'},
                            'height': 400,
                            'width': 600
                        }
                    }),
                html.P("Esta gráfica muestra el total de peticiones especiales por tipo de cliente."),
                html.P("Esta información es útil para identificar qué tipo de cliente tiene más peticiones especiales y, por lo tanto, podría requerir más atención o recursos."),
            ], style={'margin-top': '20px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),  
            html.Div([
                html.H3("Grafica de peticiones especiales por día"),
                html.Label("Selecciona una fecha de inicio:", style={'margin-right': '10px', 'font-weight': 'bold'}),
                dcc.DatePickerSingle(
                    id='start-date-special', 
                    display_format='YYYY-MM-DD', 
                    date='2016-01-01',
                    style={'margin-bottom': '10px', 'margin-right': '30px', 'background': '#f0f0f0', 'border-radius': '5px', 'padding': '5px'}
                ),     
                html.Label("Selecciona una fecha de fin:", style={'margin-right': '10px', 'font-weight': 'bold'}),
                dcc.DatePickerSingle(
                    id='end-date-special', 
                    display_format='YYYY-MM-DD', 
                    date='2016-12-31',
                    style={'margin-bottom': '10px', 'background': '#f0f0f0', 'border-radius': '5px', 'padding': '5px'}
                ),
                dcc.Graph(id='special-requests-graph', style={'height': '400px', 'width': '100%', 'margin-top': '20px'}), 
                html.P("Esta gráfica muestra el total de peticiones especiales por día en el rango de fechas seleccionado.", style={'margin-top': '10px'}),
                html.P("Esta información es útil para identificar patrones y tendencias en las peticiones especiales a lo largo del tiempo.", style={'margin-top': '10px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '5px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0', 'margin-top': '15px'} ),
            html.Div([
                html.H3("Justificación del modelo utilizado"),
                html.P("Se utilizó un árbol de decisión por su facilidad de interpretación y por permitir visualizar reglas claras para identificar clientes exigentes."),
                html.P("Se priorizó una configuración del modelo que reduce la complejidad (poca profundidad y muestras mínimas elevadas) para facilitar la toma de decisiones basada en las reglas generadas."),
                html.P("También se utilizó un modelo de Regresión Lineal para cuantificar el número de peticiones especiales esperadas en función de las características del cliente."),
                html.P("Ambos modelos permitirán al hotel anticiparse a las necesidades de los clientes y mejorar la experiencia del cliente."),
            ]),
            html.Div([
                html.H3("Impacto de los clientes exigentes en el negocio"),
                html.P("Detectar correctamente a los clientes exigentes permite al hotel anticiparse a sus necesidades y mejorar la satisfacción del cliente."),
                html.P("Un error de tipo falso negativo (no detectar un cliente exigente) puede llevar a una mala experiencia, lo cual impacta en la reputación del hotel, reduce la posibilidad de recomendaciones y afecta el retorno del cliente."),
                html.P("Es importante maximizar el 'recall' de la clase 'exigente', aunque ello implique tener más falsos positivos.")
            ]),
            html.Div([
                html.H3("Recomendaciones basadas en los hallazgos"),
                html.Ul([
                    html.Li("Ajustar el personal en hoteles con más peticiones especiales."),
                    html.Li("Asignar personal adicional en fechas con alta cantidad de peticiones especiales."),
                    html.Li("Aplicar medidas de seguimiento especial a los clientes identificados como potencialmente exigentes."),
                    html.Li("Ajustar la oferta de servicios y recursos en función de los segmentos de mercado con más peticiones especiales."),
                ])
            ])
        ], style={'padding': '20px', 'margin': '20px'})
    
    def register_callbacks(self):
        @self.app.callback(
            Output('special-requests-graph', 'figure'),
            Input('start-date-special', 'date'),
            Input('end-date-special', 'date')
        )
        def update_special_requests_graph(start_date, end_date):
            if not start_date or not end_date:
                return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "Selecciona un rango de fechas", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]})
            if 'processed_data' not in DATAFRAMES:
                return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No hay datos procesados", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]})
            df = DATAFRAMES['processed_data']
            if not all(col in df.columns for col in ['reservation_status_date', 'total_of_special_requests']):
                return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "Faltan columnas necesarias en los datos", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]})
            mask = (df['reservation_status_date'] >= start_date) & (df['reservation_status_date'] <= end_date)
            filtered = df[mask]
            if filtered.empty:
                return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No hay datos para el rango seleccionado", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]})
            grouped = filtered.groupby(pd.Grouper(key='reservation_status_date', freq='D'))['total_of_special_requests'].sum().reset_index()
            if grouped.empty:
                return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No hay datos agrupados para mostrar", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]})
            fig = go.Figure([
                go.Bar(
                    x=grouped['reservation_status_date'].dt.strftime('%Y-%m-%d'),
                    y=grouped['total_of_special_requests'],
                    name='Total de peticiones especiales',
                )
            ])
            fig.update_layout(
                title='Total de peticiones especiales por día',
                xaxis_title='Día',
                yaxis_title='Total de peticiones especiales',
                bargap=0.2
            )
            return fig