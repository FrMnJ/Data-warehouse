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
            cake = dcc.Graph(
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
            )

            return html.Div([
            html.H2("Toma de decisiones",
                    style={'margin': '20px'}),  
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
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '5px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} ),
            # Grafica de pastel de clientes que hace petciones especiales
            html.Div([
                html.H3("Grafica de clientes que hacen peticiones especiales"),
                cake,
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '5px', 'padding': '20px', 'background': '#fafbfc', 'border-radius': '8px', 'box-shadow': '0 2px 8px #e0e0e0'} )
        ])
    
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