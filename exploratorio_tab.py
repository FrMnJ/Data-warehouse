from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from info_compartida import DATAFRAMES

class ExploratorioTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        return html.Div([
            html.H2("Análisis Exploratorio", style={
                'margin': '20px',
                'color': '#333',
                'fontWeight': 'bold',
                'letterSpacing': '1px'
            }),

            html.Div([
                html.Label("Selecciona una columna:", style={
                    'fontWeight': 'bold',
                    'fontSize': '17px',
                    'marginBottom': '8px',
                    'display': 'block',
                    'letterSpacing': '0.5px'
                }),
                dcc.Dropdown(
                    id='eda-column-dropdown',
                    placeholder="Selecciona una columna",
                    style={
                        'width': '100%',
                        'fontSize': '16px',
                        'borderRadius': '6px',
                        'border': '1px solid #bdbdbd',
                        'padding': '4px 6px',         
                        'minHeight': '32px',
                        'backgroundColor': '#f8f9fa',
                        'color': '#222',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.04)'
                    }
                ),
            ], style={
                'margin': '30px 20px 20px 20px',
                'width': '40%',
                'minWidth': '260px'
            }),

            html.Div(id='eda-stats-output', style={'margin': '20px'}),

            html.Div(id='eda-explanation', style={'margin': '20px', 'fontStyle': 'italic', 'color': '#333'}),

            html.Div([
                html.Div([
                    dcc.Graph(id='histogram-graph', style={'height': '350px', 'width': '100%'})
                ], style={'flex': '1', 'marginRight': '10px'}),
                html.Div([
                    dcc.Graph(id='boxplot-graph', style={'height': '350px', 'width': '100%'})
                ], style={'flex': '1', 'marginLeft': '10px'}),
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'margin': '20px 0 30px 0',
                'gap': '20px',
                'width': '100%'
            }),
            
        ], style={'maxWidth': '1100px', 'margin': 'auto', 'fontFamily': 'Segoe UI, Arial, sans-serif'})

    def register_callbacks(self):
        @self.app.callback(
            Output('eda-column-dropdown', 'options'),
            Output('eda-column-dropdown', 'value'),
            Input('tabs', 'value')  # Trigger when the tab changes
        )
        def update_dropdown_options(tab_value):
            if tab_value != 'exploratorio' or 'processed_data' not in DATAFRAMES:
                return [], None
            df = DATAFRAMES['processed_data']
            all_columns = df.columns
            options = [{'label': col, 'value': col} for col in all_columns]
            return options, None

        @self.app.callback(
            Output('eda-stats-output', 'children'),
            Output('histogram-graph', 'figure'),
            Output('boxplot-graph', 'figure'),
            Output('eda-explanation', 'children'),
            Input('eda-column-dropdown', 'value')
        )
        def update_graphs(column):
            if not column or 'processed_data' not in DATAFRAMES:
                raise PreventUpdate

            df = DATAFRAMES['processed_data']
            if df.empty or column not in df.columns:
                return html.P("No hay datos disponibles para la columna seleccionada."), px.scatter(), px.box(), ""

            col_data = df[column].dropna()
            if pd.api.types.is_numeric_dtype(col_data):
                desc = col_data.describe().round(2)
                translation = {
                    "count": "Cantidad",
                    "mean": "Media",
                    "std": "Desviación estándar",
                    "min": "Mínimo",
                    "25%": "Percentil 25",
                    "50%": "Mediana",
                    "75%": "Percentil 75",
                    "max": "Máximo"
                }

                explanation = html.Div([
                    html.H3("📊 Explicación automática del análisis:"),
                    html.P(f"La media de '{column}' es {desc['mean']}, y su mediana es {desc['50%']}. Esto sugiere que la distribución es "
                        f"{'simétrica' if abs(desc['mean'] - desc['50%']) < desc['std'] * 0.1 else 'asimétrica'}."),

                    html.P(f"El rango intercuartil (IQR) va de {desc['25%']} a {desc['75%']}, indicando una dispersión "
                        f"{'moderada' if desc['75%'] - desc['25%'] < desc['std'] else 'amplia'} de los valores."),

                    html.P(f"Se detectaron valores mínimos y máximos de {desc['min']} a {desc['max']}, "
                        f"{'con presencia de posibles outliers.' if (desc['min'] < desc['25%'] - 1.5 * (desc['75%'] - desc['25%']) or desc['max'] > desc['75%'] + 1.5 * (desc['75%'] - desc['25%'])) else 'sin valores atípicos destacados.'}")
                ], style={'margin': '30px', 'fontSize': '16px', 'color': '#333'})

                stats_table = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Estadística", style={'backgroundColor': '#e3e3e3', 'fontWeight': 'bold', 'fontSize': '16px', 'textAlign': 'center', 'padding': '8px'}),
                        html.Th("Valor", style={'backgroundColor': '#e3e3e3', 'fontWeight': 'bold', 'fontSize': '16px', 'textAlign': 'center', 'padding': '8px'})
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(translation.get(str(index), str(index)), style={'textAlign': 'center', 'padding': '8px', 'fontSize': '15px', 'backgroundColor': '#fcfcfc' if i % 2 == 0 else '#f5f7fa'}),
                        html.Td(str(value), style={'textAlign': 'center', 'padding': '8px', 'fontSize': '15px', 'backgroundColor': '#fcfcfc' if i % 2 == 0 else '#f5f7fa'})
                    ]) for i, (index, value) in enumerate(desc.items())
                ])
            ], style={
                "width": "340px",
                "margin": "20px auto",
                "borderCollapse": "collapse",
                "border": "1px solid #ccc",
                "borderRadius": "8px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
                "fontFamily": "Segoe UI, Arial, sans-serif"
            })

                fig_hist = px.histogram(df, x=column, nbins=30, title=f"Histograma de {column}", color_discrete_sequence=['#636efa'])
                fig_box = px.box(df, y=column, title=f"Boxplot de {column}", color_discrete_sequence=['#636efa'])

                
                return stats_table, fig_hist, fig_box, explanation
            else:
                value_counts = col_data.value_counts().reset_index()
                value_counts.columns = [column, 'count']

                explanation = html.Div([
                    html.H3("📊 Frecuencia de categorías:"),
                    html.P(f"Se detectaron {value_counts.shape[0]} valores únicos para la columna '{column}'."),
                    html.P("El gráfico muestra la distribución de frecuencias.")
                ], style={'margin': '30px', 'fontSize': '16px', 'color': '#333'})

                fig = px.bar(value_counts, x=column, y='count', title=f"Frecuencia de {column}", color_discrete_sequence=['#636efa'])

                table = html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th(column, style={'backgroundColor': '#e3e3e3', 'fontWeight': 'bold', 'fontSize': '16px', 'textAlign': 'center', 'padding': '8px'}),
                            html.Th("Frecuencia", style={'backgroundColor': '#e3e3e3', 'fontWeight': 'bold', 'fontSize': '16px', 'textAlign': 'center', 'padding': '8px'})
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(row[column], style={'textAlign': 'center', 'padding': '8px', 'fontSize': '15px', 'backgroundColor': '#fcfcfc' if i % 2 == 0 else '#f5f7fa'}),
                            html.Td(row['count'], style={'textAlign': 'center', 'padding': '8px', 'fontSize': '15px', 'backgroundColor': '#fcfcfc' if i % 2 == 0 else '#f5f7fa'})
                        ]) for i, (_, row) in enumerate(value_counts.iterrows())
                    ])
                ], style={
                    "borderCollapse": "collapse",
                    "border": "1px solid #ccc",
                    "margin": "20px auto",
                    "width": "60%",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.07)",
                    "fontFamily": "Segoe UI, Arial, sans-serif"
                })

                return table, fig, px.box(), explanation