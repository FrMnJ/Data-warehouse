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
            html.H2("Análisis Exploratorio", style={'margin': '20px'}),

            html.Div([
                html.Label("Selecciona una columna:"),
                dcc.Dropdown(id='eda-column-dropdown', placeholder="Selecciona una columna"),
            ], style={'margin': '20px', 'width': '40%'}),

            html.Div(id='eda-stats-output', style={'margin': '20px'}),

            html.Div([
                dcc.Graph(id='histogram-graph'),
                dcc.Graph(id='boxplot-graph')
            ], style={'margin': '20px'}),

            html.Div(id='eda-explanation', style={'margin': '20px', 'fontStyle': 'italic', 'color': '#333'}),
        ])

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
            if col_data.empty:
                return html.P("No hay datos disponibles para la columna seleccionada."), px.scatter(), px.box(), ""
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

                stats_table = html.Table([
                    html.Thead(html.Tr([html.Th("Estadística"), html.Th("Valor")]))
                ] + [
                    html.Tr([html.Td(translation.get(str(index), str(index))), html.Td(str(value))])
                    for index, value in desc.items()
                ], style={"width": "300px", "borderCollapse": "collapse", "border": "1px solid #ccc"})

                fig_hist = px.histogram(df, x=column, nbins=30, title=f"Histograma de {column}", color_discrete_sequence=['#636efa'])
                fig_box = px.box(df, y=column, title=f"Boxplot de {column}", color_discrete_sequence=['#636efa'])

                explanation = html.Div([
                    html.H4("📊 Explicación automática del análisis:", style={'marginTop': '30px'}),
                    html.P(f"La media de '{column}' es {desc['mean']}, y su mediana es {desc['50%']}. Esto sugiere que la distribución es "
                        f"{'simétrica' if abs(desc['mean'] - desc['50%']) < desc['std'] * 0.1 else 'asimétrica'}."),

                    html.P(f"El rango intercuartil (IQR) va de {desc['25%']} a {desc['75%']}, indicando una dispersión "
                        f"{'moderada' if desc['75%'] - desc['25%'] < desc['std'] else 'amplia'} de los valores."),

                    html.P(f"Se detectaron valores mínimos y máximos de {desc['min']} a {desc['max']}, "
                        f"{'con presencia de posibles outliers.' if (desc['min'] < desc['25%'] - 1.5 * (desc['75%'] - desc['25%']) or desc['max'] > desc['75%'] + 1.5 * (desc['75%'] - desc['25%'])) else 'sin valores atípicos destacados.'}")
                ])
                return stats_table, fig_hist, fig_box, explanation
            else:
                value_counts = col_data.value_counts().reset_index()
                value_counts.columns = [column, 'count']

                fig = px.bar(value_counts, x=column, y='count', title=f"Frecuencia de {column}", color_discrete_sequence=['#636efa'])

                table = html.Table([
                    html.Thead(html.Tr([html.Th(column), html.Th("Frecuencia")])),
                    html.Tbody([html.Tr([html.Td(row[column]), html.Td(row['count'])]) for _, row in value_counts.iterrows()])
                ], style={"borderCollapse": "collapse", "border": "1px solid #ccc"})

                explanation = html.Div([
                    html.H4("📊 Frecuencia de categorías:", style={'marginTop': '30px'}),
                    html.P(f"Se detectaron {value_counts.shape[0]} valores únicos para la columna '{column}'."),
                    html.P("El gráfico muestra la distribución de frecuencias.")
                ])

                return table, fig, px.box(), explanation