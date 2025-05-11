import base64
from dash import dcc, html, dash_table, callback, Output, Input, State
import pandas as pd
import io

class CargaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()
    
    def render(self):
        return html.Div([
            html.H2("Carga de Datos",
                    style={'margin': '20px'}),  
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Arrastra y suelta o ',
                    html.A('Selecciona un archivo'),

                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),
        html.Div(id='output-data-upload'),
    ])

    def describe_table(self, df):
        table_info = []
        for column in df.columns:
            missing_values = int(df[column].isnull().sum())
            column_info = {
                'column_name': column,
                'type': str(df[column].dtype),
                'unique_values': int(df[column].nunique()),
                'missing_values': missing_values,
                'percent_missing': round(missing_values / len(df) * 100, 2),
            }
            table_info.append(column_info)
        return table_info

    def parse_contents(self, contents, filename):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in filename:
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
            else:
                return html.Div([
                    'El archivo no es un csv, xlsx o json',
                ])
        except Exception as e:
            print(e)
            return html.Div([
                'Hubo un error al cargar el archivo',
            ])

        table_info = self.describe_table(df)
        print(table_info)

        return html.Div([
            html.Hr(),
            html.H3(f"Nombre del archivo: {filename}"),
            html.H4(f"Tipo de archivo: {filename.split('.')[-1]}"),
            html.H4(f"Número de registros: {len(df)}"),
            html.H4(f"Número de columnas: {len(df.columns)}"),
            dash_table.DataTable(table_info, 
                                 [{"name": i, "id": i} for i in ['column_name',
                                                                 'type',
                                                                 'unique_values',
                                                                 'missing_values',
                                                                 'percent_missing']],
                                 style_table={'overflowX': 'auto'},
                                 page_size=10,),
            html.Hr(),
            dash_table.DataTable(df.to_dict('records'), 
                                 [{"name": i, "id": i} for i in df.columns],
                                 style_table={'overflowX': 'auto'},
                                 page_size=10,), 
            html.Hr(),
        ], style={"margin": "20px",})

    def register_callbacks(self):
        @self.app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
        )
        def update_output(list_contents, list_filenames):
            if list_contents is not None:
                children = [
                    self.parse_contents(content, name)
                    for content, name in zip(list_contents, list_filenames)
                ]
                return children
            return ""
              