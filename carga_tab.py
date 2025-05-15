import base64
from typing import Tuple
from dash import dcc, html, dash_table, callback, Output, Input, State
import pandas as pd
import io
import time
import dash

DATAFRAMES = {}

class CargaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()
    
    def render(self):
        children = []
        if len(DATAFRAMES) > 0:
            for name, df in DATAFRAMES.items():
                table = self.produce_table(df, name)
                children.append(table)

        return html.Div([
        html.H2("Carga de Datos",
                style={'margin': '20px'}),  
        dcc.Store(id='stored-dataframes', data=[], storage_type='session'),
        html.Button(
            'Limpiar todo',
            id='clear-button',
            style={'margin': '20px'}
        ),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Arrastra y suelta o ',
                html.A('Selecciona un archivo'),
            ]),
            style={
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '30px'
            },
            multiple=True,
        ),
        dcc.Loading(
            id="loading-output",
            type="circle",
            children=html.Div(id='output-data-upload'),
            style={'margin': '30px'}
        ),
    ])

    def describe_table(self, df) -> dict:
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

    def parse_contents(self, contents, filename)-> pd.DataFrame | str:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                return df
            elif 'xlsx' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
                return df
            elif 'json' in filename:
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
                return df
            else:
                return 'Tipo de archivo no soportado'
        except Exception as e:
            print(e)
            return 'Error al procesar el archivo'

    def produce_table(self, df, filename) -> Tuple[html.Div]:
        table_info = self.describe_table(df)
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
            dash_table.DataTable(df.head(10).to_dict('records'), 
                                 [{"name": i, "id": i} for i in df.columns],
                                 style_table={'overflowX': 'auto'},
                                ), 
            html.Hr(),
        ], style={"margin": "20px",})

    def register_callbacks(self):
        # Define un callback de Dash que actualiza dos salidas:
        # 1. El contenido visualizado ('output-data-upload', 'children')
        # 2. Los dataframes almacenados ('stored-dataframes', 'data')
        @self.app.callback(
            Output('output-data-upload', 'children'),
            Output('stored-dataframes', 'data'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('stored-dataframes', 'data'),
        )
        def update_output(list_contents, list_filenames, stored_data):
            # Si no hay dataframes almacenados, inicializa la lista
            if stored_data is None:
                stored_data = []

            new_data = []   # Para guardar los nuevos archivos subidos
            children = []   # Para los elementos visuales a mostrar

            # Si se subieron archivos y hay nombres de archivos
            if list_contents and list_filenames:
                # Procesa cada archivo subido
                for content, name in zip(list_contents, list_filenames):
                    df = self.parse_contents(content, name)  # Convierte el contenido en DataFrame
                    if isinstance(df, pd.DataFrame):
                        # Si el archivo no está ya almacenado, agrégalo a la lista de nuevos y agregalo a DATAFRAMES
                        if not any(item == name for item in stored_data):
                            DATAFRAMES[name] = df
                            new_data.append(name)
                        # Agrega la tabla visual al output
                        children.append(self.produce_table(df, name))
                    else:
                        # Si hubo error al leer, muestra el mensaje de error
                        children.append(html.Div([df]))

                # Para cada archivo previamente almacenado que no esté en los nuevos, recupéralo y muéstralo
                for item in stored_data:
                    if not any(item == new for new in new_data):
                        df = DATAFRAMES[item]
                        children.append(self.produce_table(df, item))

                # Combina los archivos previos y los nuevos para actualizar el almacenamiento
                combined_data = stored_data + new_data
                return children, combined_data
            else:
                # Si no hay archivos nuevos, solo muestra los almacenados
                for item in stored_data:
                    df =  DATAFRAMES.get(item, None)
                    children.append(self.produce_table(df, item))
                return children, stored_data
        
        @self.app.callback(
            Output('stored-dataframes', 'data', allow_duplicate=True),
            Output('output-data-upload', 'children', allow_duplicate=True),
            Input('clear-button', 'n_clicks'),
            prevent_initial_call=True,
        )
        def clear_data(n_clicks):
            # Verifica si el botón de "Limpiar todo" fue presionado al menos una vez
            if n_clicks:
                # Limpia el diccionario global donde se almacenan los DataFrames
                DATAFRAMES.clear()
                # Retorna una lista vacía para:
                # 1. Borrar los nombres de archivos en el componente 'stored-dataframes'
                # 2. Borrar visualmente las tablas del componente 'output-data-upload'
                return [], []
            
            # Si no se ha hecho clic, no actualiza nada (deja los datos y la visualización como están)
            return dash.no_update, dash.no_update
