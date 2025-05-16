from typing import Tuple
from dash import dcc, html, dash_table, callback, Output, Input, State
import pandas as pd
import base64
import io
import dash
from info_compartida import DATAFRAMES, PROCESS_DATASET

class CargaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()
    
    def render(self):
        # Verificar si hay resultados almacenados para mostrar
        uploaded_files_output = PROCESS_DATASET.get('uploaded_files_output', [])
        
        return html.Div([
            html.H2("Carga de Datos", style={'margin': '20px'}),  
            
            # Mantenemos el dcc.Store para mantener compatibilidad
            dcc.Store(id='stored-dataframes', data=list(DATAFRAMES.keys()), storage_type='session'),
            
            html.Button(
                'Limpiar todo',
                id='clear-button',
                style={'margin': '20px',
                        'backgroundColor': "#606060",
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'padding': '10px 20px',
                        'cursor': 'pointer',
                        'fontSize': '14px'}
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
                    'margin': '30px',
                    'cursor': 'pointer',
                },
                multiple=True,
            ),
            
            dcc.Loading(
                id="loading-output",
                type="circle",
                # Mostramos los resultados almacenados si existen
                children=html.Div(
                    id='output-data-upload', 
                    children=uploaded_files_output
                ),
                style={'margin': '30px'}
            ),
        ])

    def describe_table(self, df) -> dict:
        table_info = []
        for column in df.columns:
            column_type = str(df[column].dtype)
            unique_values = df[column].nunique()
            missing_values = df[column].isna().sum()
            percent_missing = round((missing_values / len(df)) * 100, 2)
            
            table_info.append({
                'column_name': column,
                'type': column_type,
                'unique_values': unique_values,
                'missing_values': missing_values,
                'percent_missing': f"{percent_missing}%"
            })
            
        return table_info

    def parse_contents(self, contents, filename) -> pd.DataFrame | str:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in filename or 'js' in filename:
                df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
            else:
                return f"El formato del archivo {filename} no es compatible"
                
            return df
        except Exception as e:
            return f"Hubo un error procesando el archivo {filename}: {e}"

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
                    df = self.parse_contents(content, name) # Convierte el contenido en DataFrame
                    if isinstance(df, pd.DataFrame):
                        # Guardar en el DATAFRAMES compartido
                        DATAFRAMES[name] = df
                        print(f"Guardando DataFrame '{name}' en DATAFRAMES")
                        print(f"Tamaño del DataFrame: {df.shape}")
                        
                        if name not in stored_data:
                            new_data.append(name)
                        children.append(self.produce_table(df, name))
                    else:
                        children.append(html.Div([html.P(df)]))

            # Para cada archivo previamente almacenado que no esté en los nuevos, recupéralo y muéstralo
            for item in stored_data:
                if item not in new_data and item in DATAFRAMES:
                    df = DATAFRAMES[item]
                    children.append(self.produce_table(df, item))

            # Combina los archivos previos y los nuevos para actualizar el almacenamiento
            combined_data = list(set(stored_data + new_data))
            
            # Almacena la visualización en el diccionario compartido para persistencia
            PROCESS_DATASET['uploaded_files_output'] = children
            
            # Imprimir información de diagnóstico
            print(f"DATAFRAMES contiene: {list(DATAFRAMES.keys())}")
            print(f"Número total de DataFrames: {len(DATAFRAMES)}")
            
            return children, combined_data

        @self.app.callback(
            Output('stored-dataframes', 'data', allow_duplicate=True),
            Output('output-data-upload', 'children', allow_duplicate=True),
            Input('clear-button', 'n_clicks'),
            prevent_initial_call=True,
        )
        def clear_data(n_clicks):
            if n_clicks:
                # Limpiar los datos globales
                DATAFRAMES.clear()
                
                # Limpiar también la visualización almacenada
                if 'uploaded_files_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['uploaded_files_output']
                
                # Limpiar otros datos relacionados en PROCESS_DATASET si es necesario
                if 'etl_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['etl_output']
                if 'save_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['save_output']
                if 'mineria_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['mineria_output']
                
                print("Limpiando todos los DataFrames y resultados almacenados")
                return [], []
            
            return dash.no_update, dash.no_update
