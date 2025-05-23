import os
from typing import Tuple
from dash import dcc, html, dash_table, callback, Output, Input, State
import pandas as pd
import base64
import io
import dash
import json
from info_compartida import DATAFRAMES, PROCESS_DATASET

class CargaTab:
    def __init__(self, app):
        # Inicializa la clase con la app de Dash y registra los callbacks
        self.app = app
        self.register_callbacks()

    def render(self):
        # Genera el layout de la pestaña de carga de datos
        uploaded_files_output = PROCESS_DATASET.get('uploaded_files_output', [])

        return html.Div([
            html.H2("Carga de Datos", style={'margin': '20px'}),

            # Almacena los nombres de archivos cargados
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
                children=html.Div(
                    id='output-data-upload', 
                    children=uploaded_files_output
                ),
                style={'margin': '30px'}
            ),
        ])

    def describe_table(self, df) -> dict:
        # Retorna un resumen con nombre, tipo, valores únicos y nulos de cada columna
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
        # Decodifica el contenido del archivo cargado y lo convierte en DataFrame
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in filename or 'js' in filename:
                try:
                    raw_json = json.loads(decoded.decode('utf-8'))
                    if isinstance(raw_json, dict):
                        # Si el JSON es un objeto con claves, se intenta convertir a lista de registros
                        df = pd.json_normalize(raw_json)
                    elif isinstance(raw_json, list):
                        # Si ya es una lista de dicts (registros), se carga directamente
                        df = pd.DataFrame(raw_json)
                    else:
                        return f"Formato de JSON no compatible en el archivo {filename}"
                except Exception as e:
                    return f"Error leyendo el archivo JSON {filename}: {e}"
            else:
                return f"Formato de archivo no soportado: {filename}"

            return df  # ✅ Este return debe ir aquí

        except Exception as e:
            return f"Error procesando el archivo {filename}: {e}"


    def produce_table(self, df, filename) -> Tuple[html.Div]:
        # Crea una tabla resumen de los datos y una vista previa de los primeros registros
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
            # Maneja la carga y visualización de archivos, además ejecuta el ETL automáticamente
            if stored_data is None:
                stored_data = []

            new_data = []
            children = []

            if list_contents and list_filenames:
                for content, name in zip(list_contents, list_filenames):
                    df = self.parse_contents(content, name)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Guardar los datos en memoria compartida y mantener una copia original
                        DATAFRAMES.clear()
                        DATAFRAMES[name] = df
                        DATAFRAMES['raw'] = df

                        print(f"Guardando DataFrame '{name}' en DATAFRAMES")
                        print(f"Tamaño del DataFrame: {df.shape}")

                        if name not in stored_data:
                            new_data.append(name)
                        children.append(self.produce_table(df, name))
                    else:
                        children.append(html.Div([html.P(df)]))

            for item in stored_data:
                if item not in new_data and item in DATAFRAMES:
                    df = DATAFRAMES[item]
                    children.append(self.produce_table(df, item))

            combined_data = list(set(stored_data + new_data))
            PROCESS_DATASET['uploaded_files_output'] = children

            # 🟢 Ejecuta automáticamente ETL al cargar archivo y guarda como 'processed_data'
            if new_data:
                from etl_tab import ETLTab
                etl = ETLTab(None)
                processed_df, _ = etl.process_dataframe(DATAFRAMES[new_data[0]])
                DATAFRAMES['processed_data'] = processed_df
                print("Data procesada y almacenada como 'processed_data'")
                print("Columnas disponibles en processed_data:", processed_df.columns.tolist())
                print("Primeros registros:", processed_df.head())

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
            # Limpia todos los datos y resultados almacenados
            if n_clicks:
                DATAFRAMES.clear()
                if 'uploaded_files_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['uploaded_files_output']
                if 'etl_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['etl_output']
                if 'save_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['save_output']
                if 'data_mining_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['data_mining_output']
                print("Limpiando todos los DataFrames y resultados almacenados")
                return [], []
            return dash.no_update, dash.no_update