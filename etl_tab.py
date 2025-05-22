from typing import Tuple
import os
import pandas as pd
import dash
from dash import Input, dcc, html, callback, State, dash_table, Output
from info_compartida import DATAFRAMES, PROCESS_DATASET
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
import json
import io
from io import StringIO


class ETLTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        # Verificar si hay resultados almacenados para mostrar
        etl_output = PROCESS_DATASET.get('etl_output', [])
        save_output = PROCESS_DATASET.get('save_output', html.Div())
        export_options_div = PROCESS_DATASET.get('export_options_div', {'display': 'none'})
        
        return html.Div([
            html.H2("ETL", style={'margin': '20px'}),
            
            # Botón de inicio del proceso ETL (solo procesamiento)
            html.Button(
                'Iniciar Proceso ETL',
                id='start-etl-button',
                style={'margin': '20px',
                    'backgroundColor': "#007bff",
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'padding': '10px 20px',
                    'cursor': 'pointer',
                    'fontSize': '14px'}
            ),
            
            # Botón para limpiar resultados del ETL
            html.Button(
                'Limpiar resultados',
                id='clear-etl-button',
                style={'margin': '20px',
                    'backgroundColor': "#606060",
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'padding': '10px 20px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'marginLeft': '10px'}
            ),
            
            # Sección para opciones de descarga DESPUÉS del ETL (inicialmente oculta)
            html.Div([
                html.H3("Opciones de descarga", style={'marginTop': '20px', 'marginBottom': '15px'}),
                html.P("Seleccione el formato y presione 'Descargar' para obtener los datos procesados"),
                html.Div([
                    html.Label("Formato de exportación:"),
                    dcc.RadioItems(
                        id='export-format',
                        options=[
                            {'label': 'CSV (.csv)', 'value': 'csv'},
                            {'label': 'Excel (.xlsx)', 'value': 'xlsx'},
                            {'label': 'JSON (.json)', 'value': 'json'},
                            {'label': 'Base de datos (PostgreSQL)', 'value': 'postgres'}
                        ],
                        value='csv',
                        style={'marginBottom': '15px'},
                        labelStyle={'marginRight': '15px', 'display': 'inline-block'}
                    ),
                ]),
                # Campos para PostgreSQL que se mostrarán condicionalmente
                html.Div(id='postgres-config', children=[
                    html.Div([
                        html.Label("Host:"),
                        dcc.Input(id='postgres-host', type='text', value='localhost', 
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                        html.Label("Puerto:"),
                        dcc.Input(id='postgres-port', type='text', value='5432',
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                    ]),
                    html.Div([
                        html.Label("Base de datos:"),
                        dcc.Input(id='postgres-db', type='text', value='postgres', 
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                        html.Label("Esquema:"),
                        dcc.Input(id='postgres-schema', type='text', value='public',
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                    ]),
                    html.Div([
                        html.Label("Tabla:"),
                        dcc.Input(id='postgres-table', type='text', value='hotel_bookings',
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                    ]),
                    html.Div([
                        html.Label("Usuario:"),
                        dcc.Input(id='postgres-user', type='text', value='postgres',
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                        html.Label("Contraseña:"),
                        dcc.Input(id='postgres-password', type='password',
                                style={'marginRight': '15px', 'marginBottom': '10px'}),
                    ]),
                ], style={'display': 'none', 'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'marginTop': '10px'}),
                # Campo para nombre de archivo (para formatos de archivo)
                html.Div(id='file-config', children=[
                    html.Label("Nombre del archivo:"),
                    dcc.Input(
                        id='filename-input',
                        type='text',
                        value='hotel_bookings_processed',
                        style={'width': '400px', 'marginBottom': '15px'}
                    ),
                    html.P("Solo especifique el nombre sin extensión. La extensión se añadirá automáticamente según el formato seleccionado.", 
                        style={'fontSize': '12px', 'color': '#666', 'margin': '0 0 15px 0'}),
                ]),
                
                # Botón para descargar después del ETL
                html.Button(
                    'Descargar',
                    id='download-etl-button',
                    style={'margin': '10px 0',
                        'backgroundColor': "#28a745",
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'padding': '10px 20px',
                        'cursor': 'pointer',
                        'fontSize': '14px'}
                ),
                
                # Componente de descarga (invisible, pero necesario)
                dcc.Download(id="download-data"),
                
                # Botón para guardar en PostgreSQL (separado ya que no implica descarga)
                html.Button(
                    'Guardar en PostgreSQL',
                    id='save-postgres-button',
                    style={'margin': '10px 0 10px 10px',
                        'backgroundColor': "#17a2b8",
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'padding': '10px 20px',
                        'cursor': 'pointer',
                        'fontSize': '14px',
                        'display': 'none'}, # Inicialmente oculto
                    n_clicks=0
                ),
                
                # Área para mostrar resultados del guardado/descarga
                html.Div(id='save-output', children=save_output, 
                        style={'marginTop': '15px', 'padding': '15px', 'borderTop': '1px solid #ddd'}),
            ], 
            id='export-options-div',
            style=export_options_div),
            
            # Separador visual
            html.Hr(style={'margin': '30px 20px', 'borderTop': '1px solid #eee'}),
            
            # Indicador de carga y resultados del ETL - mostrar resultados almacenados si existen
            html.Div([
                html.H3("Resultados del proceso ETL", 
                        style={'margin': '20px 0', 'display': 'none' if not etl_output else 'block'}),
                dcc.Loading(
                    id="loading-etl",
                    type="circle",
                    children=html.Div(id='etl-output',
                                    children=etl_output if etl_output else [html.P("", style={'margin': '20px'})]),
                    style={'margin': '20px'}
                )
            ], id='etl-results-container'),
        ])

    def register_callbacks(self):
        @self.app.callback(
            Output('etl-output', 'children'),
            Output('export-options-div', 'style'),
            Input('start-etl-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def run_etl_process(n_clicks):
            if n_clicks is None:
                return [html.P("")], {'display': 'none'}
            
            # Verificar que haya datos cargados
            if not DATAFRAMES:
                return [html.P("No hay datos cargados. Por favor, cargue al menos un conjunto de datos en la pestaña 'Carga de datos'.", 
                            style={'color': 'red', 'fontWeight': 'bold', 'margin': '20px'})], {'display': 'none'}
            
            # Procesar los DataFrames disponibles
            dfs_to_concat = []
            print(f"Procesando {len(DATAFRAMES)} DataFrames")
            for name, df in DATAFRAMES.items():
                if isinstance(df, pd.DataFrame):
                    dfs_to_concat.append(df)
            
            if not dfs_to_concat:
                return [html.P("No se encontraron DataFrames válidos para procesar.", 
                            style={'color': 'red', 'fontWeight': 'bold', 'margin': '20px'})], {'display': 'none'}
            
            try:
                print(f"Concatenando {len(dfs_to_concat)} DataFrames")
                full_df = pd.concat(dfs_to_concat, ignore_index=True)
                print(f"DataFrame concatenado con forma: {full_df.shape}")
                processed_df, etl_output = self.process_dataframe(full_df)
                
                # Almacenar resultados en estado compartido
                print(f"Almacenando DataFrame procesado con forma: {processed_df.shape}")
                DATAFRAMES['processed_data'] = processed_df
                PROCESS_DATASET['etl_output'] = etl_output
                
                # Crear resumen para mostrar
                summary = html.Div([
                    html.H3("✅ Procesamiento ETL completado", style={'color': 'green', 'fontWeight': 'bold'}),
                    html.P(f"Se procesaron {len(full_df)} registros y se obtuvieron {len(processed_df)} registros limpios."),
                    html.P(f"Se concatenaron {len(dfs_to_concat)} conjuntos de datos."),
                    html.Hr()
                ])
                
                # Mostrar las opciones de exportación después del ETL
                export_options_style = {
                    'display': 'block',
                    'margin': '20px',
                    'padding': '20px',
                    'backgroundColor': '#f9f9f9',
                    'borderRadius': '5px',
                    'border': '1px solid #28a745'
                }
                PROCESS_DATASET['export_options_div'] = export_options_style
                
                return [summary] + etl_output, export_options_style
                    
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return [html.Div([
                    html.H3("❌ Error durante el procesamiento ETL", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P(f"Detalle: {str(e)}")
                ])], {'display': 'none'}

        # Callback para mostrar/ocultar el botón de PostgreSQL
        @self.app.callback(
            [Output('save-postgres-button', 'style'),
            Output('download-etl-button', 'style')],  # Añadir esta salida
            Input('export-format', 'value')
        )
        def toggle_format_buttons(export_format):
            # Estilo base para el botón de PostgreSQL
            postgres_button_style = {
                'margin': '10px 0 10px 10px',
                'backgroundColor': "#17a2b8",
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'fontSize': '14px'
            }
            
            # Estilo base para el botón de descarga
            download_button_style = {
                'margin': '10px 0',
                'backgroundColor': "#28a745",
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'fontSize': '14px'
            }
            
            if export_format == 'postgres':
                # Mostrar botón de PostgreSQL, ocultar botón de descarga
                postgres_button_style['display'] = 'inline-block'
                download_button_style['display'] = 'none'
            else:
                # Ocultar botón de PostgreSQL, mostrar botón de descarga
                postgres_button_style['display'] = 'none'
                download_button_style['display'] = 'inline-block'
            
            return postgres_button_style, download_button_style
        
        # Callback para la descarga de archivos
        @self.app.callback(
            Output('download-data', 'data'),
            Input('download-etl-button', 'n_clicks'),
            State('export-format', 'value'),
            State('filename-input', 'value'),
            prevent_initial_call=True
        )
        def generate_download(n_clicks, export_format, filename):
            if n_clicks is None or n_clicks == 0:
                return dash.no_update
                    
            # Verificar que se haya procesado el ETL
            if 'processed_data' not in DATAFRAMES:
                # No podemos descargar nada si no hay datos
                return dash.no_update
            
            processed_df = DATAFRAMES['processed_data']
            
            try:
                if export_format == 'csv':
                    # Crear un buffer en memoria para el CSV
                    buffer = io.StringIO()
                    processed_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    return dict(
                        content=buffer.getvalue(),
                        filename=f"{filename}.csv",
                        type="text/csv"
                    )
                
                elif export_format == 'xlsx':
                    # Para Excel, necesitamos codificar el contenido en base64
                    import base64
                    from io import BytesIO
                    
                    # Crear un buffer en memoria para el Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        processed_df.to_excel(writer, index=False)
                    output.seek(0)
                    
                    # Codificar en base64
                    content = base64.b64encode(output.getvalue()).decode('utf-8')
                    
                    return dict(
                        content=content,
                        filename=f"{filename}.xlsx",
                        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        base64=True
                    )
                
                elif export_format == 'json':
                    # Para JSON, preparamos una representación en string
                    records = processed_df.replace({pd.NA: None}).to_dict('records')
                    
                    # Limpiar valores nulos y convertir a formato JSON
                    def clean_value(value):
                        if pd.isna(value) or value is pd.NA:
                            return None
                        return value
                    
                    clean_records = []
                    for record in records:
                        clean_record = {}
                        for key, value in record.items():
                            clean_record[key] = clean_value(value)
                        clean_records.append(clean_record)
                    
                    # Convertir a JSON string
                    json_string = json.dumps(clean_records, ensure_ascii=False, default=str, indent=4)
                    
                    return dict(
                        content=json_string,
                        filename=f"{filename}.json",
                        type="application/json"
                    )
                
                elif export_format == 'postgres':
                    # Para PostgreSQL no hay descarga, se maneja por otro botón
                    return dash.no_update
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Mostrar mensaje de error en la interfaz
                PROCESS_DATASET['save_output'] = html.Div([
                    html.H3("❌ Error al preparar la descarga", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P(f"Detalle: {str(e)}")
                ])
                return dash.no_update
        
        # Callback para guardar en PostgreSQL
        @self.app.callback(
            Output('save-output', 'children'),
            [Input('save-postgres-button', 'n_clicks')],
            [State('postgres-host', 'value'),
            State('postgres-port', 'value'),
            State('postgres-db', 'value'),
            State('postgres-schema', 'value'),
            State('postgres-table', 'value'),
            State('postgres-user', 'value'),
            State('postgres-password', 'value')],
            prevent_initial_call=True
        )
        def save_to_postgres(n_clicks, pg_host, pg_port, pg_db, pg_schema, pg_table, pg_user, pg_password):
            if n_clicks is None or n_clicks == 0:
                return dash.no_update
                    
            # Verificar que se haya procesado el ETL
            if 'processed_data' not in DATAFRAMES:
                return html.Div([
                    html.H3("❌ Error al guardar", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P("No hay datos procesados disponibles. Por favor, ejecute primero el proceso ETL.")
                ])
            
            processed_df = DATAFRAMES['processed_data']
            
            try:
                # Convertir DataFrame a CSV en memoria
                csv_buffer = StringIO()
                processed_df.to_csv(csv_buffer, index=False, header=True)
                csv_buffer.seek(0)
                
                # Establecer conexión directa con PostgreSQL
                conn = psycopg2.connect(
                    host=pg_host,
                    port=pg_port,
                    database=pg_db,
                    user=pg_user,
                    password=pg_password
                )
                
                # Asegurarse de que el esquema existe
                with conn.cursor() as cursor:
                    if pg_schema != 'public':
                        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                            sql.Identifier(pg_schema)
                        ))
                    
                    # Crear tabla basada en las columnas del DataFrame
                    columns = []
                    for col_name, dtype in processed_df.dtypes.items():
                        # Mapeo simple de tipos pandas a PostgreSQL
                        if "int" in str(dtype):
                            pg_type = "INTEGER"
                        elif "float" in str(dtype):
                            pg_type = "FLOAT"
                        elif "datetime" in str(dtype):
                            pg_type = "TIMESTAMP"
                        elif "bool" in str(dtype):
                            pg_type = "BOOLEAN"
                        else:
                            pg_type = "TEXT"
                            
                        columns.append(sql.SQL("{} {}").format(
                            sql.Identifier(col_name),
                            sql.SQL(pg_type)
                        ))
                    
                    # Construir y ejecutar la sentencia CREATE TABLE
                    create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                        sql.Identifier(pg_schema),
                        sql.Identifier(pg_table),
                        sql.SQL(", ").join(columns)
                    )
                    cursor.execute(create_table_query)
                    
                    # Eliminar datos existentes si la tabla ya existía
                    cursor.execute(sql.SQL("DELETE FROM {}.{}").format(
                        sql.Identifier(pg_schema),
                        sql.Identifier(pg_table)
                    ))
                    
                    # Usar COPY para importar CSV a la tabla
                    cursor.copy_expert(
                        sql.SQL("COPY {}.{} FROM STDIN WITH CSV HEADER").format(
                            sql.Identifier(pg_schema),
                            sql.Identifier(pg_table)
                        ),
                        csv_buffer
                    )
                    
                    conn.commit()
                
                conn.close()
                
                save_result = html.Div([
                    html.H3("✅ Guardado exitoso en PostgreSQL", style={'color': 'green', 'fontWeight': 'bold'}),
                    html.P(f"Los datos se han guardado correctamente en: {pg_host}:{pg_port}/{pg_db}.{pg_schema}.{pg_table}"),
                    html.P(f"Se guardaron {len(processed_df)} registros en la tabla.")
                ])
                
                # Almacenar resultado del guardado en estado compartido
                PROCESS_DATASET['save_output'] = save_result
                return save_result
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                save_result = html.Div([
                    html.H3("❌ Error al guardar en PostgreSQL", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P(f"Detalle: {str(e)}")
                ])
                PROCESS_DATASET['save_output'] = save_result
                return save_result

        # Añadir un callback para limpiar los resultados del ETL
        @self.app.callback(
            Output('etl-output', 'children', allow_duplicate=True),
            Output('save-output', 'children', allow_duplicate=True),
            Output('export-options-div', 'style', allow_duplicate=True),
            Input('clear-etl-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def clear_etl_results(n_clicks):
            if n_clicks:
                # Limpiar resultados ETL del estado compartido
                if 'etl_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['etl_output']
                if 'save_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['save_output']
                if 'processed_data' in DATAFRAMES:
                    del DATAFRAMES['processed_data']
                if 'export_options_div' in PROCESS_DATASET:
                    del PROCESS_DATASET['export_options_div']
                if 'data_mining_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['data_mining_output']

                return [html.P("", style={'margin': '20px'})], [], {'display': 'none'}
            
            return dash.no_update, dash.no_update, dash.no_update
        
        # Mantener este callback para mostrar/ocultar configuración de PostgreSQL
        @self.app.callback(
            Output('postgres-config', 'style'),
            Output('file-config', 'style'),
            Input('export-format', 'value')
        )
        def toggle_postgres_config(export_format):
            if export_format == 'postgres':
                return {'display': 'block', 'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'marginTop': '10px'}, {'display': 'none'}
            else:
                return {'display': 'none'}, {'display': 'block'}        


    def process_dataframe(self, df) -> Tuple[pd.DataFrame, list]:
        process_elements = []
        copy_df = df.copy()
        initial_count = len(copy_df)
        
        # Eliminar duplicados
        copy_df = copy_df.drop_duplicates()
        removed = initial_count - len(copy_df)
        element = html.Div([
            html.H3("Eliminación de duplicados"),
            html.P(f"Se eliminaron {removed} duplicados. Quedan {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=5,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)

        # Rellenar filas de columnas numericas con datos que no son numericos con 0 o 0.0
        for column in copy_df.columns:
            if copy_df[column].dtype in ['int64', 'float64']:
                len_before = len(copy_df)
                copy_df[column] = pd.to_numeric(copy_df[column], errors='coerce').fillna(0)
                if len_before != len(copy_df):
                    element = html.Div([
                        html.H3(f"Relleno de filas en {column}"),
                        html.P(f"Se reemplazaron {len_before-len(copy_df)} valores. Manteniendo {len(copy_df)} registros."),
                        dash_table.DataTable(
                            data=copy_df.head(5).to_dict('records'),
                            columns=[{"name": i, "id": i} for i in copy_df.columns],
                            page_size=5,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                        ),
                        html.Hr(),
                    ])
                    process_elements.append(element)

        # Borrar columnas con un porcentaje de nulos mayor al 50%
        columns_to_drop = []
        for column in copy_df.columns:
            missing_values = int(copy_df[column].isnull().sum())
            if missing_values / len(copy_df) > 0.5:
                columns_to_drop.append(column)
                
        copy_df.drop(columns=columns_to_drop, inplace=True)
            
        element = html.Div([
            html.H3("Eliminación de columnas con más del 50% de nulos"),
            html.P(f"Se eliminaron {len(columns_to_drop)} columnas. Quedan {len(copy_df.columns)} columnas y {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=5,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)

        # En la columna 'lead_time', bajamos los valores que eran demasiado altos al nivel del promedio.
        mean_lead_time = int(copy_df['lead_time'].mean())
        affected_rows = len(copy_df[copy_df['lead_time'] > mean_lead_time])
        copy_df['lead_time'] = copy_df['lead_time'].apply(lambda x: mean_lead_time if x > mean_lead_time else x)
        element = html.Div([
            html.H3("Reemplazo de valores mayores al promedio en lead_time"),
            html.P(f"Se reemplazaron {affected_rows} valores. Quedan {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=5,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)

        # Eliminar filas con 0 o más de 10 adultos 
        if 'adults' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df.query('0 < adults <= 10')
            element = html.Div([
                html.H3("Eliminación de filas con 0 o más de 10 adultos"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Eliminar NaN en babies 
        if 'babies' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df.dropna(subset=['babies'])
            if len_before > len(copy_df):
                element = html.Div([
                    html.H3("Eliminación de filas con babies nulo"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        page_size=5,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Eliminar filas con menos de 0 o más de 10 children
        if 'children' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df.query('0 <= children <= 10')
            element = html.Div([
                html.H3("Eliminación de filas con menos de 0 o más de 10 children"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        #  Eliminar mayores al promedio stays_in_weekend_nights
        if 'stays_in_weekend_nights' in copy_df.columns:
            len_before = len(copy_df)
            mean_weekend_nights = copy_df['stays_in_weekend_nights'].mean()
            copy_df = copy_df.query('0 <= stays_in_weekend_nights <= @mean_weekend_nights')
            element = html.Div([
                html.H3("Eliminación de filas donde stays_in_weekend_nightss es mayor al promedio"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        #  Eliminar mayores al promedio stays_in_week_nights
        if 'stays_in_week_nights' in copy_df.columns:
            len_before = len(copy_df)          
            mean_week_nights = copy_df['stays_in_week_nights']
            copy_df = copy_df.query('0 <= stays_in_week_nights <= @mean_week_nights')
            element = html.Div([
                html.H3("Eliminación de filas donde stays_in_week_nights es mayor al promedio"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=5,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Days_in_waiting_list > 0 and 1.5 rango intercuartilico
        if 'days_in_waiting_list' in copy_df.columns:
            len_before = len(copy_df)
            # Solo considerar valores mayores a 0
            filtered = copy_df[copy_df['days_in_waiting_list'] > 0]
            if not filtered.empty:
                Q1 = filtered['days_in_waiting_list'].quantile(0.25)
                Q3 = filtered['days_in_waiting_list'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Mantener los que están dentro del rango o son 0
                mask = (
                    (copy_df['days_in_waiting_list'] == 0) |
                    ((copy_df['days_in_waiting_list'] > 0) &
                     (copy_df['days_in_waiting_list'] >= lower_bound) &
                     (copy_df['days_in_waiting_list'] <= upper_bound))
                )
                copy_df = copy_df[mask]
                element = html.Div([
                    html.H3("Filtrado de outliers en days_in_waiting_list usando rango intercuartílico"),
                    html.P(f"Se eliminaron {len_before - len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        page_size=5,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        #   previous cancellations / babies / children / is_repetead_guests ??? / previous_bookings_not_canceled / booking_changes / 
        copy_df.drop(columns=['previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes'], inplace=True, errors='ignore')

        # cambiar a frecuencia company y agent de cuantitativo a cualitativo
        if 'company' in copy_df.columns:
            copy_df['company'] = copy_df['company'].astype('category')
        if 'agent' in copy_df.columns:
            copy_df['agent'] = copy_df['agent'].astype('category') 

        # Eliminar filas con Undefined
        for column in copy_df.columns:
            if copy_df[column].dtype == 'object':
                len_before = len(copy_df)
                copy_df = copy_df[copy_df[column] != 'Undefined']
                if len_before != len(copy_df):  # Solo mostrar si hubo cambios
                    element = html.Div([
                        html.H3(f"Eliminación de filas con Undefined en {column}"),
                        html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                        dash_table.DataTable(
                            data=copy_df.head(5).to_dict('records'),
                            columns=[{"name": i, "id": i} for i in copy_df.columns],
                            page_size=5,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                        ),
                        html.Hr(),
                    ])
                    process_elements.append(element)

        # Eliminar filas de previous_cancellation mayores al promedio y menores a 0
        if 'previous_cancellations' in copy_df.columns:
            mean_previous_cancellations = copy_df['previous_cancellations'].mean()
            if pd.notna(mean_previous_cancellations):
                mean_previous_cancellations = int(mean_previous_cancellations)
                len_before = len(copy_df)
                copy_df = copy_df.query('0 <= previous_cancellations <= @mean_previous_cancellations')
                element = html.Div([
                    html.H3("Eliminación de filas con previous_cancellations mayores al promedio y menores a 0"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Eliminar filas previous_bookings_not_canceled mayores al promedio y menores a 0
        if 'previous_bookings_not_canceled' in copy_df.columns:
            mean_previous_bookings_not_canceled = copy_df['previous_bookings_not_canceled'].mean()
            if pd.notna(mean_previous_bookings_not_canceled):
                mean_previous_bookings_not_canceled = int(mean_previous_bookings_not_canceled)
                len_before = len(copy_df)
                copy_df = copy_df.query('0 <= previous_bookings_not_canceled <= @mean_previous_bookings_not_canceled')
                element = html.Div([
                    html.H3("Eliminación de filas con previous_bookings_not_canceled mayores al promedio y menores a 0"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Elminar filas en booking_changes mayores al promedio y menores a 0
        if 'booking_changes' in copy_df.columns:
            mean_booking_changes = copy_df['booking_changes'].mean()
            if pd.notna(mean_booking_changes):
                mean_booking_changes = int(mean_booking_changes)
                len_before = len(copy_df)
                copy_df = copy_df.query('0 <= booking_changes <= @mean_booking_changes')
                if len_before > len(copy_df):  # Solo mostrar si hubo cambios
                    element = html.Div([
                        html.H3("Eliminación de filas con booking_changes mayores al promedio y menores a 0"),
                        html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                        dash_table.DataTable(
                            data=copy_df.head(5).to_dict('records'),
                            columns=[{"name": i, "id": i} for i in copy_df.columns],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                        ),
                        html.Hr(),
                    ])
                    process_elements.append(element)

        # Llenar agent con la moda
        if 'agent' in copy_df.columns:
            mode_series = copy_df['agent'].mode()
            if not mode_series.empty:
                mode_agent = mode_series[0]
                columns_to_fill = copy_df['agent'].isna().sum()
                if columns_to_fill > 0:
                    copy_df['agent'] = copy_df['agent'].apply(lambda x: mode_agent if pd.isna(x) else x)
                    element = html.Div([
                        html.H3("Llenado de agent con la moda"),
                        html.P(f"Se llenaron {columns_to_fill} valores. Manteniendo {len(copy_df)} registros."),
                        dash_table.DataTable(
                            data=copy_df.head(5).to_dict('records'),
                            columns=[{"name": i, "id": i} for i in copy_df.columns],
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                        ),
                        html.Hr(),
                    ])
                    process_elements.append(element)

        # Eliminar valores nulos en country
        if 'country' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df.dropna(subset=['country'])
            if len_before > len(copy_df):  # Solo mostrar si hubo cambios
                element = html.Div([
                    html.H3("Eliminación de filas con country nulo"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Reemplazar valores menores e igual a 0 o a 2 de desviación estándar en adr con la media
        if 'adr' in copy_df.columns:
            std_adr = copy_df['adr'].std()
            mean_adr = copy_df['adr'].mean()
            lower_bound = mean_adr - 2 * std_adr
            upper_bound = mean_adr + 2 * std_adr
            affected_rows = len(copy_df[(copy_df['adr'] <= 0) | (~copy_df['adr'].between(lower_bound, upper_bound))])
            copy_df['adr'] = copy_df['adr'].apply(
                lambda x: mean_adr if x <= 0 or not (lower_bound <= x <= upper_bound) else x
            )
            if affected_rows > 0:
                element = html.Div([
                    html.H3("Reemplazo de valores menores e igual a 0 o a 2 de desviación estándar en adr con la media"),
                    html.P(f"Se reemplazaron {affected_rows} valores. Manteniendo {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)
            
            # Eliminar filas con adr nulo
            len_before = len(copy_df)
            copy_df = copy_df.dropna(subset=['adr'])
            if len_before > len(copy_df):  # Solo mostrar si hubo cambios
                element = html.Div([
                    html.H3("Eliminación de filas con adr nulo"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Eliminar reservation_status_date con frecuencia menor a la media
        if 'reservation_status_date' in copy_df.columns:
            reservation_counts = copy_df['reservation_status_date'].value_counts()
            mean_count = reservation_counts.mean()
            valid_dates = reservation_counts[reservation_counts >= mean_count].index
            len_before = len(copy_df)
            copy_df = copy_df[copy_df['reservation_status_date'].isin(valid_dates)]
            if len_before > len(copy_df):  # Solo mostrar si hubo cambios
                element = html.Div([
                    html.H3("Eliminación de reservation_status_date con frecuencia menor a la media"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)
            
            # Cambiar a formato de fecha y ordenar
            len_before = len(copy_df)
            copy_df['reservation_status_date'] = copy_df['reservation_status_date'].astype('datetime64[ns]', errors='ignore')
            copy_df = copy_df[copy_df['reservation_status_date'].notna()]
            copy_df = copy_df.sort_values(by='reservation_status_date')
            element = html.Div([
                html.H3("Cambio de reservation_status_date a formato de fecha real"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas con fechas inválidas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        # Reemplazar valores nulos en required_car_parking_spaces con 0
        if 'required_car_parking_spaces' in copy_df.columns:
            rows_to_fill = copy_df['required_car_parking_spaces'].isna().sum()
            copy_df['required_car_parking_spaces'] = copy_df['required_car_parking_spaces'].fillna(0)
            element = html.Div([
                html.H3("Reemplazo de valores nulos en required_car_parking_spaces con 0"),
                html.P(f"Se reemplazaron {rows_to_fill} valores. Manteniendo {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
                
        # Eliminar filas en children menores a 0 y mayores al promedio
        if 'children' in copy_df.columns:
            mean_children = copy_df['children'].mean()
            if pd.notna(mean_children):
                mean_children = int(mean_children)
                len_before = len(copy_df)
                copy_df = copy_df.query('0 <= children <= @mean_children')
                element = html.Div([
                html.H3("Eliminación de filas en children menores a 0 y mayores al promedio"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Cambiamos el texto 'INVALID_MONTH' por un valor nulo, y después quitamos esas filas.
        # Así nos aseguramos de quedarnos solo con meses válidos.
        if 'arrival_date_month' in copy_df.columns:
            copy_df['arrival_date_month'] = copy_df['arrival_date_month'].replace('INVALID_MONTH', pd.NA)
            len_before = len(copy_df)
            copy_df = copy_df[copy_df['arrival_date_month'].notna()]
            element = html.Div([
                html.H3("Eliminación de filas con arrival_date_month INVALID_MONTH"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        # Filtramos todas las filas donde el tipo de habitación empieza con 'X' (que significa que está mal).
        # Esto lo hicimos con una función que detecta si una cadena empieza con cierta letra.
        if 'assigned_room_type' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df[~copy_df['assigned_room_type'].str.startswith('X')]
            element = html.Div([
                html.H3("Eliminación de filas con assigned_room_type que empiezan con 'X'"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Reemplazamos el valor 'UNKNOWN' en la columna 'deposit_type' por algo más entendible: 'Otro'.
        # Así evitamos quedarnos con categorías sin sentido.
        if 'deposit_type' in copy_df.columns:
            replaced_count = len(copy_df[copy_df['deposit_type'] == 'UNKNOWN'])
            copy_df['deposit_type'] = copy_df['deposit_type'].replace('UNKNOWN', 'Otro')
            element = html.Div([
                html.H3("Reemplazo de valores UNKNOWN en deposit_type por Otro"),
                html.P(f"Se reemplazaron {replaced_count} valores. Manteniendo {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Quitamos las filas donde el país fuera 'INVALID_COUNTRY', usando una función que revisa coincidencias exactas de texto.
        # Así nos aseguramos de trabajar solo con países válidos.
        if 'country' in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df[copy_df['country'] != 'INVALID_COUNTRY']
            element = html.Div([
                html.H3("Eliminación de filas con country INVALID_COUNTRY"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Feature engineering
        # Agregar columna derivada total_guests
        copy_df['total_guests'] = copy_df['adults'] + copy_df['children'] + copy_df['babies']
        element = html.Div([
            html.H3("Agregando columna total_guests"),
            html.P(f"Se mantienen {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)
        
        # Agregar un campo booleano para saber si la estancia fue mayor a 7 días
        copy_df['stays_longer_than_7_days'] = copy_df['stays_in_weekend_nights'] + copy_df['stays_in_week_nights'] > 7
        element = html.Div([
            html.H3("Agregando columna stay_longer_than_7_days"),
            html.P(f"Se mantienen {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)
        
        # Agregar campo total_nights
        copy_df['total_nights'] = copy_df['stays_in_weekend_nights'] + copy_df['stays_in_week_nights']
        element = html.Div([
            html.H3("Agregando columna total_nights"),
            html.P(f"Se mantienen {len(copy_df)} registros."),
            dash_table.DataTable(
                data=copy_df.head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)

        # total nights eliminar valores atipicos
        if 'total_nights' in copy_df.columns:
            len_before = len(copy_df)
            mean_total_nights = copy_df['total_nights'].mean()
            copy_df = copy_df.query('0 <= total_nights <= @mean_total_nights')
            element = html.Div([
                html.H3("Eliminación de filas con total_nights mayores al promedio y menores a 0"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # is_canceled cambiar a cualitativo
        if 'is_canceled' in copy_df.columns:
            copy_df['is_canceled'] = copy_df['is_canceled'].astype('category')
            element = html.Div([
                html.H3("Cambio de is_canceled a cualitativo"),
                html.P(f"Se mantienen {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # arrival_date_year cambiar a cualitativo
        if 'arrival_date_year' in copy_df.columns:
            copy_df['arrival_date_year'] = copy_df['arrival_date_year'].astype('category')
            element = html.Div([
                html.H3("Cambio de arrival_date_year a cualitativo"),
                html.P(f"Se mantienen {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        # TODO: arrival_date_year eliminar valores atipicos
        if 'arrival_date_year' in copy_df.columns:
            copy_df = copy_df[copy_df['arrival_date_year'].isin([2015, 2016, 2017, 2018])]
            element = html.Div([
                html.H3("Eliminación de filas con arrival_date_year fuera del rango 2015-2018"),
                html.P(f"Se mantienen {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
        # is_repeated_guest cambiar a cualitativo
        if 'is_repeated_guest' in copy_df.columns:
            copy_df['is_repeated_guest'] = copy_df['is_repeated_guest'].astype('category')
            element = html.Div([
                html.H3("Cambio de is_repeated_guest a cualitativo"),
                html.P(f"Se mantienen {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # total_guests eliminar atipicos
        if 'total_guests' in copy_df.columns:
           q1 = copy_df['total_guests'].quantile(0.25)
           q3 = copy_df['total_guests'].quantile(0.75)
           iqr = q3 - q1
           lower_bound = q1 - 1.5 * iqr
           upper_bound = q3 + 1.5 * iqr
           len_before = len(copy_df)
           copy_df = copy_df.query('0 <= total_guests <= @upper_bound')
           element = html.Div([
                html.H3("Eliminación de filas con total_guests mayores al promedio y menores a 0"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
           
        # stay_longer_than_7_days cambiar a cualitativo
        if 'stays_longer_than_7_days' in copy_df.columns:
                copy_df['stays_longer_than_7_days'] = copy_df['stays_longer_than_7_days'].astype('category')
                element = html.Div([
                     html.H3("Cambio de stays_longer_than_7_days a cualitativo"),
                     html.P(f"Se mantienen {len(copy_df)} registros."),
                     dash_table.DataTable(
                          data=copy_df.head(5).to_dict('records'),
                          columns=[{"name": i, "id": i} for i in copy_df.columns],
                          style_table={'overflowX': 'auto'},
                          style_cell={'textAlign': 'left'},
                     ),
                     html.Hr(),
                ])
                process_elements.append(element)
            
        # total_nights eliminar los que tienen 0 no sirven para el analisis.
        if 'total_nights' in copy_df.columns:
                len_before = len(copy_df)
                copy_df = copy_df[copy_df['total_nights'] > 0]
                element = html.Div([
                    html.H3("Eliminación de filas con total_nights igual a 0"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas. Quedan {len(copy_df)} registros."),
                    dash_table.DataTable(
                        data=copy_df.head(5).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])

        # agregar columna is_demading_client >= total_of_special_requests 2
        if 'total_of_special_requests' in copy_df.columns:
            copy_df['is_demanding_client'] = copy_df['total_of_special_requests'] >= 2
            copy_df['is_demanding_client'] = copy_df['is_demanding_client'].astype('category')
            element = html.Div([
                html.H3("Agregando columna is_demading_client"),
                html.P(f"Se mantienen {len(copy_df)} registros."),
                dash_table.DataTable(
                    data=copy_df.head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Resumen final del procesamiento
        element = html.Div([
            html.H3("Resumen de procesamiento"),
            html.P(f"Registros iniciales: {initial_count}"),
            html.P(f"Registros finales: {len(copy_df)}"),
            html.P(f"Reducción total: {initial_count - len(copy_df)} registros ({(initial_count - len(copy_df))/initial_count*100:.2f}%)"),
            html.Hr(),
        ])
        process_elements.append(element)
        
        return copy_df, process_elements
    
    @classmethod
    def save_to_file(cls, df, filepath, format_type):
        try:
            # Verificar si es una ruta absoluta o relativa
            if os.path.isabs(filepath):
                # Si es ruta absoluta, usarla directamente
                full_path = filepath
            else:
                # Si es ruta relativa, combinarla con el directorio actual
                full_path = os.path.join(os.getcwd(), filepath)
            
            # Asegurar que el directorio existe
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Creando directorio: {directory}")
            
            # Asegurar la extensión correcta
            if format_type == 'csv' and not filepath.lower().endswith('.csv'):
                full_path += '.csv'
            elif format_type == 'xlsx' and not filepath.lower().endswith('.xlsx'):
                full_path += '.xlsx'
            elif format_type == 'json' and not filepath.lower().endswith('.json'):
                full_path += '.json'
            
            print(f"Guardando archivo en: {full_path}")
            
            if format_type == 'csv':
                df.to_csv(full_path, index=False)
            elif format_type == 'xlsx':
                df.to_excel(full_path, index=False)
            elif format_type == 'json':
                # Convertir a formato JSON más limpio
                # Convertir el DataFrame a una lista de diccionarios
                records = df.replace({pd.NA: None}).to_dict('records')
                
                # Para valores NaN, None, etc.
                def clean_value(value):
                    if pd.isna(value) or value is pd.NA:
                        return None
                    return value
                    
                # Procesar cada registro para limpiar valores nulos
                clean_records = []
                for record in records:
                    clean_record = {}
                    for key, value in record.items():
                        clean_record[key] = clean_value(value)
                    clean_records.append(clean_record)
                
                # Guardar como JSON con formato adecuado
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(clean_records, f, indent=4, ensure_ascii=False, default=str)
            
            return True, full_path
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    @classmethod
    def save_to_postgres(cls, df, host, port, dbname, schema, table, user, password):
        try:            
            print(f"Conectando a PostgreSQL: {host}:{port}/{dbname} como {user}")
            
            # Asegurarse de que todos los parámetros sean strings
            host = str(host)
            port = str(port)
            dbname = str(dbname)
            user = str(user)
            schema = str(schema)
            table = str(table)
            password = str(password)
            
            # Crear conexión usando psycopg2
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                client_encoding='UTF8'
            )
            
            # Establecer autocommit a False para usar transacciones
            conn.autocommit = False
            
            try:
                with conn.cursor() as cursor:
                    # Verificar si el esquema existe, si no crearlo
                    cursor.execute(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema}';")
                    if cursor.fetchone() is None:
                        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema)))
                        print(f"Esquema {schema} creado")
                        conn.commit()
                    
                    # Crear tabla (eliminar si existe)
                    cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}.{};").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    ))
                    conn.commit()
                    
                    # Generar lista de columnas para crear la tabla
                    columns = []
                    for col_name, dtype in df.dtypes.items():
                        safe_col_name = str(col_name).replace(" ", "_").replace("-", "_")
                        
                        if pd.api.types.is_object_dtype(dtype):
                            columns.append(f"\"{safe_col_name}\" TEXT")
                        elif pd.api.types.is_integer_dtype(dtype):
                            columns.append(f"\"{safe_col_name}\" INTEGER")
                        elif pd.api.types.is_float_dtype(dtype):
                            columns.append(f"\"{safe_col_name}\" NUMERIC")
                        elif pd.api.types.is_datetime64_any_dtype(dtype):
                            columns.append(f"\"{safe_col_name}\" TIMESTAMP")
                        elif pd.api.types.is_bool_dtype(dtype):
                            columns.append(f"\"{safe_col_name}\" BOOLEAN")
                        else:
                            columns.append(f"\"{safe_col_name}\" TEXT")
                    
                    # Crear la tabla
                    create_table_sql = f"CREATE TABLE {schema}.{table} ({', '.join(columns)});"
                    print(f"Creando tabla con: {create_table_sql}")
                    cursor.execute(create_table_sql)
                    conn.commit()
                    print(f"Tabla {schema}.{table} creada")
                    
                    # Preparar el DataFrame para inserción
                    df_copy = df.copy()
                    df_copy.columns = [str(col).replace(" ", "_").replace("-", "_") for col in df_copy.columns]
                    
                    # Convertir datetime a string para evitar problemas
                    for col in df_copy.select_dtypes(include=['datetime64']).columns:
                        df_copy[col] = df_copy[col].astype(str)
                    
                    # Usar INSERT en lugar de COPY FROM
                    print(f"Insertando {len(df_copy)} filas en {schema}.{table}...")
                    
                    # Crear la plantilla para el INSERT
                    columns_str = ", ".join([f'"{col}"' for col in df_copy.columns])
                    placeholders = ", ".join(["%s" for _ in df_copy.columns])
                    insert_template = f"INSERT INTO {schema}.{table} ({columns_str}) VALUES ({placeholders})"
                    
                    # Insertar en lotes para mayor eficiencia
                    batch_size = 100
                    for i in range(0, len(df_copy), batch_size):
                        batch = df_copy.iloc[i:min(i+batch_size, len(df_copy))]
                        batch_data = [tuple(row) for _, row in batch.iterrows()]
                        
                        cursor.executemany(insert_template, batch_data)
                        conn.commit()
                        
                        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(df_copy):
                            print(f"Insertados {min(i+batch_size, len(df_copy))} de {len(df_copy)} registros")
                    
                    print(f"Inserción completa en {schema}.{table}")
                
                # Confirmar y cerrar la conexión
                conn.commit()
                conn.close()
                
                connection_info = f"{host}:{port}, base de datos: {dbname}, esquema: {schema}, tabla: {table}"
                return True, connection_info
                
            except Exception as e:
                # Hacer rollback en caso de error
                conn.rollback()
                conn.close()
                raise e
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Error al conectar con PostgreSQL: {str(e)}"
