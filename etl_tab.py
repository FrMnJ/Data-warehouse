from typing import Tuple
from dash import Input, dcc, html, callback, State, dash_table, Output
import pandas as pd

PROCESS_DATASET = {}
class ETLTab:
    def __init__(self, app):
        self.app = app

    def render(self):
        etl_output = PROCESS_DATASET.get('etl_output', [html.Div("No hay datos procesados")])
        return html.Div([
            html.H2("ETL",
                    style={'margin': '20px'}),  
            html.Div(
                id='etl-output',
                children=etl_output
            ),
        ])

    @staticmethod
    def process_dataframe(df) -> Tuple[pd.DataFrame, list]:
        process_elements = []
        copy_df = df.copy()
        # Eliminar duplicados
        copy_df = copy_df.where(~copy_df.duplicated())
        element = html.Div([
            html.H3("Eliminación de duplicados"),
            html.P(f"Se eliminaron {len(df) - len(copy_df)} duplicados"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)
        # Eliminar filas NaN de columnas numericas
        copy_df = copy_df.dropna(subset=copy_df.select_dtypes(include=['number']).columns)
        element = html.Div([
            html.H3("Eliminación de filas NaN de columnas numéricas"),
            html.P(f"Se eliminaron {len(df) - len(copy_df)} filas"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
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
            html.P(f"Se eliminaron {len(columns_to_drop)} columnas"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)

        # En la columna 'lead_time', bajamos los valores que eran demasiado altos al nivel del promedio.
        mean_lead_time = int(copy_df['lead_time'].mean())
        copy_df['lead_time'].where(copy_df['lead_time'] > mean_lead_time, mean_lead_time)
        element = html.Div([
            html.H3("Reemplazo de valores mayores al promedio en lead_time"),
            html.P(f"Se reemplazaron {len(copy_df[copy_df['lead_time'] > mean_lead_time])} valores"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
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
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Eliminar  filas con más de 2 bebes column 'babies'
        if "babies" in copy_df.columns:
            len_before = len(copy_df)
            copy_df = copy_df.query('0 <= babies <= 2')
            element = html.Div([
                html.H3("Eliminación de filas con más de 2 bebes"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        # Eliminar filas con Undefined
        for column in copy_df.columns:
            if copy_df[column].dtype == 'object':
                len_before = len(copy_df)
                copy_df = copy_df[copy_df[column] != 'Undefined']
                element = html.Div([
                    html.H3(f"Eliminación de filas con Undefined en {column}"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                    dash_table.DataTable(
                        data=copy_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)

        # Eliminar filas con previous_cancellations mayores a 1.5 x el rango intercuartil y menores a 0
        if 'previous_cancellations' in copy_df.columns:
            q1 = copy_df['previous_cancellations'].quantile(0.25)
            q3 = copy_df['previous_cancellations'].quantile(0.75)
            iqr = q3 - q1
            len_before = len(copy_df)
            copy_df = copy_df.query('0 < previous_cancellations <= @q3 + 1.5 * @iqr')
            element = html.Div([
                html.H3("Eliminación de filas con previous_cancellations mayores a 1.5 x el rango intercuartil y menores a 0"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        # Eliminar filas previous_bookings_not_canceled mayores a 1.5 x el rango intercuartil y menores a 0
        if 'previous_bookings_not_canceled' in copy_df.columns:
            q1 = copy_df['previous_bookings_not_canceled'].quantile(0.25)
            q3 = copy_df['previous_bookings_not_canceled'].quantile(0.75)
            iqr = q3 - q1
            len_before = len(copy_df)
            copy_df = copy_df.query('0 < previous_bookings_not_canceled <= @q3 + 1.5 * @iqr')
            element = html.Div([
                html.H3("Eliminación de filas con previous_bookings_not_canceled mayores a 1.5 x el rango intercuartil y menores a 0"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
                copy_df = copy_df.query('0 < booking_changes <= @mean_booking_changes')
                element = html.Div([
                    html.H3("Eliminación de filas con booking_changes mayores al promedio y menores a 0"),
                    html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                    dash_table.DataTable(
                        data=copy_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)
            # else: do nothing if mean is NaN

        # Llenar agent con la moda
        if 'agent' in copy_df.columns:
            mode_series = copy_df['agent'].mode()
            if not mode_series.empty:
                mode_agent = mode_series[0]
                columns_to_fill = copy_df['agent'].isna().sum()
                copy_df['agent'] = copy_df['agent'].apply(lambda x: mode_agent if pd.isna(x) else x)
                element = html.Div([
                    html.H3("Llenado de agent con la moda"),
                    html.P(f"Se llenaron {columns_to_fill} valores"),
                    dash_table.DataTable(
                        data=copy_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in copy_df.columns],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                    ),
                    html.Hr(),
                ])
                process_elements.append(element)
            # else: do nothing if mode is empty

        # Eliminar valores nulos en country
        len_before = len(copy_df)
        copy_df = copy_df.query('country.notna()', engine='python')
        element = html.Div([
            html.H3("Eliminación de filas con country nulo"),
            html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
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
            len_before = len(copy_df)
            lower_bound = mean_adr - 2 * std_adr
            upper_bound = mean_adr + 2 * std_adr
            copy_df['adr'] = copy_df['adr'].where((copy_df['adr'] > 0) & (copy_df['adr'].between(lower_bound, upper_bound)), mean_adr)
            element = html.Div([
                html.H3("Reemplazo de valores menores e igual a 0 o a 2 de desviación estándar en adr con la media"),
                html.P(f"Se reemplazaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
            # Eliminamos todas las filas que tenían valores vacíos (nulos) en la columna adr usando .notna().
            # Esto nos asegura trabajar solo con datos completos en esa columna, ya que adr representa el precio
            # promedio por noche y es un valor clave para cualquier análisis relacionado con ingresos, tarifas o rendimiento financiero.
            # Como no tiene sentido estimar este valor sin una base sólida, preferimos eliminar esas filas para mantener la calidad del análisis.
            copy_df = copy_df[copy_df['adr'].notna()]
            element = html.Div([
                html.H3("Eliminación de filas con adr nulo"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
            element = html.Div([
                html.H3("Eliminación de reservation_status_date con frecuencia menor a la media"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
            # Cambiamos las fechas a formato de fecha real, y luego ordenamos el dataset de más antiguo a más reciente.
            copy_df['reservation_status_date'] = copy_df['reservation_status_date'].astype('datetime64[ns]', errors='ignore')
            copy_df = copy_df[copy_df['reservation_status_date'].notna()]
            copy_df = copy_df.sort_values(by='reservation_status_date')
            element = html.Div([
                html.H3("Cambio de reservation_status_date a formato de fecha real"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)

        # Reemplazar valores nulos en required_car_parking_spaces con 0
        if 'required_car_parking_spaces' in copy_df.columns:
            rows_to_fill = copy_df['required_car_parking_spaces'].isna()
            copy_df['required_car_parking_spaces'] = copy_df['required_car_parking_spaces'].fillna(0)
            element = html.Div([
                html.H3("Reemplazo de valores nulos en required_car_parking_spaces con 0"),
                html.P(f"Se reemplazaron {len(rows_to_fill)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
                copy_df = copy_df.query('0 < children <= @mean_children')
                element = html.Div([
                html.H3("Eliminación de filas en children menores a 0 y mayores al promedio"),
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                ),
                html.Hr(),
            ])
            process_elements.append(element)
        
        # Reemplazamos el valor 'UNKNOWN' en la columna 'deposit_type' por algo más entendible: 'Otro'.
        # Así evitamos quedarnos con categorías sin sentido.
        if 'deposit_type' in copy_df.columns:
            copy_df['deposit_type'] = copy_df['deposit_type'].replace('UNKNOWN', 'Otro')
            element = html.Div([
                html.H3("Reemplazo de valores UNKNOWN en deposit_type por Otro"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
                html.P(f"Se eliminaron {len_before-len(copy_df)} filas"),
                dash_table.DataTable(
                    data=copy_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in copy_df.columns],
                    page_size=10,
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
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)
        # Agregar un campo booleano para saber si la estancia fue mayor a 7 días
        copy_df['stay_longer_than_7_days'] = copy_df['stays_in_weekend_nights'] + copy_df['stays_in_week_nights'] > 7
        element = html.Div([
            html.H3("Agregando columna stay_longer_than_7_days"),
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
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
            dash_table.DataTable(
                data=copy_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in copy_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            ),
            html.Hr(),
        ])
        process_elements.append(element)
        # Devolvemos el dataframe procesado y los elementos de proceso
        return copy_df, process_elements

