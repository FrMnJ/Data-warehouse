from dash import Input, Output, State, dcc, html, dash_table
import dash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objs as go
from imblearn.over_sampling import SMOTE
import dash_cytoscape as cyto

from info_compartida import DATAFRAMES, PROCESS_DATASET


class MineriaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        data_mining_output = PROCESS_DATASET.get('data_mining_output', [])

        return html.Div([
            html.H2("Minería de datos", style={
                'margin': '20px',
                'color': '#333',
                'fontWeight': 'bold',
                'letterSpacing': '1px'
            }),
            html.Div([
                html.Button(
                    'Ejecutar Minería de datos',
                    id='run-data-mining',
                    n_clicks=0,
                    style={
                        'margin': '10px 20px 10px 0',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 24px',
                        'borderRadius': '6px',
                        'fontWeight': 'bold',
                        'fontSize': '16px',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.08)',
                        'display': 'inline-block' if not data_mining_output else 'none',
                        'transition': 'background 0.3s',
                        'cursor': 'pointer'
                    }
                ),
                html.Button(
                    'Limpiar resultados Minería de datos',
                    id='clear-data-mining-output',
                    n_clicks=0,
                    style={
                        'margin': '10px 0',
                        'backgroundColor': '#f44336',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 24px',
                        'borderRadius': '6px',
                        'fontWeight': 'bold',
                        'fontSize': '16px',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.08)',
                        'display': 'none' if not data_mining_output else 'inline-block',
                        'transition': 'background 0.3s',
                        'cursor': 'pointer'
                    }
                ),
            ], style={'margin': '20px 0'}),
            html.Div([
                html.H3("Resultados de la minería de datos", style={
                    'margin': '20px 0 10px 0',
                    'color': '#222',
                    'fontWeight': 'bold',
                    'display': 'none' if not data_mining_output else 'block'
                }),
                dcc.Loading(
                    id="loading-data-mining",
                    type="circle",
                    children=html.Div(
                        id='data-mining-output',
                        children=data_mining_output if data_mining_output else [html.P("", style={'margin': '20px'})]
                    ),
                    style={'margin': '20px'}
                )
            ], id='data-mining-results-container', style={
                'backgroundColor': '#f5f7fa',
                'borderRadius': '12px',
                'padding': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            }),
        ], style={'maxWidth': '1100px', 'margin': 'auto', 'fontFamily': 'Segoe UI, Arial, sans-serif'})


    def run_decision_tree(self, X_resampled, y_resampled, criterion,
                           splitter,max_depth, min_samples_split,  min_samples_leaf):

                # Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

        # Entrenar el modelo
        model = DecisionTreeClassifier(criterion=criterion, 
                                        splitter=splitter, 
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf,
                                        class_weight='balanced',
                                        )
        model.fit(X_train, y_train)
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return  model, X_train, X_test, y_train, y_test, y_pred

    def iterate_decision_tree(self, df):
        # Definimos X y Y 
        # Eliminar columnas de tipo datetime
        df = df.drop(columns=df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns)
        df = df.drop(columns=['total_of_special_requests', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'is_canceled', 'adr'])
        assert 'is_demanding_client' in df.columns, "La columna 'is_demanding_client' no existe en el DataFrame."
        df['is_demanding_client'] = df['is_demanding_client'].astype('int')

        # Convertir las variables categoricas a numericas
        # Usando one hot encoding
        df = pd.get_dummies(df)

        # Separar las variables predictoras y la variable objetivo
        X = df.drop('is_demanding_client', axis=1)
        y = df['is_demanding_client']

        # Rebalance con SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Definimos los parametros a usar pre prunning
        criterion = ['gini']
        splitter = ['best']   
        max_depth = [2, 3, 4] 
        min_samples_split = [20, 30, 40]
        min_samples_leaf = [10, 20, 30]  
        # Guardamos los resultados de las iteraciones
        results = []
        for sl in min_samples_leaf:
            for s in min_samples_split:
                for md in max_depth:
                    for sp in splitter:
                        for cr in criterion:
                            model, X_train, X_test, y_train, y_test, y_pred = self.run_decision_tree(
                                X_resampled, y_resampled, cr, sp, md, s, sl)
                            f1_score_model = f1_score(y_test, y_pred)
                            results.append({
                                'model': model,
                                'X_train': X_train,
                                'X_test': X_test,
                                'y_train': y_train,
                                'y_test': y_test,
                                'y_pred': y_pred,
                                'criterion': cr,
                                'splitter': sp,
                                'max_depth': md,
                                'min_samples_split': s,
                                'min_samples_leaf': sl,
                                'f1_score': f1_score_model
                            })
        # Iteramos sobre las resultados para obtener el mejor
        best_model = results[0] 
        for result in results:
            if best_model['f1_score'] < result['f1_score']:
                best_model = result
        best_report = classification_report(best_model['y_test'], best_model['y_pred'], output_dict=True)
        print(f"Mejor modelo: ")
        print(best_report)
        best_model['report'] = best_report
        return best_model
    
    def sklearn_tree_to_cytoscape(self, model, feature_names, class_names):
        from sklearn.tree import _tree
        tree_ = model.tree_
        nodes = []
        edges = []
        def recurse(node, parent=None):
            if node == _tree.TREE_LEAF:
                return
            label = f"Impurity: {tree_.impurity[node]:.2f}\n"
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                if tree_.threshold[node] != 0.5:
                    label += f"{feature_names[tree_.feature[node]]} <= {tree_.threshold[node]:.2f}\n"
                else:
                    label += f"¿{feature_names[tree_.feature[node]]}?\n"
            label += f"samples: {tree_.n_node_samples[node]}\n"
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                # leaf
                value = tree_.value[node][0]
                class_idx = value.argmax()
                label += f"Clase: {class_names[class_idx]}\n"
            nodes.append({
                'data': {'id': str(node), 'label': label},
                'classes': 'leaf' if tree_.feature[node] == _tree.TREE_UNDEFINED else 'split'
            })
            if parent is not None:
                edges.append({'data': {'source': str(parent), 'target': str(node)}})
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            if left != _tree.TREE_LEAF:
                recurse(left, node)
            if right != _tree.TREE_LEAF:
                recurse(right, node)
        recurse(0)
        return nodes + edges
    
    def run_linear_regression_special_requests(self, df):
        # Elimina columnas que no deben usarse como predictores
        drop_cols = [
            'total_of_special_requests', 'reservation_status_date'
        ]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['total_of_special_requests']

        # Antes de pd.get_dummies(X)
        for col in ['agent', 'company']:
            if col in X.columns:
                X[col] = X[col].astype(str)
        # One-hot encoding para variables categóricas
        X = pd.get_dummies(X)

        # Entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

    def register_callbacks(self):
        @self.app.callback(
            Output('data-mining-output', 'children', allow_duplicate=True),
            Output('run-data-mining', 'style', allow_duplicate=True),
            Output('clear-data-mining-output', 'style', allow_duplicate=True),
            Input('run-data-mining', 'n_clicks'),
            State('run-data-mining', 'style'),
            State('clear-data-mining-output', 'style'),
            prevent_initial_call=True,
        )
        def run_data_mining_process(n_clicks, run_style, clear_style):
            print("Ejecutando la minería de datos")
            if n_clicks is None:
                print("No se ha hecho clic en el botón")
                return [html.P("")], dash.no_update, dash.no_update

            if 'processed_data' not in DATAFRAMES:
                print("No hay datos procesados")
                return [html.P("No hay datos procesados. Por favor, ejecute el ETL en  la pestaña 'ETL'.", 
                            style={'color': 'red', 'fontWeight': 'bold', 'margin': '20px'})], dash.no_update, dash.no_update
            
            processed_df = DATAFRAMES['processed_data'] 
            print("Iterando sobre parametros del arbol de decisión")
            model = self.iterate_decision_tree(processed_df)

            children = []
            print("Mostrando resultados del arbol de decisión")
            # Mostrar confusión matrix con Plotly
            cm = confusion_matrix(model['y_test'], model['y_pred'], labels=[0, 1])
            heatmap = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Exigente', 'Exigente'],
                y=['No Exigente', 'Exigente'],
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}"
            ))
            heatmap.update_layout(
                xaxis_title='Predicción',
                yaxis_title='Realidad',
            )
            children.append(
                html.H3("Matriz de confusión clasificación de is_demanding_client", style={'margin': '20px'})
            )
            children.append(
                dcc.Graph(
                    id='confusion-matrix',
                    figure=heatmap,
                    style={'margin': '20px'}
                )
            )
            # TODO: Mostrar el classification report
            report = model['report']
            report_df = pd.DataFrame(report).transpose().round(2).reset_index()
            report_df.rename(columns={'index': 'Clase'}, inplace=True)   

            element = html.Div([
                html.H3("📋 Reporte de clasificación", style={'margin': '20px 20px 10px 20px'}),

                dash_table.DataTable(
                    data=report_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in report_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'padding': '8px', 'textAlign': 'center'},
                    style_header={
                        'backgroundColor': '#f0f0f0',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#fafafa'
                        }
                    ],
                    page_size=10,
                ),

                html.Div([
                    html.H4("📌 Explicación de métricas de clasificación", style={"marginTop": "20px", "marginBottom": "10px"}),

                    html.Div([
                        html.P([
                            html.B("🎯 Accuracy (Exactitud): "),
                            "Proporción de todas las clasificaciones correctas, positivas y negativas."
                        ]),
                        html.P([
                            html.B("✅ Precision (Precisión): "),
                            "Proporción de las clasificaciones positivas que realmente son positivas."
                        ]),
                        html.P([
                            html.B("🔍 Recall (Sensibilidad): "),
                            "Porcentaje de casos positivos reales correctamente identificados por el modelo."
                        ]),
                        html.P([
                            html.B("⚖️ F1-Score: "),
                            "Media armónica entre precisión y recall, útil para balancear ambos, especialmente con clases desbalanceadas."
                        ]),
                        html.P([
                            html.B("📊 Support: "),
                            "Cantidad de ejemplos reales de cada clase en los datos de prueba."
                        ]),
                    ], style={"marginBottom": "15px"}),

                    html.P(
                        "En nuestro caso, es crucial un alto recall para la clase 'Exigente', "
                        "ya que no identificar a estos clientes puede afectar negativamente al negocio.",
                        style={"marginBottom": "10px"}
                    ),

                    html.P("Algunas consecuencias de falsos negativos:"),
                    html.Ul([
                        html.Li("Malas reseñas o quejas"),
                        html.Li("Insatisfacción del cliente"),
                        html.Li("Pérdida de clientes frecuentes o valiosos"),
                        html.Li("Daño a la reputación de la empresa"),
                    ], style={"marginBottom": "15px"}),

                    html.P(
                        "Preferimos que el modelo clasifique como 'Exigente' a un cliente que no lo es (falso positivo) "
                        "antes que dejar pasar un cliente 'Exigente' (falso negativo), "
                        "porque los falsos positivos se pueden manejar con atención personalizada, "
                        "mientras que los falsos negativos pueden causar mayores problemas.",
                        style={"fontStyle": "italic"}
                    ),

                ], style={
                    'backgroundColor': '#f9f9f9',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                }),

            ], style={'margin': '20px'})
            children.append(element)

            
            # Mostrar el arbol de decisión
            # --- Interactive Cytoscape Tree ---
            cyto_elements = self.sklearn_tree_to_cytoscape(
                model['model'],
                feature_names=list(model['X_train'].columns),
                class_names=['No Exigente', 'Exigente']
            )
            children.append(
                html.H3("Árbol de decisión interactivo", style={'margin': '20px'})
            )
            children.append(
                cyto.Cytoscape(
                    id='cytoscape-decision-tree',
                    elements=cyto_elements,
                    layout={'name': 'breadthfirst', 'directed': True, 'padding': 10, 'spacingFactor': 2},
                    style={'width': '80%', 'height': '500px', 'background': '#f9f9f9'},
                    stylesheet=[
                        {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '16px', 'text-wrap': 'wrap', 'text-max-width': 120}},
                        {'selector': '.leaf', 'style': {'background-color': '#4CAF50'}},
                        {'selector': '.split', 'style': {'background-color': '#2196F3'}},
                        {'selector': 'edge', 'style': {'width': 2, 'line-color': '#888'}}
                    ]
                )
            )

            regression_result = self.run_linear_regression_special_requests(processed_df)
            children.append(
                html.H3("Regresión lineal: Predicción de solicitudes especiales (total_of_special_requests)", style={'margin': '20px'})
            )
            children.append(
                html.Div([
                    html.H4("📊 Métricas de rendimiento del modelo", style={"marginBottom": "0px"}),

                    html.Div([
                        html.P([
                            html.B("🔹 MSE (Mean Squared Error): "),
                            f"{regression_result['mse']:.2f}"
                        ]),
                        html.P(
                            "Es el promedio de los cuadrados de las diferencias entre los valores reales y los valores predichos. "
                            "Mientras más bajo sea, mejor. Un valor de 0 indica predicción perfecta."
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Hr(),

                    html.Div([
                        html.P([
                            html.B("🔸 MAE (Mean Absolute Error): "),
                            f"{regression_result['mae']:.2f}"
                        ]),
                        html.P(
                            "Promedio de las diferencias absolutas entre los valores reales y los predichos. "
                            "Debe ser lo más bajo posible."
                        )
                    ], style={"marginBottom": "20px"}),

                    html.Hr(),

                    html.Div([
                        html.P([
                            html.B("🟢 R² (Coeficiente de determinación): "),
                            f"{regression_result['r2']:.2f}"
                        ]),
                        html.P(
                            "Indica qué tanto del comportamiento de lo que queremos predecir logra explicar el modelo. "
                            "Un valor de 1 es perfecto, y valores negativos indican que el modelo es peor que predecir el promedio."
                        )
                    ]),

                    dcc.Graph(
                        figure=go.Figure(
                            data=[
                                go.Scatter(
                                    x=regression_result['y_test'],
                                    y=regression_result['y_pred'],
                                    mode='markers',
                                    name='Predicción',
                                    marker=dict(size=8, color='blue', opacity=0.7),
                                    hovertemplate=(
                                        'Valor real: %{x}<br>'+
                                        'Valor predicho: %{y}<br>'+
                                        'Índice: %{customdata}'
                                    ),
                                    customdata=regression_result['y_test'].index
                                ),
                                go.Scatter(
                                    x=regression_result['y_test'],
                                    y=regression_result['y_test'],
                                    mode='lines',
                                    name='Ideal',
                                    line=dict(color='red')
                                )
                            ],
                            layout=go.Layout(
                                title='Regresión lineal: Real vs Predicho',
                                xaxis_title='Real',
                                yaxis_title='Predicho'
                            )
                        )
                    ),

                    html.P(
                        "* Estimar el número de solicitudes especiales que un cliente hará puede ayudar a la empresa a anticipar "
                        "y satisfacer mejor las necesidades del cliente, mejorando así la experiencia del cliente y aumentando la satisfacción.",
                        style={"marginTop": "20px", "fontStyle": "italic"}
                    ),

                ], style={
                    'margin': '20px',
                    'backgroundColor': '#f9f9f9',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                })
            )


            run_style['display'] = 'none'
            clear_style['display'] = 'inline-block'
            PROCESS_DATASET['data_mining_output'] = children
            return children, run_style, clear_style

        @self.app.callback(
            Output('data-mining-output', 'children', allow_duplicate=True),
            Output('run-data-mining', 'style', allow_duplicate=True),
            Output('clear-data-mining-output', 'style', allow_duplicate=True),
            Input('clear-data-mining-output', 'n_clicks'),
            State('run-data-mining', 'style'),
            State('clear-data-mining-output', 'style'),
            prevent_initial_call=True,
        )
        def clear_data_mining_output(n_clicks, run_style, clear_style):
            if n_clicks:
                if 'data_mining_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['data_mining_output']
                run_style['display'] = 'inline-block'
                clear_style['display'] = 'none'
                return [html.P("")], run_style, clear_style
            return dash.no_update, dash.no_update, dash.no_update