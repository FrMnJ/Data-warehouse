import os
from dash import Input, Output, State, dcc, html, dash_table
import dash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import matplotlib
from imblearn.over_sampling import SMOTE
matplotlib.use('Agg')
import dash_cytoscape as cyto

from info_compartida import DATAFRAMES, PROCESS_DATASET


class MineriaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        data_mining_output = PROCESS_DATASET.get('data_mining_output', [])

        return html.Div([
            html.H2("Minería de datos",
                    style={'margin': '20px'}),  
            html.Button('Ejecutar Minería de datos',
                         id='run-data-mining', 
                         n_clicks=0, 
                         style={'margin': '20px',
                                'background-color': '#4CAF50',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'text-align': 'center',
                                'text-decoration': 'none',
                                'display': 'inline-block' if not data_mining_output else 'none',
                                }),
            html.Button('Limpiar resultados Minería de datos',
                            id='clear-data-mining-output', 
                            n_clicks=0, 
                            style={'margin': '20px',
                                    'background-color': '#f44336',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'text-align': 'center',
                                    'text-decoration': 'none',
                                    'display': 'none' if not data_mining_output else 'inline-block',
                                    }),
            html.Div([
                html.H3("Resultados de la minería de datos", 
                        style={'margin': '20px 0', 'display': 'none' if not data_mining_output else 'block'}),
                dcc.Loading(
                    id="loading-data-mining",
                    type="circle",
                    children=html.Div(id='data-mining-output',
                                    children=data_mining_output if data_mining_output else [html.P("", style={'margin': '20px'})]),
                    style={'margin': '20px'}
                )
            ], id='data-mining-results-container'),
    ])


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
        df = df.drop(columns=['total_of_special_requests'])
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
                label += f"{feature_names[tree_.feature[node]]} <= {tree_.threshold[node]:.2f}\n"
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
                title='Matriz de confusión',
                xaxis_title='Predicción',
                yaxis_title='Realidad',
            )
            children.append(
                html.H3("Matriz de confusión", style={'margin': '20px'})
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
                html.H3("Reporte de clasificación", style={'margin': '20px'}),
                dash_table.DataTable(
                    data=report_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in report_df.columns],
                    style_table={'overflowX': 'auto'},
                )
            ])
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
                    style={'width': '100%', 'height': '700px', 'background': '#f9f9f9'},
                    stylesheet=[
                        {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '16px', 'text-wrap': 'wrap', 'text-max-width': 120}},
                        {'selector': '.leaf', 'style': {'background-color': '#4CAF50'}},
                        {'selector': '.split', 'style': {'background-color': '#2196F3'}},
                        {'selector': 'edge', 'style': {'width': 2, 'line-color': '#888'}}
                    ]
                )
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