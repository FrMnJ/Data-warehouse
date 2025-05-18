import os
from dash import Input, Output, State, dcc, html
import dash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import matplotlib
matplotlib.use('Agg')

from info_compartida import DATAFRAMES, PROCESS_DATASET


class MineriaTab:
    def __init__(self, app):
        self.app = app
        self.register_callbacks()

    def render(self):
        decision_tree_output = PROCESS_DATASET.get('decision_tree_output', [])
        kmeans_output = PROCESS_DATASET.get('kmeans_output', [])

        return html.Div([
            html.H2("Minería de datos",
                    style={'margin': '20px'}),  
            html.Button('Ejecutar Arbol de decisión',
                         id='run-decision-tree', 
                         n_clicks=0, 
                         style={'margin': '20px',
                                'background-color': '#4CAF50',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'text-align': 'center',
                                'text-decoration': 'none',
                                'display': 'inline-block' if not decision_tree_output else 'none',
                                }),
            html.Button('Limpiar resultados Arbol de decisión',
                            id='clear-decision-tree-output', 
                            n_clicks=0, 
                            style={'margin': '20px',
                                    'background-color': '#f44336',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'text-align': 'center',
                                    'text-decoration': 'none',
                                    'display': 'none' if not decision_tree_output else 'inline-block',
                                    }),
            html.Div([
                html.H3("Resultados del Arbol de decisión", 
                        style={'margin': '20px 0', 'display': 'none' if not decision_tree_output else 'block'}),
                dcc.Loading(
                    id="loading-decision-tree",
                    type="circle",
                    children=html.Div(id='decision-tree-output',
                                    children=decision_tree_output if decision_tree_output else [html.P("", style={'margin': '20px'})]),
                    style={'margin': '20px'}
                )
            ], id='decision-tree-results-container'),
            
            html.Button('Ejecutar KMeans',
                        id='run-kmeans',
                        n_clicks=0,
                        style={'margin': '20px',
                                'background-color': '#4CAF50',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'text-align': 'center',
                                'text-decoration': 'none',
                                'display': 'inline-block' if not kmeans_output else 'none',
                                }),
            html.Button('Limpiar resultados KMeans',
                            id='clear-kmeans-output', 
                            n_clicks=0, 
                            style={'margin': '20px',
                                    'background-color': '#f44336',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'text-align': 'center',
                                    'text-decoration': 'none',
                                    'display': 'inline-block' if kmeans_output else 'none',
                                    }),
            html.Div([
                html.H3("Resultados del KMeans", 
                        style={'margin': '20px 0', 'display': 'none' if not kmeans_output else 'block'}),
                dcc.Loading(
                    id="loading-kmeans",
                    type="circle",
                    children=html.Div(id='kmeans-output',
                                    children=kmeans_output if kmeans_output else [html.P("", style={'margin': '20px'})]),
                    style={'margin': '20px'}
                )
            ], id='kmeans-results-container'),
    ])


    def run_decision_tree(self, df, criterion,
                           splitter,max_depth, min_samples_split,  min_samples_leaf):
        # Separar las caracteristicas para el analisis
        df = df[[
        'lead_time', 'is_repeated_guest', 'previous_cancellations', 
        'booking_changes', 'deposit_type', 'total_guests', 'is_canceled'
        ]]

        # Convertir las variables categoricas a numericas
        # Usando one hot encoding
        df = pd.get_dummies(df, columns=['deposit_type'], drop_first=True)

        # Separar las variables predictoras y la variable objetivo
        X = df.drop('is_canceled', axis=1)
        y = df['is_canceled']

        # Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Entrenar el modelo
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        model.fit(X_train, y_train)
        # Evaluar el modelo
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return  model, X_train, X_test, y_train, y_test, y_pred

    def iterate_decision_tree(self, df):
        # Definimos los parametros a usar
        criterion = ['gini', 'entropy']
        splitter = ['best', 'random']
        max_depth = [None, 5, 10, 15]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        # Guardamos los resultados de las iteraciones
        results = []
        for sl in min_samples_leaf:
            for s in min_samples_split:
                for md in max_depth:
                    for sp in splitter:
                        for cr in criterion:
                            model, X_train, X_test, y_train, y_test, y_pred = self.run_decision_tree(
                                df, cr, sp, md, s, sl)
                            accuracy = model.score(X_test, y_test)
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
                                'accuracy': accuracy
                            })
        # Iteramos sobre las resultados para obtener el mejor
        best_model = results[0] 
        for result in results:
            if best_model['accuracy'] < result['accuracy']:
                best_model = result
        return best_model
    
    def run_kmeans(self, df, n_clusters, max_iter, algorithm, init):
        # Variables a usar para el clustering
        features = ['lead_time', 'total_guests', 'adr', 'previous_cancellations', 'total_nights']
        # Eliminamos columnas que no son necesarias
        df_cluster = df[features].copy()
        # Normalizamos los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)
        # Convertir a DataFrame para poder agregar la columna de cluster
        X_scaled_df = pd.DataFrame(X_scaled, columns=df_cluster.columns, index=df_cluster.index)
        # Aplicamos el modelo
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                         algorithm=algorithm, init=init)
        kmeans.fit(X_scaled_df)
        X_scaled_df['cluster'] = kmeans.labels_
        return X_scaled_df, kmeans
    

    def iterate_kmeans(self, df):
        # Parámetros a probar
        n_clusters_options = [2, 3, 4, 5]
        max_iter_options = [100, 200]
        algorithm_options = ['lloyd', 'elkan']
        init_options = ['k-means++', 'random']

        best_result = None
        best_inertia = float('inf')
        results = []

        for n_clusters in n_clusters_options:
            for max_iter in max_iter_options:
                for algorithm in algorithm_options:
                    for init in init_options:
                        try:
                            df_clustered, kmeans_model = self.run_kmeans(
                                df, n_clusters, max_iter, algorithm, init
                            )
                            inertia = kmeans_model.inertia_
                            result = {
                            'model': kmeans_model,
                            'df_clustered': df_clustered,
                            'n_clusters': n_clusters,
                            'max_iter': max_iter,
                            'algorithm': algorithm,
                            'init': init,
                            'inertia': inertia
                        }
                            results.append(result)

                            if inertia < best_inertia:
                                best_inertia = inertia
                                best_result = result
                        except Exception as e:
                            print(f"Error with params: n_clusters={n_clusters}, max_iter={max_iter}, "
                              f"algorithm={algorithm}, init={init}")
                            print(e)

        return best_result

    def register_callbacks(self):
        @self.app.callback(
            Output('decision-tree-output', 'children', allow_duplicate=True),
            Output('run-decision-tree', 'style', allow_duplicate=True),
            Output('clear-decision-tree-output', 'style', allow_duplicate=True),
            Input('run-decision-tree', 'n_clicks'),
            State('run-decision-tree', 'style'),
            State('clear-decision-tree-output', 'style'),
            prevent_initial_call=True,
        )
        def run_decision_tree_process(n_clicks, run_style, clear_style):
            print("Ejecutando el arbol de decisión")
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
                x=['No cancelado', 'Cancelado'],
                y=['No cancelado', 'Cancelado'],
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
            # Mostrar el arbol de decisión
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model['model'], feature_names=model['X_train'].columns, filled=True, ax=ax)
            plt.title('Árbol de decisión')  
            plt.tight_layout()
            plt.savefig('decision_tree.png')
            plt.close(fig)
            import base64
            with open('decision_tree.png', 'rb') as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode()
            img_src = f'data:image/png;base64,{img_b64}'
            children.append(
                html.H3("Árbol de decisión", style={'margin': '20px'})
            )
            children.append(
                html.Img(src=img_src, style={'width': '100%', 'height': 'auto'})
            )
            run_style['display'] = 'none'
            clear_style['display'] = 'inline-block'
            PROCESS_DATASET['decision_tree_output'] = children
            return children, run_style, clear_style

        @self.app.callback(
            Output('decision-tree-output', 'children', allow_duplicate=True),
            Output('run-decision-tree', 'style', allow_duplicate=True),
            Output('clear-decision-tree-output', 'style', allow_duplicate=True),
            Input('clear-decision-tree-output', 'n_clicks'),
            State('run-decision-tree', 'style'),
            State('clear-decision-tree-output', 'style'),
            prevent_initial_call=True,
        )
        def clear_decision_tree_output(n_clicks, run_style, clear_style):
            if n_clicks:
                if 'decision_tree_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['decision_tree_output']
                if os.path.exists('decision_tree.png'):
                    os.remove('decision_tree.png')
                run_style['display'] = 'inline-block'
                clear_style['display'] = 'none'
                return [html.P("")], run_style, clear_style
            return dash.no_update, dash.no_update, dash.no_update
        
        @self.app.callback(
            Output('kmeans-output', 'children', allow_duplicate=True),
            Output('run-kmeans', 'style', allow_duplicate=True),
            Output('clear-kmeans-output', 'style', allow_duplicate=True),
            Input('run-kmeans', 'n_clicks'),
            State('run-kmeans', 'style'),
            State('clear-kmeans-output', 'style'),
            prevent_initial_call=True,
        )
        def run_kmeans_process(n_clicks, run_style, clear_style):
            print("Ejecutando el KMeans")
            if n_clicks is None:
                print("No se ha hecho clic en el botón")
                return [html.P("")], dash.no_update, dash.no_update

            if 'processed_data' not in DATAFRAMES:
                print("No hay datos procesados")
                return [html.P("No hay datos procesados. Por favor, ejecute el ETL en  la pestaña 'ETL'.", 
                            style={'color': 'red', 'fontWeight': 'bold', 'margin': '20px'})], dash.no_update, dash.no_update
            
            processed_df = DATAFRAMES['processed_data'] 
            print("Iterando sobre parametros del KMeans")
            model = self.iterate_kmeans(processed_df)

            children = []
            if model is None:
                print("No se pudo encontrar un modelo KMeans válido.")
                children.append(html.P("No se pudo encontrar un modelo KMeans válido. Por favor, revise los datos o los parámetros.", style={'color': 'red', 'fontWeight': 'bold', 'margin': '20px'}))
                return children, run_style, clear_style
            print("Mostrando resultados del KMeans")
            # Mostrar el clustering
            fig = plt.figure(figsize=(12, 8))
            sns.scatterplot(data=model['df_clustered'], x='previous_cancellations', y='adr', hue='cluster', palette='Set1')
            plt.title('Clustering KMeans')  
            plt.tight_layout()
            plt.savefig('kmeans.png')
            plt.close(fig)
            import base64
            with open('kmeans.png', 'rb') as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode()
            img_src = f'data:image/png;base64,{img_b64}'
            children.append(
                html.H3("KMeans Clustering", style={'margin': '20px'})
            )
            children.append(
                html.Img(src=img_src, style={'width': '100%', 'height': 'auto'})
            )
            
            run_style['display'] = 'none'
            clear_style['display'] = 'inline-block'
            
            PROCESS_DATASET['kmeans_output'] = children
            return children, run_style, clear_style
        
        @self.app.callback(
            Output('kmeans-output', 'children', allow_duplicate=True),
            Output('run-kmeans', 'style', allow_duplicate=True),
            Output('clear-kmeans-output', 'style', allow_duplicate=True),
            Input('clear-kmeans-output', 'n_clicks'),
            State('run-kmeans', 'style'),
            State('clear-kmeans-output', 'style'),
            prevent_initial_call=True,
        )
        def clear_kmeans_output(n_clicks, run_style, clear_style):
            if n_clicks:
                if 'kmeans_output' in PROCESS_DATASET:
                    del PROCESS_DATASET['kmeans_output']
                if os.path.exists('kmeans.png'):
                    os.remove('kmeans.png')
                run_style['display'] = 'inline-block'
                clear_style['display'] = 'none'
                return [html.P("")], run_style, clear_style
            return dash.no_update, dash.no_update, dash.no_update

