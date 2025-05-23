from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
from carga_tab import CargaTab
from etl_tab import ETLTab
from mineria_tab import MineriaTab
from decision_tab import DecisionTab
from exploratorio_tab import ExploratorioTab
from info_compartida import DATAFRAMES, PROCESS_DATASET


carga_tab = None
mineria_tab = None
etl_tab = None
decision_tab = None
app = Dash(name="Data warehouse",suppress_callback_exceptions=True)

# App layout
app.layout = [
    html.H1(
        "Data Warehouse",
        style={
            'textAlign': 'center',
            'marginTop': '30px',
            'marginBottom': '10px',
            'color': '#0077b6',
            'fontWeight': 'bold',
            'fontSize': '2.5rem',
            'letterSpacing': '2px',
            'fontFamily': 'Segoe UI, Arial, sans-serif',
            'textShadow': '0 2px 8px rgba(0,0,0,0.07)'
        }
    ),
    dcc.Tabs(
        id='tabs',
        value='carga',
        style={
            'margin': '20px auto',
            'borderRadius': '12px',
            'width': 'calc(100% - 40px)',
            'backgroundColor': '#f8f9fa',
            'fontFamily': 'Segoe UI, sans-serif',
            'overflow': 'hidden',
        },
        children=[
            dcc.Tab(label='📦 Carga de datos', value='carga',
                    style={
                        'backgroundColor': "#d3d3d3",
                        'color': "#333",
                        'padding': '14px 24px',
                        'borderRight': '2px solid #ffffff',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease',
                    },
                    selected_style={
                        'backgroundColor': "#0077b6",
                        'color': "#ffffff",
                        'fontWeight': '700',
                        'padding': '14px 24px',
                        'borderRight': '2px solid transparent',
                        'transition': 'all 0.3s ease',
                    }
            ),
            dcc.Tab(label='🔄 ETL', value='etl',
                    style={
                        'backgroundColor': "#d3d3d3",
                        'color': "#333",
                        'padding': '14px 24px',
                        'borderLeft': '2px solid #ffffff',
                        'borderRight': '2px solid #ffffff',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease',
                    },
                    selected_style={
                        'backgroundColor': "#0077b6",
                        'color': "#ffffff",
                        'fontWeight': '700',
                        'padding': '14px 24px',
                        'borderLeft': '2px solid transparent',
                        'borderRight': '2px solid transparent',
                        'transition': 'all 0.3s ease',
                    }
            ),
            dcc.Tab(label='📈 Análisis exploratorio', value='exploratorio'
                    , style={
                        'backgroundColor': "#d3d3d3",
                        'color': "#333",
                        'padding': '14px 24px',
                        'borderLeft': '2px solid #ffffff',
                        'borderRight': '2px solid #ffffff',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease',
                    },
                    selected_style={
                        'backgroundColor': "#0077b6",
                        'color': "#ffffff",
                        'fontWeight': '700',
                        'padding': '14px 24px',
                        'borderLeft': '2px solid transparent',
                        'borderRight': '2px solid transparent',
                        'transition': 'all 0.3s ease',
                    }),
            dcc.Tab(label='📊 Minería de datos', value='mineria',
                    style={
                        'backgroundColor': "#d3d3d3",
                        'color': "#333",
                        'padding': '14px 24px',
                        'borderLeft': '2px solid #ffffff',
                        'borderRight': '2px solid #ffffff',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease',
                    },
                    selected_style={
                        'backgroundColor': "#0077b6",
                        'color': "#ffffff",
                        'fontWeight': '700',
                        'padding': '14px 24px',
                        'borderLeft': '2px solid transparent',
                        'borderRight': '2px solid transparent',
                        'transition': 'all 0.3s ease',
                    }
            ),
            dcc.Tab(label='🧠 Toma de decisiones', value='decision',
                    style={
                        'backgroundColor': "#d3d3d3",
                        'color': "#333",
                        'padding': '14px 24px',
                        'borderLeft': '2px solid #ffffff',
                        'fontWeight': '500',
                        'transition': 'all 0.3s ease',
                    },
                    selected_style={
                        'backgroundColor': "#0077b6",
                        'color': "#ffffff",
                        'fontWeight': '700',
                        'padding': '14px 24px',
                        'borderLeft': '2px solid transparent',
                        'transition': 'all 0.3s ease',
                    }
            ),
        ]
    ),
    html.Div(id='tabs-content'),
]


@callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'carga':
        return carga_tab.render()
    elif tab == 'etl':
        return etl_tab.render()
    elif tab == 'exploratorio':
        return exploratorio_tab.render()
    elif tab == 'mineria':
        return mineria_tab.render()
    elif tab == 'decision':
        return decision_tab.render()

# Run the app
if __name__ == '__main__':
    carga_tab = CargaTab(app)
    mineria_tab = MineriaTab(app)
    etl_tab = ETLTab(app)
    exploratorio_tab = ExploratorioTab(app)
    decision_tab = DecisionTab(app)
    app.run(debug=True)
