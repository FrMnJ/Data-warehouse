from dash import dcc, html

class CargaTab:
    def render(self):
        return html.Div([
            html.H3('Carga de datos'),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload Data'),
                multiple=False
            ),
        ])