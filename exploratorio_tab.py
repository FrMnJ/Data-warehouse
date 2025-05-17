from dash import html, dcc

class ExploratorioTab:
    def __init__(self, app):
       self.app = app 
       self.register_callbacks()

    def render(self):
        return html.Div([
            html.H2("Exploratorio",
                    style={'margin':
                    '20px'}),
        ])
    
    def register_callbacks(self):
        pass