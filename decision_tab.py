from dash import dcc, html

class DecisionTab:
    def render(self):
        return html.Div([
            html.H2("Toma de decisiones",
                    style={'margin': '20px'}),  
        ])