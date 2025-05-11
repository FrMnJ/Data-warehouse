from dash import dcc, html

class ETLTab:
    def render(self):
        return html.Div([
            html.H2("ETL",
                    style={'margin': '20px'}),  
        ])
