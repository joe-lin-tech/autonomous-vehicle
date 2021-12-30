from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from display.app import app
from display.apps.zadar_display import layout as zadar_layout
from display.apps.nuScenes_display import layout as nuScenes_layout


def display_app():
    app.run_server(debug=True)


navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Select Dataset", header=True),
                dbc.DropdownMenuItem("ZadarLabs", href="/zadar"),
                dbc.DropdownMenuItem("nuScenes", href="/nuScenes")
            ],
            nav=True,
            in_navbar=True,
            label="Current Dataset: None",
            right=True,
            id="navbar"
        ),
    ],
    brand="Project Configs",
    brand_href="#",
    fluid=True,
    color="primary",
    dark=True,
    style=dict(paddingLeft="1.5vw", paddingRight="1.5vw")
)

app.layout = html.Div(
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(navbar),
        html.Div(id='page-content')
    ],
    style=dict(minHeight="100vh")
)

home_layout = html.Div(
    children=[
        html.H1("Project Configs Dashboard"),
        html.P("Select a dataset from the dropdown menu above to begin."),
        html.P("""The dashboard will then display the configuration options for that dataset.
            You can also navigate to the datasets page to see the datasets available."""),
        html.I(
            "The dashboard is currently in beta and is not optimized for production use."),
        html.P("If you have any questions, please contact the developer at: "),
        html.A("joe-lin-tech", href="https://github.com/joe-lin-tech"),
    ],
    style=dict(paddingLeft="2vw", paddingRight="2vw",
               paddingTop="4vw", paddingBottom="4vw", textAlign="center")
)


@app.callback(Output('navbar', 'label'),
              Output('page-content', 'children'),
              Input('url', 'pathname'))
def select_dataset(pathname):
    if pathname == '/zadar':
        return "Current Dataset: ZadarLabs", zadar_layout
    elif pathname == '/nuScenes':
        return "Current Dataset: nuScenes", nuScenes_layout
    else:
        return "Current Dataset: None", home_layout
