import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from display.app import app
from model_scripts.voxel_train import voxel_train

voxelrcnn_select = html.Div(
    children=[
        html.P("batch_size"),
        dcc.Slider(min=1, max=5, value=2, step=1,
                   marks={
                       1: {'label': '1'},
                       2: {'label': '2'},
                       5: {'label': '5'}
                   }, id="voxelrcnn-batch-select"),
    ], id="voxelrcnn-select", style=dict(display="none"))

voxelnet_select = html.Div(
    children=[
        html.P("batch_size"),
        dcc.Slider(min=1, max=5, value=2, step=1,
                   marks={
                       1: {'label': '1'},
                       2: {'label': '2'},
                       5: {'label': '5'}
                   }, id="voxelnet-batch-select")
    ], id="voxelnet-select", style=dict(display="none"))

sidebar = html.Div(
    [
        html.H2("Model Options", className="display-5"),
        html.Hr(),
        html.P(
            "Select model to train.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("VoxelRCNN", href="#voxelrcnn",
                            id="voxelrcnn-nav", active=False),
                dbc.NavLink("VoxelNet", href="#voxelnet",
                            id="voxelnet-nav", active=False),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div([
            html.Hr(),
            html.P("Select model parameters", className="lead"),
            voxelrcnn_select,
            voxelnet_select,
            html.Hr(),
            dbc.Button("Start Training", id="train-button")
        ], id="zadar-param-section", style=dict(display="none")),
    ],
)

content = html.Div(id="zadar-content")

row = html.Div(
    [
        dbc.Row(html.Hr(), style=dict(width="100%")),
        dbc.Row(
            [
                dbc.Col(sidebar, width="3"),
                dbc.Col(content, width="8")
            ],
            justify="around",
            style=dict(width="100%")
        ),
        html.Br()
    ],
)

layout = html.Div([
    dcc.Location(id="window-location"),
    html.Div([
        row,
    ]),
])

@app.callback(
    Output("zadar-param-section", "style"),
    Output("zadar-content", "children"),
    Output("voxelrcnn-select", "style"),
    Output("voxelnet-select", "style"),
    Output("voxelrcnn-nav", "active"),
    Output("voxelnet-nav", "active"),
    Input("window-location", "hash"),
    Input("train-button", "n_clicks"))
def render_content(window_location, train_button):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    train = "train-button" in changed_id

    content = html.Div()

    # Set unsupervised algorithm content
    if window_location == "#voxelrcnn":
        if train:
            print("TRAINING")
            voxel_train()
        content = html.Div([
            html.Br()
        ])
        return dict(display="block"), content, dict(display="block"), dict(display="none"), True, False
    elif window_location == "#voxelnet":
        if train:
            print("TRAINING")
        content = html.Div([
            html.Br()
        ])
        return dict(display="block"), content, dict(display="none"), dict(display="block"), False, True
    else:
        return dict(display="none"), html.Div(), dict(display="none"), dict(display="none"), False, False
