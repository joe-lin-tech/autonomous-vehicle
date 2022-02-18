import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from display.app import app
from model_scripts.voxel_train import voxel_train
from display.utils.zadar_utils import box_to_corners, get_pointcloud_scatter, get_layout_plots
import os
import json
import numpy as np


voxel_transformer_select = html.Div(
    children=[
        html.P("batch_size"),
        dcc.Slider(min=1, max=5, value=2, step=1,
                   marks={
                       1: {'label': '1'},
                       2: {'label': '2'},
                       5: {'label': '5'}
                   }, id="voxel-transformer-batch-select"),
    ], id="voxel-transformer-select", style=dict(display="none"))

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
        html.H2("Model Options", className="display-5 fw-bold"),
        html.Hr(),
        html.P(
            "Select model to train.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Voxel Transformer", href="#voxeltransformer",
                            id="voxel-transformer-nav", active=False),
                dbc.NavLink("VoxelNet", href="#voxelnet",
                            id="voxelnet-nav", active=False),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div([
            html.Hr(),
            html.P("Select model parameters", className="lead"),
            voxel_transformer_select,
            voxelnet_select,
            html.Hr(),
            dbc.Button("Start Training", id="train-button")
        ], id="zadar-param-section", style=dict(display="none")),
    ],
)

file_list = os.listdir(os.path.join(os.getcwd(), "display/data"))

content = html.Div(
    # id="zadar-content",
    children=[
        dcc.Dropdown(
            id='graph-dropdown',
            options=[{'label': file, 'value': file} for file in file_list],
            placeholder="Select a Training Result File..."
        ),
        html.Div(
            id="zadar-graphs"
        )
    ]
)

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
    # dcc.Interval(
    #     id='interval-component',
    #     interval=1*1000, # in milliseconds
    #     n_intervals=0
    # )
])


def update_layout(file):
    with open(os.path.join(os.getcwd(), "display/data/" + file), "r") as f:
        data = json.load(f)
        targets = data["targets"]
        detections = data["boxes"], data["scores"]
        losses = data["losses"]
        pc_scatters = get_layout_plots(targets, detections, losses)
    pc_graphs = [dcc.Graph(figure=scatter) for scatter in pc_scatters]
    return pc_graphs


@app.callback(
    Output("zadar-param-section", "style"),
    # Output("zadar-content", "children"),
    Output("zadar-graphs", "children"),
    Output("voxel-transformer-select", "style"),
    Output("voxelnet-select", "style"),
    Output("voxel-transformer-nav", "active"),
    Output("voxelnet-nav", "active"),
    Input("window-location", "hash"),
    Input("train-button", "n_clicks"),
    Input("graph-dropdown", "value"))
def render_content(window_location, train_button, graph_dropdown):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    train = "train-button" in changed_id

    try:
        file = graph_dropdown
        pc_graphs = update_layout(file)
    except:
        pc_graphs = []

    # Set unsupervised algorithm content
    if window_location == "#voxeltransformer":
        if train:
            print("TRAINING")
            voxel_train()
        # content = html.Div([
        #     html.Br()
        # ])
        # return dict(display="block"), content, dict(display="block"), dict(display="none"), True, False
        return dict(display="block"), pc_graphs, dict(display="block"), dict(display="none"), True, False
    elif window_location == "#voxelnet":
        if train:
            print("TRAINING")
        # content = html.Div([
        #     html.Br()
        # ])
        # return dict(display="block"), content, dict(display="none"), dict(display="block"), False, True
        return dict(display="block"), pc_graphs, dict(display="none"), dict(display="block"), False, True
    else:
        # return dict(display="none"), html.Div(), dict(display="none"), dict(display="none"), False, False
        return dict(display="none"), pc_graphs, dict(display="none"), dict(display="none"), False, False
