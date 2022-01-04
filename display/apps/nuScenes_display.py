import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from display.app import app

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box, view_points
from sklearn.cluster import OPTICS, DBSCAN
from algorithms.gbdbscan import GBDBSCAN
import hdbscan as hd

from display.utils.nuScenes_utils import overlay_clusters_on_image

import numpy as np

nusc = NuScenes(version='v1.0-mini',
                dataroot='dataset/nuscenesDataset', verbose=True)
my_sample = nusc.sample[0]
radar_front_data = nusc.get('sample_data', my_sample['data']['RADAR_FRONT'])
RadarPointCloud.disable_filters()
pc = RadarPointCloud.from_file(
    'dataset/nuscenesDataset/' + radar_front_data['filename'])
radar_points = np.transpose(pc.points)
radar_data_attributes = ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp',
                         'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 'vy_rms']
_, boxes, _ = nusc.get_sample_data(
    my_sample['data']['RADAR_FRONT'], use_flat_vehicle_coordinates=False)
processed_radar_points = np.hstack(
    (radar_points[:, 0:2], radar_points[:, 8:10]))


def top_scatter(colors, name):
    return go.Figure(
        data=go.Scatter(
            x=-processed_radar_points[:, 1],
            y=processed_radar_points[:, 0],
            mode="markers",
            marker=dict(
                color=colors,
                colorscale="rainbow"
            )
        ),
        layout=go.Layout(
            title=go.layout.Title(
                text=name + " Plot of nuScenes Dataset"),
            height=750,
            showlegend=False
        )
    )


def draw_boxes(scatter):
    for b in boxes:
        box = view_points(b.corners(), np.eye(4), normalize=False)[:2, :]
        scatter.add_trace(go.Scatter(
            x=[-box[1][0], -box[1][1], -box[1][5], -box[1][4], -box[1][3]],
            y=[box[0][0], box[0][1], box[0][5], box[0][4], box[0][3]],
            mode="lines",
            line_color='#%02x%02x%02x' % nusc.colormap[b.name]
        ))


def overlay(labels):
    points, colors, im = overlay_clusters_on_image(
        nusc=nusc, sample_token=my_sample["token"], labels=labels)
    overlay = go.Figure(data=px.imshow(im))
    overlay.add_trace(go.Scatter(
        x=points[0, :],
        y=points[1, :],
        mode="markers",
        marker=dict(
            color=colors,
            colorscale="rainbow"
        ),
    ))
    return overlay


dbscan_select = html.Div(
    children=[
        html.P("eps"),
        dcc.Slider(min=1, max=10, value=3, step=1,
                   marks={
                       1: {'label': '1 eps'},
                       5: {'label': '5 eps'},
                       10: {'label': '10 eps'}
                   }, id="dbscan-eps-select"),
        html.Br(),
        html.P("min-pts"),
        dcc.Slider(min=1, max=10, value=3, step=1,
                   marks={
                       1: {'label': '1 pts'},
                       5: {'label': '5 pts'},
                       10: {'label': '10 pts'}
                   }, id="dbscan-min-select")
    ], id="dbscan-select", style=dict(display="none"))

optics_select = html.Div(
    children=[
        html.P("min_pts"),
        dcc.Slider(min=1, max=10, value=3, step=1,
                   marks={
                       1: {'label': '1 eps'},
                       5: {'label': '5 eps'},
                       10: {'label': '10 eps'}
                   }, id="optics-min-select")
    ], id="optics-select", style=dict(display="none"))

gbdbscan_select = html.Div(
    children=[
        html.P("g"),
        dcc.Slider(min=1, max=5, value=1, step=1,
                   marks={
                       1: {'label': '1'},
                       3: {'label': '3'},
                       5: {'label': '5'}
                   }, id="gbdbscan-g-select"),
        html.Br(),
        html.P("f"),
        dcc.Slider(min=1, max=10, value=2, step=1,
                   marks={
                       1: {'label': '1'},
                       5: {'label': '5'},
                       10: {'label': '10'}
                   }, id="gbdbscan-f-select"),
        html.Br(),
        html.P("k"),
        dcc.Slider(min=0.1, max=1, value=0.3, step=0.1,
                   marks={
                       0.1: {'label': '0.1'},
                       0.5: {'label': '0.5'},
                       1: {'label': '1'}
                   }, id="gbdbscan-k-select"),
        html.Br(),
        html.P("num_r"),
        dcc.Slider(min=1, max=100, value=75, step=1,
                   marks={
                       1: {'label': '1'},
                       50: {'label': '50'},
                       100: {'label': '100'}
                   }, id="gbdbscan-num_r-select"),
        html.Br(),
        html.P("num_t"),
        dcc.Slider(min=1, max=30, value=18, step=1,
                   marks={
                       1: {'label': '1'},
                       15: {'label': '15'},
                       30: {'label': '30'}
                   }, id="gbdbscan-num_t-select")
    ], id="gbdbscan-select", style=dict(display="none"))

hdbscan_select = html.Div(
    children=[
        html.P("min_cluster_size"),
        dcc.Slider(min=1, max=10, value=5, step=1,
                   marks={
                       1: {'label': '1'},
                       5: {'label': '5'},
                       10: {'label': '10'}
                   }, id="hdbscan-min-cluster-size-select")
    ], id="hdbscan-select", style=dict(display="none"))

sidebar = html.Div(
    [
        html.H2("Clustering Options", className="display-5 fw-bold"),
        html.Hr(),
        html.P(
            "Select algorithm to display.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("DBSCAN", href="#dbscan",
                            id="dbscan-nav", active=False),
                dbc.NavLink("OPTICS", href="#optics",
                            id="optics-nav", active=False),
                dbc.NavLink("GBDBSCAN", href="#gbdbscan",
                            id="gbdbscan-nav", active=False),
                dbc.NavLink("HDBSCAN", href="#hdbscan",
                            id="hdbscan-nav", active=False),
            ],
            vertical=True,
            pills=True,
        ),
        html.Div([
            html.Hr(),
            html.P("Select algorithm parameters", className="lead"),
            dbscan_select,
            optics_select,
            gbdbscan_select,
            hdbscan_select
        ], id="nuscenes-param-section", style=dict(display="none")),
    ],
)

content = html.Div(id="nuscenes-content")

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
    Output("nuscenes-param-section", "style"),
    Output("nuscenes-content", "children"),
    Output("dbscan-select", "style"),
    Output("optics-select", "style"),
    Output("gbdbscan-select", "style"),
    Output("hdbscan-select", "style"),
    Output("dbscan-nav", "active"),
    Output("optics-nav", "active"),
    Output("gbdbscan-nav", "active"),
    Output("hdbscan-nav", "active"),
    Input("window-location", "hash"),
    Input("dbscan-eps-select", "value"),
    Input("dbscan-min-select", "value"),
    Input("optics-min-select", "value"),
    Input("gbdbscan-g-select", "value"),
    Input("gbdbscan-f-select", "value"),
    Input("gbdbscan-k-select", "value"),
    Input("gbdbscan-num_r-select", "value"),
    Input("gbdbscan-num_t-select", "value"),
    Input("hdbscan-min-cluster-size-select", "value"))
def render_content(window_location,
                   dbscan_eps, dbscan_min_samples,
                   optics_min_samples,
                   gbdbscan_g, gbdbscan_f, gbdbscan_k, gbdbscan_num_r, gbdbscan_num_t,
                   hdbscan_min_cluster_size):

    # Set unsupervised algorithm content
    if window_location == "#dbscan":
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
            processed_radar_points)
        dbscan_scatter = top_scatter(dbscan.labels_ + 1, "DBSCAN")
        draw_boxes(dbscan_scatter)
        dbscan_overlay = overlay(dbscan.labels_)
        content = html.Div([
            dcc.Graph(figure=dbscan_scatter),
            # dcc.Graph(figure=dbscan_overlay),
            # html.Br()
        ])
        return dict(display="block"), content, dict(display="block"), dict(display="none"), \
            dict(display="none"), dict(
                display="none"), True, False, False, False
    elif window_location == "#optics":
        optics = OPTICS(min_samples=optics_min_samples).fit(
            processed_radar_points)
        optics_scatter = top_scatter(optics.labels_ + 1, "OPTICS")
        draw_boxes(optics_scatter)
        optics_overlay = overlay(optics.labels_)
        content = html.Div([
            dcc.Graph(figure=optics_scatter),
            # dcc.Graph(figure=optics_overlay),
            # html.Br()
        ])
        return dict(display="block"), content, dict(display="none"), dict(display="block"), \
            dict(display="none"), dict(
                display="none"), False, True, False, False
    elif window_location == "#gbdbscan":
        polar_radar_data = []
        max_r = processed_radar_points[0][0]
        for p in processed_radar_points:
            r = np.sqrt(p[0] ** 2 + p[1] ** 2)
            max_r = max(max_r, r)
            t = np.arctan(p[1] / p[0])
            polar_radar_data.append([r, t])
        polar_radar_data = np.array(polar_radar_data)
        gbdbscan = GBDBSCAN(g=gbdbscan_g, f=gbdbscan_f, k=gbdbscan_k,
                            num_r=gbdbscan_num_r, max_r=max_r + 5, num_t=gbdbscan_num_t).fit(polar_radar_data)
        gbdbscan_scatter = top_scatter(
            np.array(gbdbscan.labels_) + 1, "GBDBSCAN")
        draw_boxes(gbdbscan_scatter)
        gbdbscan_overlay = overlay(gbdbscan.labels_)
        content = html.Div([
            dcc.Graph(figure=gbdbscan_scatter),
            # dcc.Graph(figure=gbdbscan_overlay),
            # html.Br()
        ])
        return dict(display="block"), content, dict(display="none"), dict(display="none"), \
            dict(display="block"), dict(
                display="none"), False, False, True, False
    elif window_location == "#hdbscan":
        hdbscan = hd.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size, gen_min_span_tree=True).fit(processed_radar_points)
        hdbscan_scatter = top_scatter(np.array(hdbscan.labels_) + 1, "HDBSCAN")
        draw_boxes(hdbscan_scatter)
        hdbscan_overlay = overlay(hdbscan.labels_)
        content = html.Div([
            dcc.Graph(figure=hdbscan_scatter),
            # dcc.Graph(figure=hdbscan_overlay),
            # html.Br()
        ])
        return dict(display="block"), content, dict(display="none"), dict(display="none"), \
            dict(display="none"), dict(
                display="block"), False, False, False, True
    else:
        return dict(display="none"), html.Div(), dict(display="none"), dict(display="none"), \
            dict(display="none"), dict(
                display="none"), False, False, False, False
