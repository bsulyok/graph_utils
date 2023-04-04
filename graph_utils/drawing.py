from plotly import graph_objects as go

def edge_trace(edges, coords, **kwargs):
    x_edges, y_edges = [], []
    for u, v in edges:
        x_u, y_u, x_v, y_v = *coords[u], *coords[v]
        x_edges += [x_u, x_v, None]
        y_edges += [y_u, y_v, None]
    edge_trace = go.Scattergl(
        x=x_edges,
        y=y_edges,
        mode='lines',
        **kwargs
    )
    return edge_trace


def vertex_trace(vertices, coords, **kwargs):
    x_vertices, y_vertices = zip(*[coords[v] for v in vertices])
    vertex_trace = go.Scattergl(
        x=x_vertices,
        y=y_vertices,
        mode='markers',
        **kwargs
    )
    return vertex_trace


def draw(G, coords, **kwargs):
    args = {
        'layout_width': 800,
        'layout_height': 800,
        'layout_xaxis_scaleanchor': 'y',
        'layout_xaxis_tickvals': [],
        'layout_xaxis_zeroline': False,
        'layout_yaxis_scaleanchor': 'x',
        'layout_yaxis_tickvals': [],
        'layout_yaxis_zeroline': False,
        'layout_showlegend': False,
        'layout_plot_bgcolor': 'rgba(0,0,0,0)',
        'layout_margin_l': 0,
        'layout_margin_r': 0,
        'layout_margin_t': 0,
        'layout_margin_b': 0,
        'edge_line_width': 0.5,
        'edge_line_color': 'gray',
        'edge_name': 'edges',
        'edge_hoverinfo': 'none',
        'vertex_name': 'vertices',
    }
    args.update(kwargs)
    data = [
        edge_trace(G.edges, coords, **{key[5:]: value for key, value in args.items() if key.startswith('edge_')}),
        vertex_trace(G.nodes, coords, **{key[7:]: value for key, value in args.items() if key.startswith('vertex_')})
    ]
    layout = go.Layout(**{key[7:]: value for key, value in args.items() if key.startswith('layout_')})
    return go.Figure(data=data, layout=layout)