from dash import html
import dash_bootstrap_components as dbc

def sidebar():
    return html.Div(
        id="accordionSidebar",
        children=[
            html.H2([
                html.Img(className="", src="assets/logo2.jpg", style={'width': '120px'}),
                html.Span("Forecasting", id="sidebar-text-Dashboard")
            ], className="display-8 row d-flex justify-content-center align-items-center"),
            html.Hr(),
            html.P([
                html.I(className="fa fa-home fa-lg"),
                dbc.Label("Home", style={'padding-left': '0.5rem'}, id="sidebar-text-Home")
            ], style={"color": "#284fa1"}),
            dbc.Nav(
                [
                    dbc.NavLink([html.Span("Home", id="sidebar-text-Home-Nav")],
                                href="/home",
                                style={'color': "#696969"},
                                className="nav-link nav-home-link"),
                ],
                vertical=True,
                pills=True,
            ),
            dbc.Nav(
                [
                    # dbc.NavLink([html.Span("Overview", id="sidebar-text-Overview")],
                    #             href="/",
                    #             style={'color': "#696969"},
                    #             className="nav-link nav-home-link"),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            html.P([
                html.I(className="fa fa-file-excel fa-lg"),
                dbc.Label("Data Analysis", style={'padding-left': '0.5rem'}, id="sidebar-text-Components")
            ], style={"color": "#284fa1"}),
            dbc.Nav(
                [
                    dbc.NavLink(html.Span("GDP National Accounts", id="sidebar-text-GDP"),
                                href="/gdp",
                                style={'color': "#696969"},
                                className="nav-link nav-home-link",
                                id="GDP"),
                    dbc.NavLink([html.Span("Consumer Price Index", id="sidebar-text-CPI")],
                                href="/cpi",
                                style={'color': "#696969"},
                                className="nav-link nav-home-link",
                                id="CPI"),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            # html.P([
            #     html.I(className="fa fa-question-circle fa-lg"),
            #     dbc.Label("AI Assistant", style={'padding-left': '0.5rem'}, id="sidebar-text-machine")
            # ], style={"color": "#284fa1"}),
            # dbc.Nav(
            #     [
            #         dbc.NavLink(html.Span("Chat with the data", id="sidebar-text-chat"),
            #                     href="/chat",
            #                     style={'color': "#696969"},
            #                     className="nav-link nav-home-link",
            #                     id="LLM"),
            #     ],
            #     vertical=True,
            #     pills=True,
            # ),
            # html.Hr(),
        html.P([
            html.I(className="fa fa-table fa-lg"),
            dbc.Label("Model", style={'padding-left': '0.5rem'}, id="sidebar-text-data")
            ], style={"color": "#284fa1"}),
        dbc.Nav(
            [
                dbc.NavLink(html.Span("Forecasting using ML-Models ", id="sidebar-text-view"),
                            href="/ml_graph",
                            style={'color': "#696969"},
                            className="nav-link nav-home-link",
                            id="forecasting-link"),
                # dbc.NavLink(html.Span("Benchmark-Model", id="sidebar-text-pivot"),
                #             href="/pivottable",
                #             style={'color': "#696969"},
                #             className="nav-link nav-home-link",
                #             id="Pivot-Table"),
            ],
            vertical=True,
            pills=True,
        ),
        ], 
        className="navbar-nav sidebar sidebar-dark accordion ", 
        style={
            'padding': '0.5rem',
            "position": "fixed",
            "background-color": "#fff",  # Dark background color
            "border": "1px solid #444",
            "height": "100vh",
            "overflow-y": "auto",
            "scroll-behavior": "smooth",
            "width": "150px",  # Smaller width
        }
    )
    
