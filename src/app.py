from dash import Dash, html, callback, Output, Input, page_container  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import logging
from sidebar import sidebar
from topbar import topbar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Create the Dash app
app = Dash(__name__,
                use_pages=True,  # Ensure use_pages=True is enabled
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      "assets/vendor/fontawesome-free/css/all.min.css",
                                      "/assets/style.css",
                                      "/assets/css/sb-admin-2.min.css"])
server = app.server
app.title = 'Inflation-Model'

# Home page design
content = html.Div(
    id="wrapper",
    children=[
        # Sidebar
        dbc.Collapse(
            sidebar(),
            id="collapse-sidebar",
            is_open=True,
        ),
        # Toggle Sidebar Button
        html.Div(
            dbc.Button(html.I(className="fas fa-bars"), id="btn-sidebar", color="primary", className="mb-3"),
            id="toggle-button-container"
        ),
        # Include the topbar function here
        topbar(),
        html.Hr(),  # Add a horizontal line below the topbar

        # Content Wrapper
        html.Div(
            [
                # Main Content
                html.Div([
                    # Begin Page Content
                    html.Div([page_container], className="container-fluid", style={'margin-top': '80px','z-index':'-2'})
                ], id="content")
            ],
            id="content-wrapper",
            className="d-flex flex-column",
            style={'margin-left': '0px', 'padding': '2rem 1rem','margin-top': '100px'}  # Adjusted margin to fit the sidebar
        )
    ]
)

# Dash App main layout to display the pages
app.layout = html.Div([
    content,
], id="page-top", className="py-2")

# Callback to update the position of the toggle button
@app.callback(
    Output("toggle-button-container", "style"),
    [Input("collapse-sidebar", "is_open")]
)
def update_button_position(is_open):
    margin_left = "210px" if is_open else "10px"  # Adjust margin based on sidebar state
    return {'position': 'fixed', 'top': '10px', 'left': margin_left}

# Callback to toggle the sidebar
@app.callback(
    [Output("collapse-sidebar", "is_open"), Output("content-wrapper", "style")],
    [Input("btn-sidebar", "n_clicks")],
    [Input("collapse-sidebar", "is_open")]
)
def toggle_sidebar(n_clicks, is_open):
    if n_clicks:
        is_open = not is_open
        margin_left = "18rem" if is_open else "0"
        return is_open, {"margin-left": margin_left, "padding": "2rem 1rem"}
    return is_open, {"margin-left": "18rem", "padding": "2rem 1rem"}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
