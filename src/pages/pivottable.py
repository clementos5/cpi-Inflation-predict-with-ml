import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'CPI_time_series_September_2023.xlsm'
df = pd.read_excel(file_path, sheet_name='All Rwanda', engine='openpyxl')

# Drop the row where index is 'Weights'
df = df[df.index != 'Weights']

# Assuming 'Date' is not yet set as the index
df.set_index('Date', inplace=True)

# Selecting 'GENERAL INDEX (CPI)' column
cpi_series = df['GENERAL INDEX (CPI)']

# Ensure index is datetime type
cpi_series.index = pd.to_datetime(cpi_series.index, format='%Y-%m-%d', errors='coerce')  # Handle errors by coercing to NaT

# Remove rows with NaT (if any)
cpi_series = cpi_series[~cpi_series.index.isna()]

# Splitting data into training and testing sets
train_size = int(len(cpi_series) * 0.8)
train_data = cpi_series[:train_size]
test_data = cpi_series[train_size:]

# Defining machine learning models
models = {
    'LASSO': Lasso(),
    'KNN': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Ridge': Ridge()
}

def generate_forecast_plot_for_model(model_name):
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase width for space
    
    model = models[model_name]
    
    # Train the model
    model.fit(np.arange(len(train_data)).reshape(-1, 1), train_data)
    
    # Make forecasts
    forecast = model.predict(np.arange(len(train_data), len(cpi_series)).reshape(-1, 1))
    
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(train_data, model.predict(np.arange(len(train_data)).reshape(-1, 1))))
    rmse_test = np.sqrt(mean_squared_error(test_data, forecast))
    
    # Plot data
    ax.plot(cpi_series.index[:train_size], train_data, label='Training Data', color='blue')
    ax.plot(cpi_series.index[train_size:], test_data, label='Test Data', color='green')
    ax.plot(cpi_series.index[train_size:], forecast, label='Forecasted Values', linestyle='--', color='red')
    ax.set_title(f'{model_name} Forecasting for CPI')
    ax.set_xlabel('Date')
    ax.set_ylabel('CPI')
    ax.grid(True)
    
    # Place legend outside of the plot area, slightly up
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title='Legend')
    
    # Add RMSE to the plot in a separate area
    fig.subplots_adjust(right=0.7)  # Adjust the right margin to make space for the legend and RMSE text
    ax.text(1.05, 0.1, f'Train RMSE: {rmse_train:.6f}\nTest RMSE: {rmse_test:.6f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Save plot to a buffer and encode it as base64 for rendering in Dash
    buffer = BytesIO()
    plt.tight_layout(pad=3.0)
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    
    # Convert the image in the buffer to base64 encoding
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Close the plot to free memory
    plt.close(fig)
    
    # Return the base64 image
    return f"data:image/png;base64,{image_base64}"


# Initialize the Dash app
app = dash.Dash(__name__)

# Layout with a dropdown to select the model and a div to display the image
app.layout = html.Div([
    html.H1('CPI Forecasting Dashboard'),
    
    # Dropdown to select a model
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model, 'value': model} for model in models.keys()],
        value='LASSO',  # Default value
        clearable=False,  # Prevent clearing the selection
        style={'width': '50%'}  # Reduce width of dropdown to 50%
    ),
    
    # Title for the selected model
    html.H2(id='model-title'),
    
    # Div to display the forecast plot
    html.Div(id='output-image')
])

# Callback to update the image and title based on the selected model
@app.callback(
    [Output('model-title', 'children'),
     Output('output-image', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_output(selected_model):
    # Generate the plot for the selected model and encode it in base64
    image_base64 = generate_forecast_plot_for_model(selected_model)
    # Set the title to indicate the selected model
    model_title = f"Model Selected: {selected_model}"
    return model_title, html.Img(src=image_base64)
