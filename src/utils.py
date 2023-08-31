import pandas as pd
import numpy as np
from pathlib import Path
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import plotly.graph_objects as go

def fetch_latest_fpl_data(folder_path_str: str = '../data/fpl/') -> pd.DataFrame:
    '''Fetch most recent saved FPL data.'''
    
    folder_path = Path(folder_path_str)
    files = os.listdir(folder_path)
    # drop non-csv files (e.g. DS_Store)
    files = [file for file in files if file.endswith('.csv')]
    # sort files and pick last one
    files = np.sort(files)
    file = files[-1]
    full_path = folder_path_str + file
    latest_data = pd.read_csv(full_path, index_col=0)
    return latest_data

def calculate_performance_metrics(y_true, y_predicted, plot=True):
    '''Calculate test metrics for regression (r2, mae, mse).'''
    mae = mean_absolute_error(y_true, y_predicted)
    rmse = mean_squared_error(y_true, y_predicted, squared=False)
    r2 = r2_score(y_true, y_predicted)

    if plot:
        x0 = y_predicted
        y0 = y_true

        fig = px.scatter(
            x=x0, 
            y=y0, 
            marginal_x="histogram", 
            marginal_y="histogram",
            labels={'x':'expected points', 'y': 'actual points'},
            )
        
        fig.add_trace(
            go.Scatter(x=np.linspace(np.min(x0), np.max(x0), 100), 
                    y=np.linspace(np.min(x0), np.max(x0), 100),
                    showlegend=False,)
        )

        fig.show()

    return (mae, rmse, r2)