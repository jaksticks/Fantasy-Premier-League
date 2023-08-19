import pandas as pd
import numpy as np
from pathlib import Path
import os

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