"""
Scripts for downloading and preprocessing fbref.com data on Big 5 European 
Leagues Stats.

@author: jaakkotoivonen
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

def download_from_url(file_name:str, folder_path:str, url:str, table=0) -> pd.DataFrame:
    """Download and save data from given url.
    
    Parameters
    ----------
    file_name:str
        Filename for saving data.
    folder_path:str
       Path to folder for saving data to.
    url:str
        Url for fetching data. 
    table:int (default 0)
        0 for standard data, 1 for "vs" data (opponent data)
        
    Returns
    -------
    data: pandas DataFrame
        Dataframe of data from the given url.
    """
    
    data = pd.read_html(url)
    my_dataframe = data[table]
    # edit column names that have unnamed main headers
    new_columns = [('General',col[1]) if 'Unnamed' in col[0] \
                   else col for col in my_dataframe.columns]
    my_dataframe.columns = pd.MultiIndex.from_tuples(new_columns)
    # remove unnecessary rows
    ix_to_remove = my_dataframe[my_dataframe['General']['Squad']=='Squad'].index
    my_dataframe.drop(ix_to_remove, inplace=True)
    # fix dtypes
    my_dataframe = my_dataframe.apply(lambda col:pd.to_numeric(col,errors='ignore'))
    # save dataframe
    my_dataframe.to_csv(f'{folder_path}/{file_name}.csv')
    
    return my_dataframe

def download_player_data(folder_path:str):
    """Download player data and save as multiple csv-files to the given folder.
    
    Parameters
    ----------
    folder_path: str
        Path to folder for saving data to.
    """
    
    # standard data
    url='https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats'
    standard = download_from_url('standard',folder_path, url)
    print('standard downloaded.')
    
    # shooting data
    url='https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats'
    shooting = download_from_url('shooting',folder_path, url)
    print('shooting downloaded.')
    
    # passing data
    url='https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats'
    passing = download_from_url('passing',folder_path, url)
    print('passing downloaded.')
    
    # pass_types data
    url='https://fbref.com/en/comps/Big5/passing_types/players/Big-5-European-Leagues-Stats'
    pass_types = download_from_url('pass_types',folder_path, url)
    print('pass_types downloaded.')
    
    # gca data
    url='https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats'
    gca = download_from_url('gca',folder_path, url)
    print('gca downloaded.')
    
    # defensive data
    url='https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats'
    defense = download_from_url('defense',folder_path, url)
    print('defense downloaded.')
    
    # possession data
    url='https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats'
    possession = download_from_url('possession',folder_path, url)
    print('possession downloaded.')
    
    # play_time data
    url='https://fbref.com/en/comps/Big5/playingtime/players/Big-5-European-Leagues-Stats'
    play_time = download_from_url('play_time',folder_path, url)
    print('play_time downloaded.')
    
    # misc data
    url='https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats'
    misc = download_from_url('misc',folder_path, url)
    print('misc downloaded.')
    
    # create master dataframe of player data
    my_df = standard.copy()
    for df in [shooting, passing, pass_types, gca, defense, possession, play_time, misc]:
        new_columns = [col for col in df.columns if col not in my_df.columns]
        columns = [('General','Player'),('General','Squad')] + new_columns
        my_df = pd.merge(left=my_df, right=df[columns], on=[('General','Player'),('General','Squad')], how='outer')
        
    my_df.to_csv(f'{folder_path}/players_master.csv')
    print('master dataframe created.')
    
def download_team_data(folder_path:str, table=0):
    """Download team data and save as multiple csv-files to the given folder.

    Parameters
    ----------
    folder_path: str
        Path to folder for saving data to.
    table:int (default 0)
        0 for standard data, 1 for "vs" data (opponent data)
    """
    
    # standard data
    url='https://fbref.com/en/comps/Big5/stats/squads/Big-5-European-Leagues-Stats'
    standard = download_from_url('standard',folder_path, url, table=table)
    print('standard downloaded.')
    
    # shooting data
    url='https://fbref.com/en/comps/Big5/shooting/squads/Big-5-European-Leagues-Stats'
    shooting = download_from_url('shooting',folder_path, url, table=table)
    print('shooting downloaded.')
    
    # passing data
    url='https://fbref.com/en/comps/Big5/passing/squads/Big-5-European-Leagues-Stats'
    passing = download_from_url('passing',folder_path, url, table=table)
    print('passing downloaded.')
    
    # pass_types data
    url='https://fbref.com/en/comps/Big5/passing_types/squads/Big-5-European-Leagues-Stats'
    pass_types = download_from_url('pass_types',folder_path, url, table=table)
    print('pass_types downloaded.')
    
    # gca data
    url='https://fbref.com/en/comps/Big5/gca/squads/Big-5-European-Leagues-Stats'
    gca = download_from_url('gca',folder_path, url, table=table)
    print('gca downloaded.')
    
    # defensive data
    url='https://fbref.com/en/comps/Big5/defense/squads/Big-5-European-Leagues-Stats'
    defense = download_from_url('defense',folder_path, url, table=table)
    print('defense downloaded.')
    
    # possession data
    url='https://fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats'
    possession = download_from_url('possession',folder_path, url, table=table)
    print('possession downloaded.')
    
    # play_time data
    url='https://fbref.com/en/comps/Big5/playingtime/squads/Big-5-European-Leagues-Stats'
    play_time = download_from_url('play_time',folder_path, url, table=table)
    print('play_time downloaded.')
    
    # misc data
    url='https://fbref.com/en/comps/Big5/misc/squads/Big-5-European-Leagues-Stats'
    misc = download_from_url('misc',folder_path, url, table=table)
    print('misc downloaded.')
    
    # create master dataframe of player data
    my_df = standard.copy()
    for df in [shooting, passing, pass_types, gca, defense, possession, play_time, misc]:
        new_columns = [col for col in df.columns if col not in my_df.columns]
        my_df = pd.concat([my_df, df[new_columns]], axis=1, join='inner')
        
    if table==0:
        my_df.to_csv(f'{folder_path}/teams_master.csv')
        print('master dataframe created.')
    elif table==1:
        my_df.to_csv(f'{folder_path}/teams_vs_master.csv')
        print('master dataframe created.')
    else:
        print('table should be either 0 or 1!')
        
        
def get_match_report_links(url:str) -> list:
    """Get url addresses for all fbref.com match reports of a given league and season.
    
    Parameters
    ----------
    url:str
        Url for fetching data, e.g., https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures 
        
    Returns
    -------
    match_report_links:list
        List of url addresses for fetching match reports.
    """
    
    # fetch web site data and use boutiful soup to parse
    reply = requests.get(url)
    soup = BeautifulSoup(reply.text, 'html.parser')

    # find all hyperlinks from the "soup"
    links = []
    all_a = soup.find_all('a')
    for link in all_a:
        links.append(link)

    # first part of the url for all match report links
    main_site = 'https://fbref.com'
    # extract all links to match reports
    match_report_links = [main_site + links[i]['href'] for i in range(len(links)) if 'Match Report' in links[i]]

    return match_report_links

def download_match_report_data(match_report_links:list, tables_team1=[3, 4, 5, 6, 7, 8], 
               tables_team2=[10,11,12,13,14,15], save_data=False, save_path='') -> pd.DataFrame:
    """
    Download all player data from every game in match_report_links.

    Parameters
    ----------
    match_report_links : list
        List of url addresses for fetching match reports.
    tables_team1:list, optional
        Which data tables to collect from each match report for the home team.
    tables_team2:list, optional
        Which data tables to collect from each match report for the away team.
    save_data : boolean, optional
        Whether to save the resulting dataframe or not.
    save_path : TYPE, optional
        File path for saving the resulting dataframe, if save_data is True.

    Returns
    -------
    all_data_df:pd.DataFrame
        Dataframe containting all player data from every game in match_report_links.

    """
    
    # master dataframe to be created
    all_data_df = pd.DataFrame()
    tables = [tables_team1, tables_team2]
    # progress bar
    with tqdm(total=len(match_report_links)) as pbar:
        for i in range(len(match_report_links)):
            data = pd.read_html(match_report_links[i])
            # get team names
            team1 = data[2].columns[0][0]
            team2 = data[2].columns[1][0]
            for count,team in enumerate([team1, team2]):
                # collect all tables from a single fixture for one team
                my_df_all = []
                for j in tables[count]:
                    my_df = data[j]
                    # edit column names that have unnamed main headers
                    new_columns = [('General',col[1]) if 'Unnamed' in col[0] \
                                   else col for col in my_df.columns]
                    my_df.columns = pd.MultiIndex.from_tuples(new_columns)
                    # remove final row which only contains aggregated data
                    if my_df.loc[my_df.tail(1).index, ('General', 'Player')].str.contains('Player').values:
                        my_df = my_df.drop(my_df.tail(1).index)
                    # fix dtypes, i.e., convert strings to numeric values where sensible
                    my_df = my_df.apply(lambda col:pd.to_numeric(col,errors='ignore'))        
                    # collect all tables
                    my_df_all.append(my_df)
                # auxiliary df for adding some information
                aux_df = my_df_all[0]
                # add team name info
                aux_df[('General','Squad')] = team
                # add iwhether this is a home or away game
                if count==0:
                    aux_df[('General','HomeTeam')] = True
                else:
                    aux_df[('General','HomeTeam')] = False
                # add opponent info
                if count==0:
                    aux_df[('General','Opponent')] = team2
                else:
                    aux_df[('General','Opponent')] = team1
                # merge all dfs into one dataframe and remove duplicate columns
                for k in range(1,len(my_df_all)):
                     new_columns = [col for col in my_df_all[k].columns if col not in aux_df.columns]
                     columns = [('General','Player')] + new_columns
                     aux_df = pd.merge(left=aux_df, right=my_df_all[k][columns], on=[('General','Player')], how='outer')
                # add player data for one team from single fixture to master dataframe
                all_data_df = pd.concat([all_data_df, aux_df], ignore_index=True)
            # wait for three seconds before continuing so as to respect fbref.com request for not overloading servers
            time.sleep(3)
            pbar.update(1)
     
    if save_data:
        all_data_df.to_csv(save_path)
     
    return all_data_df

    

    
        