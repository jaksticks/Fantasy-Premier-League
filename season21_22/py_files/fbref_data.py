"""
Web scraper for fetching player data from fbref.com
"""

import pandas
import requests

def fetch_player_data():
    
    reply = requests.get("https://fbref.com/en/comps/9/stats/Premier-League-Stats")
    
    # remove strings that make part of the HTML into a comment section
    # the relevant part is commented out, so removing comment strings makes them "visible" again
    my_text = reply.text.replace('<!--', '')
    my_text = my_text.replace('-->', '')

    # now we can read the relevant table with read_html
    data_list = pandas.read_html(my_text)
    players = data_list[2]
    
    # edit column names that have unnamed main headers
    new_columns = [('General',col[1]) if 'Unnamed' in col[0] \
                   else col for col in players.columns]
    players.columns = pandas.MultiIndex.from_tuples(new_columns)
    # remove unnecessary rows
    ix_to_remove = players[players['General']['Squad']=='Squad'].index
    players.drop(ix_to_remove, inplace=True)
    # fix dtypes
    players = players.apply(lambda col:pandas.to_numeric(col,errors='ignore'))
    
    return players

def fetch_team_data():
    
    url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
    teamStats_web = pandas.read_html(url)
    teamStats = teamStats_web[0]
    # edit column names that have unnamed main headers
    #new_columns = [('General',col[1]) if 'Unnamed' in col[0] \
    #               else col for col in teamStats.columns]
    #teamStats.columns = pandas.MultiIndex.from_tuples(new_columns)
    
    return teamStats