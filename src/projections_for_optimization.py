'''
This function is used to create a csv file that includes point projections in a format similar 
to fplreview.com. This csv file can be used for making advanced team optimization plans using e.g.
Sertalp B. Cay's repo fpl-optimization.

Example usage: 
python src/projections_for_optimization.py --latest_gameweek 5
'''

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import requests

def adjust_points_via_weights(projections, projections_pivot, player_id, gameweeks, weights):
    '''
    Adjust projected player points via gameweek specific weights indicating probability of playing.

    Input:
    player_id (int): Player ID number.
    gameweeks (list): List of gameweek numbers to be adjusted.
    weights (list): Weights for adjusting points for each gameweek given in parameter gameweeks.
    '''
    
    point_cols = [col for col in projections_pivot.columns if 'Pts' in col]
    if len(projections.loc[projections.id==player_id]) > 0:
        player_name = projections.loc[projections.id==player_id, 'name'].unique().item()
        logging.info(f'Name: {player_name}')
        logging.info('Point predictions before adjustment:')
        logging.info(projections_pivot.loc[projections_pivot.ID==player_id, point_cols].values)

        # adjust points via probability of playing
        for i in range(len(gameweeks)):
            projections_pivot.loc[projections_pivot.ID==player_id, f'{gameweeks[i]}_Pts'] = \
                weights[i]*projections_pivot.loc[projections_pivot.ID==player_id, f'{gameweeks[i]}_Pts']
            
        logging.info('Point predictions after adjustment:')
        logging.info(projections_pivot.loc[projections_pivot.ID==player_id, point_cols].values)
    else:
        logging.info(f'No player with ID {player_id} found in projections!')


def initial_projection_data(latest_gameweek: int, current_season: str, season_folder):

    logging.info("Starting to create projection data.")

    # projection data file path
    filepath = Path(f'{season_folder}/data/predictions/gameweek{latest_gameweek}.csv')
    projections = pd.read_csv(filepath, index_col=0)
    if 'id' in projections.columns:
        projections = projections.drop('id', axis=1)

    # load fpl data
    if latest_gameweek>0:
        filepath = Path(f'{season_folder}/data/fpl_df.csv')
    elif latest_gameweek==0:
        filepath = Path(f'{season_folder}/data/fpl_df_preseason_online.csv')
    else:
        raise Exception('Check latest_gameweek!')
    fpl_df = pd.read_csv(filepath, index_col=0, low_memory=False)
    fpl_df = fpl_df[fpl_df.season==current_season]

    # GET PLAYER IDS AND NAMES FROM FPL DATA TO PROJECTIONS
    df = fpl_df.groupby('name').last().reset_index()[['id', 'name', 'element_type', 'points_per_game', 'total_points',]]    
    df['games_played'] = np.round(np.where(df['points_per_game']!=0, df['total_points'] / df['points_per_game'], 0),0)
    # drop duplicate players (some players get new spelling for their name during the season causing duplicates)
    duplicate_ids = df.loc[df.id.duplicated(), 'id'].unique()
    for id in duplicate_ids:
        ix = df.loc[df.id==id, 'games_played'].idxmin()
        df = df.drop(ix)
    # merge id info to projections
    projections = projections.merge(df[['id', 'name']], on='name', how='left')
    # add xMins variable that is needed later
    projections['xMins'] = 90

    # PIVOT DATA
    projections_pivot = (
        projections
        .pivot_table(
        columns=['gameweek',],
        index='id', 
        values=['expected_points','xMins'], 
        aggfunc='sum'
        )
    )

    new_cols = []
    for col in projections_pivot.columns:
        if col[0] == 'expected_points':
            new_col = str(col[1]) + '_Pts'
            new_cols.append(new_col)
        elif col[0] == 'xMins':
            new_col = str(col[1]) + '_xMins'
            new_cols.append(new_col)

    projections_pivot.columns = new_cols
    projections_pivot = projections_pivot.reset_index()
    projections_pivot = projections_pivot.rename(columns={'id':'ID'})

    position_dict = {1:'G', 2:'D', 3:'M', 4:'F'}
    df['Pos'] = df['element_type'].map(position_dict)
    df.rename(columns={'id':'ID'}, inplace=True)   
    projections_pivot = pd.merge(projections_pivot, df[['ID', 'Pos']], on='ID', how='left')

    logging.info("Initial projection data creation complete.")
    return projections, projections_pivot

def adjust_projections(projections, projections_pivot, latest_gameweek: int):

    logging.info("Adjusting points for unavailable players or players with uncertain availability")
    # fetch online data
    fpl_online_data = json.loads(requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').text)
    fpl_online_df = pd.DataFrame(fpl_online_data['elements'])

    # get play probabilities for players with injuries
    play_probabilities = []
    for _, row in fpl_online_df.iterrows():
        news = row['news']
        news_splitted = news.split('-')
        if len(news_splitted) > 1:
            first_3_chars = news_splitted[1].strip()[0:3]
            if first_3_chars == '25%':
                play_probabilities.append(0.25)
            elif first_3_chars == '50%':
                play_probabilities.append(0.5)
            elif first_3_chars == '75%':
                play_probabilities.append(0.75)
            else:
                play_probabilities.append(1.0)
        else:
            play_probabilities.append(1.0)
    fpl_online_df['play_probability'] = play_probabilities

    # adjust player points based on play probabilities
    # only include players with minutes played (which have existing projections)
    prob25_ids = fpl_online_df.loc[(fpl_online_df.play_probability==0.25) & (fpl_online_df.minutes>0), 'id'].values
    prob50_ids = fpl_online_df.loc[(fpl_online_df.play_probability==0.5) & (fpl_online_df.minutes>0), 'id'].values
    prob75_ids = fpl_online_df.loc[(fpl_online_df.play_probability==0.75) & (fpl_online_df.minutes>0), 'id'].values

    for player_id in prob25_ids:
        if latest_gameweek<=35:
            adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1, latest_gameweek+2, latest_gameweek+3], weights=[0.25,0.5,0.75])
        elif latest_gameweek==36:
            adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1, latest_gameweek+2], weights=[0.25,0.5])
        elif latest_gameweek==37:
            adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1], weights=[0.25])
        else:
            raise Exception(f'Check latest_gameweek!')

    for player_id in prob50_ids:
        if latest_gameweek<=36:
            adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1, latest_gameweek+2], weights=[0.5,0.75])
        elif latest_gameweek==37:
            adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1], weights=[0.5])
        else:
            raise Exception(f'Check latest_gameweek!')

    for player_id in prob75_ids:
        adjust_points_via_weights(projections, projections_pivot, player_id=player_id, gameweeks=[latest_gameweek+1], weights=[0.75])

    logging.info("Adding low baseline predictions for players who don't have any projections yet.")
    fpl_ids = set(fpl_online_df.id.unique())
    projection_ids = set(projections_pivot.ID.unique())
    missing_ids = fpl_ids - projection_ids

    cols = ['ID']
    cols += [f'{x}_Pts' for x in range(latest_gameweek+1, latest_gameweek+11)]
    cols += [f'{x}_xMins' for x in range(latest_gameweek+1, latest_gameweek+11)]
    cols += ['Pos']

    new_rows = []
    for id in missing_ids:
        new_row = pd.Series(index=cols)
        new_row['ID'] = id
        new_row['Pos'] = fpl_online_df.loc[fpl_online_df.id==id, 'element_type'].values[0]
        for col in cols:
            if '_Pts' in col:
                new_row[col] = 0.5
            elif '_xMins' in col:
                new_row[col] = 90.0
        new_rows.append(new_row)

    position_dict = {1:'G', 2:'D', 3:'M', 4:'F'}
    new_rows = pd.DataFrame(new_rows)
    new_rows['Pos'] = new_rows['Pos'].map(position_dict)
    projections_pivot = pd.concat([projections_pivot, new_rows], ignore_index=True)

    filepath = Path('../FPL-Optimization-Tools/data/projections.csv')
    projections_pivot.to_csv(filepath)
    logging.info(f'Projection data saved to {filepath}.')


def main(latest_gameweek: int, current_season: str, season_folder):
    
    projections, projections_pivot = initial_projection_data(latest_gameweek, current_season, season_folder)
    adjust_projections(projections, projections_pivot, latest_gameweek)


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create projection csv for team optimization model.")
    parser.add_argument(
        "latest_gameweek",
        type=int,
        help="Ongoing gameweek, i.e., not the gameweek for the next deadline."
    )
    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # get config variables
    config_json = pd.read_json('config.json', typ='series')
    SEASON_FOLDER = config_json['SEASON_FOLDER']
    current_season = SEASON_FOLDER[-5::].replace('_','-')

    main(args.latest_gameweek, current_season, SEASON_FOLDER)
