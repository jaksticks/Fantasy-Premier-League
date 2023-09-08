import pandas as pd
import numpy as np

from pathlib import Path
import datetime as dt
import json
import requests
from src.utils import fetch_latest_fpl_data
import pickle
import argparse

#from sklearn.linear_model import LogisticRegression
from scipy.stats import poisson
import catboost

#import matplotlib.pyplot as plt
#import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument('latest_gameweek')
args = parser.parse_args()
latest_gameweek = int(args.latest_gameweek)

# determine in which season folder data is stored
SEASON_FOLDER = 'season23_24'
# give the file name for the model you are using (located in season_folder/models/)
MODEL_FILE_NAME = 'catboost_20230809-201635.cbm'

def data_retrieval(latest_gameweek: int, season_folder: str):
    '''Fetch all new data'''
 
    # teams for season 23-24
    teams = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
            'Liverpool', 'Luton', 'Manchester City', 'Manchester Utd',
            'Newcastle Utd', 'Nottingham Forest', 'Sheffield Utd', 'Tottenham',
            'West Ham', 'Wolves']
    
    # FPL PLAYER DATA
    
    # fetch FPL data online
    fpl_online_data = json.loads(requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').text)
    fpl_online_df = pd.DataFrame(fpl_online_data['elements'])
    fpl_online_df['team_name'] = [teams[i] for i in fpl_online_df['team']-1]
    fpl_online_df['name'] = fpl_online_df.apply(lambda x: x['first_name'] + ' ' + x['second_name'], axis=1)
    
    # CREATE NEW DATA SET IF THERE IS NEW DATA AVAILABLE AND SAVE TO FILE

    old_data = fetch_latest_fpl_data(folder_path_str=f'{season_folder}/data/fpl/')

    # only take players who have played, i.e., minutes>0
    new_data = fpl_online_df[fpl_online_df.minutes>0].copy()
    # players who have now played but had not previously played at all
    new_data_1 = new_data[~new_data.name.isin(old_data.name.unique())].copy()
    # players whose minutes are higher now than previously
    aux = new_data[new_data.name.isin(old_data.name.unique())].copy()
    new_rows = []
    for ix, row in aux.iterrows():
        player_name = row['name']
        change_in_minutes = row['minutes'] - old_data.loc[old_data.name==player_name, 'minutes'].iloc[-1]
        if change_in_minutes > 0:
            new_rows.append(row)
    if len(new_rows) > 0:
        new_data_2 = pd.DataFrame(new_rows)
    else:
        new_data_2 = pd.DataFrame() # empty df

    # overwrites old new_data variable
    new_data = pd.concat([new_data_1, new_data_2], ignore_index=True)
    print(f'New fpl data: {new_data.shape[0]} rows.')
    # create new data set combining old and new data and save to file
    if new_data.shape[0] > 0:

        # add info
        new_data['gameweek'] = latest_gameweek
        season_str = season_folder[-5:]
        season_str = season_str.replace('_','-')
        new_data['season'] = season_str
        time_now = dt.datetime.now()
        new_data['data_retrieved_datetime'] = time_now
        
        full_data = pd.concat([old_data, new_data], ignore_index=True)
        print(f'Full data shape: {full_data.shape}')
        
        # save new full data
        path = Path(f'{season_folder}/data/fpl/data_' + str(time_now.strftime("%Y%m%d-%H%M%S")) + '.csv')
        full_data.to_csv(path)
    
    ## FPL fixtures data
    
    # get FPL fixtures data
    fpl_fixtures_data = json.loads(requests.get('https://fantasy.premierleague.com/api/fixtures/').text)
    fpl_fixtures = pd.DataFrame(fpl_fixtures_data)
    fpl_fixtures['home_team'] = [teams[i] for i in fpl_fixtures['team_h']-1]
    fpl_fixtures['away_team'] = [teams[i] for i in fpl_fixtures['team_a']-1]
    
    # save data
    filepath = Path(f'{season_folder}/data/fixtures/fpl_fixtures.csv')
    fpl_fixtures.to_csv(filepath)

    ## FBRef fixtures
    
    # fetch data
    data = pd.read_html('https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures')
    fbref_fixtures = data[0]
    fbref_fixtures = fbref_fixtures[fbref_fixtures['xG'].notnull()]
    fbref_fixtures = fbref_fixtures.rename(columns={'xG':'xG_home', 'xG.1':'xG_away'})
    
    # save data
    filepath = Path(f'{season_folder}/data/fixtures/fbref_fixtures.csv')
    fbref_fixtures.to_csv(filepath)

def fpl_data_processing(df, columns):
        '''Auxiliary function used in data procesing.'''

        xg_data = []
        xa_data = []
        xga_data = []
        for ix, row in df.iterrows():
            my_gameweek = row['gameweek']
            xg_data.append( row[f'xG_week{my_gameweek}'] )
            xa_data.append( row[f'xA_week{my_gameweek}'] )
            xga_data.append( row[f'xGA_week{my_gameweek}'] )

        df['gameweek_xG'] = xg_data
        df['gameweek_xA'] = xa_data
        df['gameweek_xGA'] = xga_data

        df_new = df[columns].copy()

        return df_new

def my_fill_na(x, gameweek_col, diff_col):
        '''Fill nan values for first items for grouped variables where diff is calculated. But also don't fill for season 22-23,
        where data is missing for a number of weeks at the beginning of the season.'''
        my_value = x[diff_col] if (np.isnan(x[gameweek_col])) & (x['minutes']<=90) else x[gameweek_col]
        return my_value

def calculate_xPoints(x,clf):
        """Expected points for a given gameweek given underlying stats for that gameweek."""

        clean_sheet_points = np.array([4,4,1,0])
        goal_points = np.array([6,6,5,4])

        # calculate expexted points
        points_played = np.array([1 if x['gameweek_minutes']>0 else 0])
        points_played_over_60 = np.array([1 if x['gameweek_minutes']>=60 else 0])
        points_xG = goal_points[x['element_type']-1] * x['gameweek_xG']
        points_xA = x['gameweek_xA'] * 3
        clean_sheet_probability = np.array(poisson.pmf(0,x['team_xGA']))
        points_clean_sheet = [clean_sheet_points[x['element_type']-1] * clean_sheet_probability if x['gameweek_minutes']>=60 else 0]
        points_saves = x['gameweek_saves'] // 3
        points_penalty_saves = x['gameweek_penalties_saved'] * 5 * 0.21 #points for save times approx. probability of penalty save
        #penalty_for_penalty_miss = x['Performance_PKatt'] * (-2*0.21) # this data only on fbref
        # estimate bonus points
        if not np.isnan(x['gameweek_bps']):
            y_pred_prob = clf.predict_proba(np.array(x['gameweek_bps']).reshape(-1, 1))
        else:
            # return nan if bonus points can't be estimated 
            return np.nan
        points_bonus = np.matmul(y_pred_prob, np.array([0,1,2,3]).reshape((4,1)))
        
        # penalty for possible points deductions based on goals conceded
        xGA = x['team_xGA']
        # calculate penalty
        xGA_conceded_penalty = -(poisson.pmf(2,xGA)+poisson.pmf(3,xGA))-(poisson.pmf(4,xGA)+poisson.pmf(5,xGA))-(poisson.pmf(6,xGA)+poisson.pmf(7,xGA))-(poisson.pmf(8,xGA)+poisson.pmf(9,xGA)-(poisson.pmf(10,xGA)+poisson.pmf(11,xGA)))
        # apply penalty only to GK and DEF
        if (x['element_type']==1) | (x['element_type']==2):
            xGA_conceded_penalty = xGA_conceded_penalty
        else:
            xGA_conceded_penalty = 0
        # scale penalty with playing time
        xGA_conceded_penalty = (x['gameweek_minutes'] / 90) * xGA_conceded_penalty

        penalty_for_cards = [-3 if x['gameweek_red_cards']==1 else -1 if x['gameweek_yellow_cards']==1 else 0]
        penalty_for_own_goal = -2 * x['gameweek_own_goals']

        # add up all point components
        total_points = float(points_played + points_played_over_60 + points_xG + points_xA + points_clean_sheet + points_saves +\
                        points_penalty_saves + points_bonus + xGA_conceded_penalty +\
                        penalty_for_cards + penalty_for_own_goal)
        
        return total_points

def data_processing(season_folder: str, shift_param: int = 1):
    '''Process FPL data.
    Input shift_param used for moving e.g. team rolling averages down so that only knowledge from previoius games
    is available when making predictions.
    '''

    ## Need to add data processing for gameweeks where one team has multiple games. 
    # This needs to be done for the "Add team data to FPL data" section. 
    # Should be doable utilizing fpl_df.data_retrieved to pick right games (the previous game before the given date).

    # load model for estimating bonus points based on gameweek bps
    model_path = Path(f"{season_folder}/models/logistic_regression_for_bonus_points.pkl")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    
    # Fetch data
    
    # fpl data from previous seasons
    filepath = Path(f'{season_folder}/data/modeling/fpl_df.csv')
    fpl_df = pd.read_csv(filepath, index_col=0)
    
    # fpl data from this season
    fpl_df_new = fetch_latest_fpl_data(f'{season_folder}/data/fpl/')
    
    # get gameweek data
    fpl_df_new['gameweek_xG'] = fpl_df_new.groupby('name')['expected_goals'].diff().fillna(fpl_df_new['expected_goals'])
    fpl_df_new['gameweek_xA'] = fpl_df_new.groupby('name')['expected_assists'].diff().fillna(fpl_df_new['expected_assists'])
    fpl_df_new['gameweek_xGA'] = fpl_df_new.groupby('name')['expected_goals_conceded'].diff().fillna(fpl_df_new['expected_goals_conceded'])

    # concatenate new fpl data with old
    fpl_df = pd.concat([fpl_df, fpl_df_new], join='outer').reset_index(drop=True)
    
    # rolling team data from past seasons
    filepath = Path(f'{season_folder}/data/modeling/team_data.csv')
    team_data = pd.read_csv(filepath, index_col=0)
    
    # fpl fixtures data from this season
    filepath = Path(f'{season_folder}/data/fixtures/fpl_fixtures.csv')
    fixtures_fpl = pd.read_csv(filepath, index_col=0)
    fixtures_fpl = fixtures_fpl[fixtures_fpl.finished]
    
    # fbref fixtures data from this season
    filepath = Path(f'{season_folder}/data/fixtures/fbref_fixtures.csv')
    fixtures_fbref = pd.read_csv(filepath, index_col=0)
    
    # Data processing
    
    ### Fix certain player identification issues
    # for Heung-Min Son, the first_name and second_name change along the data
    # here we fix the first_name and second_name so that player can be correctly identified in the data
    fpl_df.loc[fpl_df.web_name=='Son', 'first_name'] = 'Heung-Min'
    fpl_df.loc[fpl_df.web_name=='Son', 'second_name'] = 'Son'
    fpl_df.loc[fpl_df.web_name=='Son', 'name'] = 'Heung-Min Son'
    
    ### Process FPL data
    # find how many minutes a player played on a given gameweek
    fpl_df['gameweek_minutes'] = fpl_df.groupby(['first_name', 'second_name', 'season'])['minutes'].diff()
    # fill na caused at the start of each season by taking diff (but don't fill for season 22-23 where early season data is missing)
    fpl_df['gameweek_minutes'] = fpl_df.apply(lambda x: my_fill_na(x, 'gameweek_minutes', 'minutes'), axis=1)
    
    # drop rows with 0 minutes or more than 90 minutes
    fpl_df = fpl_df[(fpl_df.gameweek_minutes>0) & (fpl_df.gameweek_minutes<=90)].reset_index(drop=True)
    
    ### Add xG data to FPL fixtures data
    
    # map fbref team names to fpl team names
    fbref_teams = np.sort(pd.concat([fixtures_fbref.Home, fixtures_fbref.Away]).unique())
    fpl_teams = np.sort(pd.concat([fixtures_fpl.home_team, fixtures_fpl.away_team]).unique())
    team_name_dict = dict(zip(fbref_teams, fpl_teams))
    #print('Check team names dictionary:')
    #print(team_name_dict)

    fixtures_fbref['Home'] = fixtures_fbref['Home'].apply(lambda x: team_name_dict[x])
    fixtures_fbref['Away'] = fixtures_fbref['Away'].apply(lambda x: team_name_dict[x])
    
    home_xg = []
    away_xg = []
    for ix, row in fixtures_fpl.iterrows():
        home_team = row.home_team
        away_team = row.away_team
        home_team_xg = fixtures_fbref.loc[(fixtures_fbref['Home']==home_team) & (fixtures_fbref['Away']==away_team), 'xG_home'].values[0]
        away_team_xg = fixtures_fbref.loc[(fixtures_fbref['Home']==home_team) & (fixtures_fbref['Away']==away_team), 'xG_away'].values[0]
        home_xg.append( home_team_xg )
        away_xg.append( away_team_xg )

    fixtures_fpl['xg_home'] = home_xg
    fixtures_fpl['xg_away'] = away_xg

    ### Calculate exponentially weighted moving averages for each teams' xG data
    
    fixtures_melt = fixtures_fpl.melt(id_vars=['xg_home', 'xg_away', 'team_h_score', 'team_a_score', 'event', 'kickoff_time', 'id'], value_vars=['home_team', 'away_team'])
    season_str = season_folder[-5:]
    season_str = season_str.replace('_','-')
    fixtures_melt['season'] = season_str
    
    # concatenate fixtures_melt with team data (previous seasons)
    fixtures_melt = pd.concat([team_data, fixtures_melt], ignore_index=True)
    
    # get team's xG (home xG if at home, away xG if at an away game)
    fixtures_melt['xG'] = fixtures_melt.apply(lambda x: x['xg_home'] if x['variable']=='home_team' else x['xg_away'], axis=1)
    fixtures_melt['xGA'] = fixtures_melt.apply(lambda x: x['xg_away'] if x['variable']=='home_team' else x['xg_home'], axis=1)

    # sort by date
    fixtures_melt = fixtures_melt.sort_values(by='kickoff_time').reset_index(drop=True)

    # calculate rolling averages
    rolling_windows = [5,10,20,40]

    for i in rolling_windows:
        fixtures_melt[f'xG_ewm_{i}'] = (fixtures_melt[['value','xG']].groupby(by='value').ewm(alpha=1/i).mean()
                                        .reset_index().sort_values(by='level_1')['xG'].values)
        fixtures_melt[f'xGA_ewm_{i}'] = (fixtures_melt[['value','xGA']].groupby(by='value').ewm(alpha=1/i).mean()
                                        .reset_index().sort_values(by='level_1')['xGA'].values)
        
    # save fixtures_melt
    filepath = Path(f'{season_folder}/data/team_data.csv')
    fixtures_melt.to_csv(filepath)
    
    # shift team xg data by one so that the target game result is not included
    cols_to_shift = [col for col in fixtures_melt if 'ewm' in col]
    fixtures_melt[cols_to_shift] = fixtures_melt.groupby('value')[cols_to_shift].shift(shift_param)

    # save fixtures_melt with shifted variables
    filepath = Path(f'{season_folder}/data/team_data_shift{shift_param}.csv')
    fixtures_melt.to_csv(filepath)

    ### Add team data to FPL data
    # columns to be fetched from team data
    col_names = ['xG', 'xGA']
    col_names += [f'xG_ewm_{i}' for i in rolling_windows]
    col_names += [f'xGA_ewm_{i}' for i in rolling_windows]
    nr_cols = len(col_names) 
    team_data = []
    opponent_data = []
    home_indicator = []
    count_non_one_games = 0
    for ix, row in fpl_df[fpl_df.season==season_str].iterrows():
        gameweek = row.gameweek
        team = row.team_name
        season = row.season
        games = fixtures_melt[(fixtures_melt.value==team) & (fixtures_melt.event==gameweek) & (fixtures_melt.season==season)]
        if games.shape[0]!=1:
            team_data.append( np.array([np.nan]*nr_cols) )
            opponent_data.append( np.array([np.nan]*nr_cols) )
            home_indicator.append( np.array([np.nan]) )
            count_non_one_games += 1
        elif games.shape[0]==1:
            # add team data
            team_data.append( games[col_names].values.flatten() )
            # find opponent data
            home_game = games.variable.values[0]=='home_team'
            game_id = games.id.values[0]
            if home_game:
                home_indicator.append( np.array([1]) )
                opponent_team = fixtures_fpl.loc[(fixtures_fpl.home_team==team) & (fixtures_fpl.event==gameweek), 'away_team'].values[0]
            else:
                home_indicator.append( np.array([0]) )
                opponent_team = fixtures_fpl.loc[(fixtures_fpl.away_team==team) & (fixtures_fpl.event==gameweek), 'home_team'].values[0]
            opponent_games = fixtures_melt[(fixtures_melt.value==opponent_team) & (fixtures_melt.event==gameweek) & (fixtures_melt.season==season) & (fixtures_melt.id==game_id)]
            # add opponent data
            opponent_data.append( opponent_games[col_names].values.flatten() )
        else:
            print(f'Data processing (adding team data): check number of games for ix {ix}!')
        
    new_col_names = ['team_'+col for col in col_names]
    team_data_df = pd.DataFrame(team_data, columns=new_col_names, index=fpl_df[fpl_df.season==season_str].index)
    new_oppo_col_names = ['opponent_'+col for col in col_names]
    opponent_data_df = pd.DataFrame(opponent_data, columns=new_oppo_col_names, index=fpl_df[fpl_df.season==season_str].index)
    home_indicator_df = pd.DataFrame(home_indicator, columns=['home'], index=fpl_df[fpl_df.season==season_str].index)

    fpl_df.loc[fpl_df.season==season_str, new_col_names] = team_data_df
    fpl_df.loc[fpl_df.season==season_str, new_oppo_col_names] = opponent_data_df
    fpl_df.loc[fpl_df.season==season_str, 'home'] = home_indicator_df

    ### FPL gameweek stats
    # calculate gameweek stats by looking at differences in cumulative stats

    diff_columns = ['assists', 'bps', 'creativity', 'goals_scored', 'goals_conceded', 'own_goals', 'penalties_saved', 
                    'red_cards', 'saves', 'threat', 'yellow_cards']

    for col in diff_columns:
        fpl_df[f'gameweek_{col}'] = fpl_df.groupby(['web_name', 'season'])[col].diff()
        fpl_df[f'gameweek_{col}'] = fpl_df.apply(lambda x: my_fill_na(x, f'gameweek_{col}', col), axis=1)
    
    ### FPL expected points
    fpl_df['gameweek_xPoints'] = fpl_df.apply(lambda x: calculate_xPoints(x,clf), axis=1)
    
    ### FPL moving averages
    # calculate moving averages based on gameweek stats

    ewm_columns = ['gameweek_assists', 'gameweek_bps', 'gameweek_creativity', 'event_points', 'gameweek_goals_scored', 'gameweek_goals_conceded', 'gameweek_saves', 
                'gameweek_threat', 'gameweek_xG', 'gameweek_xA', 'gameweek_xGA', 'gameweek_minutes', 'gameweek_xPoints']

    for i in rolling_windows:
        new_columns = [col+f'_ewm_{i}' for col in ewm_columns]
        fpl_df[new_columns] = (fpl_df
                               .groupby('web_name')[ewm_columns]
                               .ewm(alpha=1/i)
                               .mean()
                               .reset_index()
                               .sort_values(by='level_1')[ewm_columns]
                               .values)
    
    # FPL expanding stats

    expanding_columns = ['gameweek_assists', 'gameweek_bps', 'gameweek_creativity', 'event_points', 'gameweek_goals_scored', 'gameweek_goals_conceded', 'gameweek_saves', 
                'gameweek_threat', 'gameweek_xG', 'gameweek_xA', 'gameweek_xGA', 'gameweek_minutes', 'gameweek_xPoints']
    expanding_col_names = [col+'_expanding' for col in expanding_columns]

    fpl_df[expanding_col_names] = (
        fpl_df
        .groupby(['first_name', 'second_name'])[expanding_columns]
        .expanding()
        .sum()
        .reset_index()
        .sort_values('level_2')[expanding_columns]
        .values
    )

    # FPL per 90 stats

    per_90_columns = [col+'_per90' for col in expanding_col_names]

    for i in range(len(per_90_columns)):
        fpl_df[per_90_columns[i]] = fpl_df[expanding_col_names[i]] / fpl_df['gameweek_minutes_expanding'] * 90

    # Add xG overperfomance

    fpl_df['xG_overperformance'] = fpl_df['gameweek_goals_scored_expanding'] / fpl_df['gameweek_xG_expanding']
    # fix if division with zero
    fpl_df.loc[np.isinf(fpl_df['xG_overperformance']), 'xG_overperformance'] = 1
        
    # Save data
    filepath = Path(f'{season_folder}/data/fpl_df.csv')
    fpl_df.to_csv(filepath)

def make_projections(latest_gameweek: int, season_folder: str, model_file_name: str):
    '''Make FPL point projections for future gameweeks.'''

    # fetch fpl data
    filepath = Path(f'{season_folder}/data/fpl_df.csv')
    fpl_df = pd.read_csv(filepath, index_col=0, low_memory=False)
    
    # fetch fpl fixtures
    filepath = Path(f'{season_folder}/data/fixtures/fpl_fixtures.csv')
    fixtures_fpl = pd.read_csv(filepath, index_col=0)
    
    # define features for the model

    features_no_shift = ['element_type', 'home']

    features_shift = ['corners_and_indirect_freekicks_order', 'creativity_rank', 
        'direct_freekicks_order', 'ict_index_rank', 'influence_rank',
        'minutes', 'now_cost', 'penalties_order', 'points_per_game', 
        'selected_by_percent', 'threat_rank',
        'team_xG_ewm_5', 'team_xG_ewm_10', 'team_xG_ewm_20',
        'team_xG_ewm_40', 'team_xGA_ewm_5', 'team_xGA_ewm_10',
        'team_xGA_ewm_20', 'team_xGA_ewm_40', 
        'opponent_xG_ewm_5', 'opponent_xG_ewm_10',
        'opponent_xG_ewm_20', 'opponent_xG_ewm_40', 'opponent_xGA_ewm_5',
        'opponent_xGA_ewm_10', 'opponent_xGA_ewm_20',
        'opponent_xGA_ewm_40', 
        'gameweek_assists_ewm_5', 'gameweek_bps_ewm_5',
        'gameweek_creativity_ewm_5', 'event_points_ewm_5',
        'gameweek_goals_scored_ewm_5', 'gameweek_goals_conceded_ewm_5',
        'gameweek_saves_ewm_5', 'gameweek_threat_ewm_5',
        'gameweek_xG_ewm_5', 'gameweek_xA_ewm_5', 'gameweek_xGA_ewm_5',
        'gameweek_minutes_ewm_5', 'gameweek_xPoints_ewm_5',
        'gameweek_assists_ewm_10', 'gameweek_bps_ewm_10',
        'gameweek_creativity_ewm_10', 'event_points_ewm_10',
        'gameweek_goals_scored_ewm_10', 'gameweek_goals_conceded_ewm_10',
        'gameweek_saves_ewm_10', 'gameweek_threat_ewm_10',
        'gameweek_xG_ewm_10', 'gameweek_xA_ewm_10', 'gameweek_xGA_ewm_10',
        'gameweek_minutes_ewm_10', 'gameweek_xPoints_ewm_10',
        'gameweek_assists_ewm_20', 'gameweek_bps_ewm_20',
        'gameweek_creativity_ewm_20', 'event_points_ewm_20',
        'gameweek_goals_scored_ewm_20', 'gameweek_goals_conceded_ewm_20',
        'gameweek_saves_ewm_20', 'gameweek_threat_ewm_20',
        'gameweek_xG_ewm_20', 'gameweek_xA_ewm_20', 'gameweek_xGA_ewm_20',
        'gameweek_minutes_ewm_20', 'gameweek_xPoints_ewm_20',
        'gameweek_assists_ewm_40', 'gameweek_bps_ewm_40',
        'gameweek_creativity_ewm_40', 'event_points_ewm_40',
        'gameweek_goals_scored_ewm_40', 'gameweek_goals_conceded_ewm_40',
        'gameweek_saves_ewm_40', 'gameweek_threat_ewm_40',
        'gameweek_xG_ewm_40', 'gameweek_xA_ewm_40', 'gameweek_xGA_ewm_40',
        'gameweek_minutes_ewm_40', 'gameweek_xPoints_ewm_40',
        'gameweek_assists_expanding', 'gameweek_bps_expanding',
        'gameweek_creativity_expanding', 'event_points_expanding',
        'gameweek_goals_scored_expanding',
        'gameweek_goals_conceded_expanding', 'gameweek_saves_expanding',
        'gameweek_threat_expanding', 'gameweek_xG_expanding',
        'gameweek_xA_expanding', 'gameweek_xGA_expanding',
        'gameweek_minutes_expanding', 'gameweek_xPoints_expanding',
        'gameweek_assists_expanding_per90', 'gameweek_bps_expanding_per90',
        'gameweek_creativity_expanding_per90',
        'event_points_expanding_per90',
        'gameweek_goals_scored_expanding_per90',
        'gameweek_goals_conceded_expanding_per90',
        'gameweek_saves_expanding_per90',
        'gameweek_threat_expanding_per90', 'gameweek_xG_expanding_per90',
        'gameweek_xA_expanding_per90', 'gameweek_xGA_expanding_per90',
        'gameweek_xPoints_expanding_per90', 'xG_overperformance'
        ]

    features = features_no_shift + features_shift
    
    season_str = season_folder[-5:]
    season_str = season_str.replace('_','-')
    df = fpl_df.loc[fpl_df.season==season_str].groupby(['name'])[features + ['team_name']].last().reset_index()
    
    path = Path(f'{season_folder}/data/team_data.csv')
    team_data = pd.read_csv(path, index_col=0)
    
    # get latest moving average info for each team
    team_data = team_data.groupby('value').last()
    ewm_cols = [col for col in team_data.columns if 'ewm' in col]
    team_data = team_data[ewm_cols]
    
    # change col names to have 'opponent' in front
    new_cols = ['opponent_' + col for col in ewm_cols]
    team_data.columns = new_cols
    team_data = team_data.reset_index()
    
    # get prediction data by adding rows for each future game for each player and getting the right opponent data
    prediction_data = []
    first_gameweek = latest_gameweek + 1
    last_gameweek = np.min((latest_gameweek + 10, 38))
    for ix, row in df.iterrows():
        my_team = row['team_name']
        for gameweek in range(first_gameweek,last_gameweek+1):
            
            opponent_data = []
            opponent_names = []
            home_game = []
            date_data = []
            gameweek_data = []

            # home games
            home_games = fixtures_fpl[(fixtures_fpl.event==gameweek) & (fixtures_fpl.home_team==my_team)]
            for ix2, row2 in home_games.iterrows():
                opponent_name = row2['away_team']
                # get opponents xg data            
                opponent_data.append( team_data.loc[team_data.value==opponent_name, new_cols] )
                # record opponent name
                opponent_names.append( opponent_name )
                # record whether home game
                home_game.append( 1 )
                # record date of game
                date_data.append( row2['kickoff_time'] )
                # record gameweek
                gameweek_data.append( gameweek )
            
            # away games
            away_games = fixtures_fpl[(fixtures_fpl.event==gameweek) & (fixtures_fpl.away_team==my_team)]
            for ix2, row2 in away_games.iterrows():
                opponent_name = row2['home_team']
                opponent_data.append( team_data.loc[team_data.value==opponent_name, new_cols] )            
                opponent_names.append( opponent_name )
                home_game.append( 0 )
                date_data.append( row2['kickoff_time'] )
                gameweek_data.append( gameweek )

            # create duplicate rows of the target player for each game and replace opponent data with correct info
            copy_of_row = row.copy()
            for i in range(0,len(opponent_data)):
                copy_of_row[new_cols] = opponent_data[i].squeeze()
                copy_of_row['opponent_team'] = opponent_names[i]
                copy_of_row['home'] = home_game[i]
                copy_of_row['date'] = date_data[i]
                copy_of_row['gameweek'] = gameweek_data[i]
                prediction_data.append( copy_of_row )

    prediction_df = pd.DataFrame(prediction_data).reset_index(drop=True)
    
    # load prediction model
    model = catboost.CatBoostRegressor()
    path = Path(f'{season_folder}/models/{model_file_name}')
    model.load_model(path)

    # make projections and save to file
    X = prediction_df[features]
    prediction_df['expected_points'] = model.predict(X)
    prediction_df.loc[prediction_df.name.str.contains('Haaland'), ['name', 'team_name', 'opponent_team', 'home', 'date', 'expected_points']]
    path = Path(f'{season_folder}/data/predictions/gameweek{latest_gameweek}.csv')
    prediction_df.to_csv(path)

def main():
    
    print('Retriving data.')
    data_retrieval(latest_gameweek, SEASON_FOLDER)
    
    print('Processing data.')
    data_processing(SEASON_FOLDER)
    
    print('Making projections.')
    make_projections(latest_gameweek, SEASON_FOLDER, MODEL_FILE_NAME)
    
    print('Done.')

if __name__ == "__main__":
    main()