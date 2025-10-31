import argparse
import logging
import json

import pandas as pd
import numpy as np
from pathlib import Path

import gradio as gr
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go
import seaborn as sns
sns.set_style("darkgrid")

class PlotPointsAndValue:
    
    def __init__(self, df):
        self.df = df

    def process(self, positions, teams, show_names, max_price, x_axis_feature, y_axis_feature):   

        fig = go.Figure()

        # aux df for manipulation
        df_out = self.df[self.df.position.isin(positions)].copy()
        # choose only given teams
        if "Select All" not in teams:
            df_out = df_out[df_out.team_name.isin(teams)]

        # drop players above max_price
        df_out = df_out[df_out.price<=max_price]

        # CREATE FIGURE
        fig.add_trace(
                go.Scatter(
                    x=df_out[x_axis_feature],
                    y=df_out[y_axis_feature],
                    mode="markers+text",
                    hovertext=df_out['name'].values,
                    showlegend=False,
                    ),
            )

        # add player names as visible
        if show_names:
            fig.update_traces(
                text = df_out['name'].values,
                textposition='top center',
                )

        if (x_axis_feature=='season_mean_xpoints') & (y_axis_feature=='season_mean_points'):
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(0,9),
                    y=np.linspace(0,9),
                    mode='lines',
                ),
            ) 

        # styling
        fig.update_layout(
            #title="",
            template='plotly_dark',
            xaxis_title=x_axis_feature,
            yaxis_title=y_axis_feature,
            #showlegend=True
        )

        return fig

class TopWeeklyProjections:

    def __init__(self, projections):
        self.projections = projections

    def process(self, positions, teams, gameweek):

        fig = go.Figure()

        # aux df for manipulation
        projections_out = self.projections[self.projections.position.isin(positions)].copy()
        # choose only given teams
        if "Select All" not in teams:
            projections_out = projections_out[projections_out.team_name.isin(teams)]
        # choose gameweek
        projections_out = projections_out[projections_out.gameweek==int(gameweek)]

        top_20 = projections_out.groupby('name').sum().sort_values(by='expected_points').reset_index().tail(15)

        # CREATE FIGURE
        fig.add_trace(
            go.Bar(
                x=top_20.expected_points,
                #y=np.arange(top_20.shape[0],0,-1), 
                y=top_20.name, 
                text=np.round(top_20.expected_points,2),
                textposition='outside',
                width=0.75,
                orientation='h'
            ),
        ) 

        # styling
        fig.update_layout(
            #title="",
            template='plotly_dark',
            xaxis_title='gameweek expected points',
            #yaxis_title='value',
            #showlegend=True
        )

        return fig

class PlayerPoints:

    def __init__(self, df, fpl_df, projections, args):
        self.df = df
        self.fpl_df = fpl_df
        self.projections = projections
        self.args = args

    def process(self, players):

        marker_colors = ['red', 'blue']
        colors = ['rgba(255, 0, 0, 0.5)','rgba(0, 0, 255, 0.5)']
        my_df = pd.DataFrame()

        fig = go.Figure()
        for count, player in enumerate(players):

            aux = self.df.loc[self.df['name']==player, ['name', 'team_name', 'position', 'price', 
                                'season_mean_xpoints', 'points_per_game', 
                                'season_total_xpoints', 'total_points', 
                                'gameweek_xPoints_ewm_5', 'gameweek_xPoints_ewm_40', ]].copy()

            my_past_data = self.fpl_df[self.fpl_df['name']==player].sort_values(by='gameweek')
            my_projections = self.projections[self.projections['name']==player].sort_values(by='gameweek')
            
            x_past = list(np.unique(my_past_data['gameweek']))
            x_future = list(np.unique(my_projections['gameweek']))
            my_x = x_past + x_future

            y_past = list(my_past_data.groupby('gameweek').sum()['gameweek_xPoints'])
            y_future = list(my_projections.groupby('gameweek').sum()['expected_points'])
            my_y = y_past + y_future
            my_y_filtered = gaussian_filter(y_past + y_future, sigma=2, mode='nearest')

            new_cols = [f'xPoints_gameweek_{i}' for i in x_future]
            aux[new_cols] = y_future
            my_df = pd.concat([my_df, aux])

            fig.add_trace(
                go.Scatter(
                    x=my_x,
                    y=my_y_filtered,
                    mode="markers+lines",
                    marker=dict(color=marker_colors[count]),  
                    line=dict(color=marker_colors[count], width=3),  
                    fill='tozeroy',    
                    fillcolor=colors[count],   
                    name=player,            
                    showlegend=True,
                    ),
            )

            if len(players)==1:
                fig.add_trace(
                    go.Scatter(
                        x=my_x,
                        y=my_y,
                        mode="markers",
                        marker=dict(color='white'),
                        name=player,            
                        showlegend=False,
                        ),
                )

        fig.add_vline(x=self.args.latest_gameweek+0.5,)

        fig.update_layout(
            #title="",
            template='plotly_dark',
            xaxis_title="gameweek",
            yaxis_title='expected points',
            #showlegend=True
        )

        return fig, my_df

def data_processing(args, season_folder, season):
    """
    Load and process data for the Gradio dashboard.
    """
    # Load data
    logging.info("Loading and processing data.")

    # Load projections
    filepath = Path(f'{season_folder}/data/predictions/gameweek{args.latest_gameweek}.csv')
    projections = pd.read_csv(filepath, index_col=0)
    # position mapping for projections
    position_dict = {1:'GK', 2:'DEF', 3:'MID', 4:'FWD'}
    projections['position'] = projections['element_type'].map(position_dict)

    if args.latest_gameweek>0:
        filepath = Path(f'{season_folder}/data/fpl_df.csv')
    elif args.latest_gameweek==0:
        filepath = Path(f'{season_folder}/data/fpl_df_preseason.csv')
    fpl_df = pd.read_csv(filepath, index_col=0, low_memory=False)
    fpl_df = fpl_df[fpl_df.season==season]

    df = fpl_df.groupby('name').last().reset_index()[['name', 'team_name', 'element_type', 'now_cost', 
                            'gameweek_minutes_ewm_20', 'points_per_game', 'total_points', 
                            'gameweek_xPoints_ewm_5', 'gameweek_xPoints_ewm_10', 'gameweek_xPoints_ewm_20', 'gameweek_xPoints_ewm_40']]
    df['games_played'] = np.round(np.where(df['points_per_game']!=0, df['total_points'] / df['points_per_game'], 0),0)
    df['price'] = df['now_cost'] / 10.0
    df['value'] = df['gameweek_xPoints_ewm_20'] / df['price']
    df['value_points'] = np.sqrt( df['gameweek_xPoints_ewm_20'] *  df['value'])

    # EXPECTED POINTS
    expected_points_next_10gw = (projections[projections.gameweek.isin( np.arange(args.latest_gameweek+1, args.latest_gameweek+11, 1) )]
    .groupby('name')
    .sum()
    )[['expected_points']].reset_index().rename(columns={'expected_points':'expected_points_next_10_GW'})

    expected_points_next_5gw = (projections[projections.gameweek.isin( np.arange(args.latest_gameweek+1, args.latest_gameweek+6, 1) )]
    .groupby('name')
    .sum()
    )[['expected_points']].reset_index().rename(columns={'expected_points':'expected_points_next_5_GW'})

    df = df.merge(expected_points_next_10gw, on='name', how='left')
    df = df.merge(expected_points_next_5gw, on='name', how='left')

    # POSITION MAPPING
    df['position'] = df['element_type'].map(position_dict)

    # SEASON TOTALS AND MEANS (XPOINTS AND POINTS)

    season_total_xPoints = \
        (fpl_df
        .groupby('name')[['gameweek_xPoints']]
        .sum()
        .rename(columns={'gameweek_xPoints':'season_total_xpoints'})
        .reset_index())

    season_total_points = \
        (fpl_df
        .groupby('name')[['event_points']]
        .sum()
        .rename(columns={'event_points':'season_total_points'})
        .reset_index())

    season_mean_xPoints = \
        (fpl_df
        .groupby('name')[['gameweek_xPoints']]
        .mean()
        .rename(columns={'gameweek_xPoints':'season_mean_xpoints'})
        .reset_index())

    season_mean_points = \
        (fpl_df
        .groupby('name')[['event_points']]
        .mean()
        .rename(columns={'event_points':'season_mean_points'})
        .reset_index())

    df = df.merge(season_total_xPoints, on='name', how='left')
    df = df.merge(season_total_points, on='name', how='left')
    df = df.merge(season_mean_xPoints, on='name', how='left')
    df = df.merge(season_mean_points, on='name', how='left')

    return df, fpl_df, projections

def create_dashboard(args, df, fpl_df, projections):

    logging.info("Creating dashboard.")

    position_list = ['GK', 'DEF', 'MID', 'FWD']
    team_name_list = ["Select All"]
    team_name_list += list(np.sort(df.team_name.unique()))
    gameweek_list = [str(x) for x in np.arange(args.latest_gameweek+1, args.latest_gameweek+11,)]
    features = ["gameweek_xPoints_ewm_5", "gameweek_xPoints_ewm_20", 'expected_points_next_10_GW', 'value', 
                'season_total_xpoints', 'season_total_points', 'season_mean_xpoints', 'season_mean_points']
    x_axis_feature = features
    y_axis_feature = features
    minimum_price = df['price'].min()
    maximum_price = df['price'].max()

    # scatter plot
    if args.latest_gameweek>0:
        scatter_demo = gr.Interface(
            PlotPointsAndValue(df).process,
            [
                gr.CheckboxGroup(position_list, label="POSITION", value=position_list),
                gr.Dropdown(team_name_list, label="TEAM", multiselect=True, value="Select All"),
                gr.Checkbox('Show player names', value=False),
                gr.Slider(minimum_price, maximum_price, value=maximum_price, info='Choose maximum allowed player value.'),
                gr.Dropdown(x_axis_feature, label="x-axis", value='season_mean_xpoints'),
                gr.Dropdown(y_axis_feature, label="y-axis", value = 'season_mean_points'),
            ],
            gr.Plot(),
        )

    # weekly top players
    weekly_top_players_demo = gr.Interface(
        TopWeeklyProjections(projections).process,
        [
            gr.CheckboxGroup(position_list, label="POSITION", value=position_list),
            gr.Dropdown(team_name_list, label="TEAM", multiselect=True, value="Select All"),
            gr.Dropdown(gameweek_list, label="GAMEWEEK", value=str(args.latest_gameweek+1)),
        ],
        gr.Plot(),
    )

    # player points over the season
    player_name_list = list(projections.name.unique())
    with gr.Blocks() as player_points_demo:
        with gr.Column():
            choose_players = gr.Dropdown(
                player_name_list, 
                value='Erling Haaland', 
                multiselect=True, 
                max_choices=2, 
                label='Choose 1 or 2 players'
                )
            submit_button = gr.Button("Submit")
            player_plot = gr.Plot()
            player_data = gr.DataFrame()
        
        submit_button.click(
            fn=PlayerPoints(df, fpl_df, projections, args).process, 
            inputs=choose_players, 
            outputs=[player_plot, player_data]
        )

    if args.latest_gameweek>0:
        full_demo = gr.TabbedInterface(
            [scatter_demo, weekly_top_players_demo, player_points_demo],
            ['Scatter plots', 'Gameweek projections top 20', 'Player xPoints and projections']
        ).launch()
    elif args.latest_gameweek==0:
        full_demo = gr.TabbedInterface(
            [weekly_top_players_demo, player_points_demo],
            ['Gameweek projections top 20', 'Player xPoints and projections']
        ).launch()
    else:
        print("Choose valid 'latest_gameweek'!")

def main(args, season_folder, season):
    """
    Create a Gradio dashboard for FPL.
    """
    logging.info(f"Starting.")
    
    df, fpl_df, projections = data_processing(args, season_folder, season)
    create_dashboard(args, df, fpl_df, projections)
    
    logging.info("Done.")


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a Gradio dashboard.")
    parser.add_argument(
        "latest_gameweek",
        type=int,
        help="Ongoing gameweek, i.e., not the gameweek for the next deadline."
    )
    args = parser.parse_args()

    # Read the config file
    config_file_path = "config.json"
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
    # Get urls from the config file
    season_folder = config["SEASON_FOLDER"]
    season = season_folder[-5:].replace('_','-')

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Call the main function 
    main(args, season_folder, season)