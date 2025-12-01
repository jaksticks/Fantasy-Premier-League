import argparse
import pandas as pd
from pathlib import Path
from projections_for_optimization import initial_projection_data

def main(player_name, latest_gameweek, current_season, season_folder):
    
    projections, _ = initial_projection_data(latest_gameweek, current_season, season_folder)
    
    # Find player ids matching given string
    result = (projections.loc
        [projections['name'].str.contains(player_name), ['name', 'team_name', 'id']]
        .drop_duplicates()
    )

    # Check if the player ID exists in the data
    if result.empty:
        print(f"No string match for {player_name}")
    else:
        print(result)

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find player id from given name string.")
    parser.add_argument(
        "latest_gameweek",
        type=int,
        help="Ongoing gameweek, i.e., not the gameweek for the next deadline."
    )
    parser.add_argument(
        "player_name",
        type=str,
        help="Player name to match."
    )
    args = parser.parse_args()

    # get config variables
    config_json = pd.read_json('config.json', typ='series')
    SEASON_FOLDER = config_json['SEASON_FOLDER']
    current_season = SEASON_FOLDER[-5::].replace('_','-')

    main(args.player_name, args.latest_gameweek, current_season, SEASON_FOLDER)