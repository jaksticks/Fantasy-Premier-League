import argparse
import pandas as pd
from pathlib import Path
from projections_for_optimization import initial_projection_data

def main(args, current_season, season_folder):
    
    projections, _ = initial_projection_data(args.latest_gameweek, current_season, season_folder)
    
    if args.player_name:
        # Find player ids matching given string
        result = (projections.loc
            [projections['name'].str.contains(args.player_name), ['name', 'team_name', 'id']]
            .drop_duplicates()
        )

        # Check if the player ID exists in the data
        if result.empty:
            print(f"No string match for {args.player_name}")
        else:
            print(result)
    elif args.player_ids:
        # Find player names matching given ids
        result = (projections.loc
            [projections['id'].isin(args.player_ids), ['name', 'team_name', 'id']]
            .drop_duplicates()
        )

        # Check if the player IDs exist in the data
        if result.empty:
            print(f"No matches for given player IDs: {args.player_ids}")
        else:
            print(result)
    else:
        print("Please provide either --player_name or --player_ids argument.")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find player id from given name string.")
    parser.add_argument(
        "latest_gameweek",
        type=int,
        help="Ongoing gameweek, i.e., not the gameweek for the next deadline."
    )
    parser.add_argument(
        "--player_name",
        type=str,
        help="Player name to match."
    )
    parser.add_argument(
        "--player_ids",
        type=lambda x: [int(i) for i in x.split(',')],
        help="Comma-separated list of player IDs."
    )
    args = parser.parse_args()

    # get config variables
    config_json = pd.read_json('config.json', typ='series')
    SEASON_FOLDER = config_json['SEASON_FOLDER']
    current_season = SEASON_FOLDER[-5::].replace('_','-')

    main(args, current_season, SEASON_FOLDER)