# Fantasy-Premier-League

This repository contains codes used for player evaluation, point projections and team optimization in Fantasy Premier League. 

Use notebook "7_gradio" to spin up an interactive dashboard for player analysis.

Note! Previous season data is not stored in this repo. Some of the following may not work correctly without access to all data.

To update latest data and run projections, run
```
source venv23-24/bin/activate  
python src/fpl_analysis.py <latest_gameweek> 
```
at the repository root. In the above <latest_gameweek> refers to the most recent FPL gameweek (not the upcoming one for which the deadline has not yet passed). 

Use notebook "3_evaluate_and_train_model" to monitor and re-train models. Use notebook "5_team_optimization" to run a linear optimization algorithm to find optimal team(s). Use notebook "6_player_analysis" for additional analysis, visualizations and gameweek point projections for your team. 

## License

The work in this repository is licensed under GNU General Public License v3.0.
