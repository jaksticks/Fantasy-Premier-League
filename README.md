# Fantasy-Premier-League

This repository contains codes used for player evaluation, point projections and team optimization in Fantasy Premier League. 

Folder "season24_25" is used for analysis for the current 2024-2025 season.

Use notebook "gradio" to spin up an interactive dashboard for player analysis.

Note! Previous season data is not stored in this repo. Some of the following may not work correctly without access to all data.

To update latest data and run projections, run
```
source venv23-24/bin/activate  
python src/fpl_analysis.py <latest_gameweek> 
```
at the repository root. In the above <latest_gameweek> refers to the most recent FPL gameweek (not the upcoming one for which the deadline has not yet passed). 

The notebook "fplreview_style_projection_data" is used to create a csv file in the same format that is used by fplreview.com. This can then be used as input for a team optimization tool such as Sertalp B. Cay's repo https://github.com/sertalpbilal/FPL-Optimization-Tools to produce week-by-week transfer plans.

## License

The work in this repository is licensed under GNU General Public License v3.0.
