# Fantasy-Premier-League

This repository contains codes used for player evaluation, point projections and team optimization in Fantasy Premier League. 

## Installation

- Create a virtual environment e.g. with conda
- Install required libraries: ```conda install --file requirements.txt```
- Install gradio with pip: ```pip install gradio```
- run ```pip install -e .```

## Analysis and projections

Folder "season24_25" is used for analysis for the current 2024-2025 season.

Use notebook "gradio" to spin up an interactive dashboard for player analysis.

To update latest data and run projections, activate your virtual environment and run
```  
python src/fpl_analysis.py <latest_gameweek> 
```
at the repository root. In the above <latest_gameweek> refers to the most recent FPL gameweek (not the upcoming one for which the deadline has not yet passed). 

The notebook "fplreview_style_projection_data" is used to create a csv file in the same format that is used by fplreview.com. This can then be used as input for a team optimization tool such as Sertalp B. Cay's repo https://github.com/sertalpbilal/FPL-Optimization-Tools to produce week-by-week transfer plans.

Note! Here we only produce point projections for players, not for the new (24-25 season) assistant manager position (which might be needed to run optimization with the latest version of the above repo).

## License

The work in this repository is licensed under GNU General Public License v3.0.
