# Fantasy-Premier-League

This repository contains Jupyter notebooks for evaluating player performance in Fantasy Premier League (www.fantasy.premierleague.com) using expected goals data. The data is used to estimate for each player how many goals, assists and clean sheets they should have received and how many FPL points they would have scored, on average. This allows for a better evaluation of player performance than relying solely on the points they have accumulated so far, since this is greatly affected by random events.

The expected goals data is provided by Statsbomb, and it is freely available at fbref.com.

## Getting started

The Jupyter notebooks can be found in the notebooks folder. They should be run in the following order: 

1. stats_processing
2. player_form
3. expected_future_points
4. playerEvaluation

The first three notebooks are used to make all the necessary calculations and in the last one several player lists are displayed showing best performing players under a few different metrics. These metrics include expected FPL-points per game, average of expected points per game from recent games (form) and a simple metric (valuePoints) to find players that provide both value (points per price) and points.  

The notebook team_selection contains a stochastic search algorithm for finding good team compositions.  

The predictive model folder contains calculations that are the basis for the model used to predict future player performance based on upcoming match fixtures.

The preseason folder contains notebooks used to evaluate players based on last season's data (useful before the start of a season and also in the early reounds of a new season, when sample size is still small for evaluating player performance).

**If you want to just see the results**, then go to the notebooks folder, open playerEvaluation and scroll down: you will find lists of top 40 players by position in terms of the valuePoints metric.

## Tableau visualization

For some simple, interactive visualizations for player evaluation, see 
https://public.tableau.com/profile/jaakko4317#!/vizhome/FantasyPremierLeague_16075410150400/PointsvsValue
https://public.tableau.com/profile/jaakko4317#!/vizhome/FPL-PlayerPerformanceoverTime/Sheet1?publish=yes

## License

The work in this repository is licensed under GNU General Public License v3.0.
