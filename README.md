# Fantasy-Premier-League
This repository contains Jupyter notebooks for evaluating player performance in Fantasy Premier League (www.fantasy.premierleague.com) using expected goals data. The data is used to estimate for each player how many goals, assists and clean sheets they should have received and how many FPL points they would have scored, on average. This allows for a better evaluation of player performance than relying solely on the points they have accumulated so far, since this is greatly affected by random events.
The expected goals data is collected from fbref.com, which receives its data from statsbomb.com.
## Getting started
The Jupyter notebooks can be found in the notebooks folder. They should be run in the following order: 
1. stats_processing
2. player_form
3. playerEvaluation
The first two notebooks are used to make all the necessary calculations and in the last one several player lists are displayed showing best performing players under a few different metrics. These metrics include expected FPL-points per game, weighted average of expected points per game from recent games (form) and a simple metric (valuePoints) to find players that provide both value (points per price) and points.
