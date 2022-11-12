# Fantasy-Premier-League

This repository contains Jupyter notebooks for evaluating player performance in Fantasy Premier League (www.fantasy.premierleague.com) using expected goals data. The data is used to estimate for each player how many goals, assists and clean sheets they should have received and how many FPL points they would have scored, on average. This allows for a better evaluation of player performance than relying solely on the points they have accumulated so far, since this is greatly affected by random events.

The expected goals data is provided by Opta, and it is freely available at fbref.com.

## Getting started

season22_23 contains the latest files. notebooks folder includes the essential Jupyter notebooks for running the analysis. They should be run in the indicated order.

The first three notebooks are used to make all the necessary calculations and in the fourth one several player lists are displayed showing best performing players under a few 
different metrics. These metrics include expected FPL-points per game, average of expected points per game from recent games (form) and a simple metric (valuePoints) to find players that provide both value (points per price) and points. The fifth notebook contains a linear optimization algorithm for finding an optimal team.  

## License

The work in this repository is licensed under GNU General Public License v3.0.
