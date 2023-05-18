# GNNs-for-NBA
Applying Graph Neural Networks to predicting NBA regular season game prediction

This repository contains the notebooks used to generate the heterogenous graphs that represenent individual NBA regular season games from 2013-2021. In the project, I created this dataset where teams and players are represented as nodes, and relationships between players and players and players and teams are represented as edges. This therefore requires a special type of neural network called a Graph Neural Network which allows for the analysis of not just the features of the players and teams, but also the relationships between them. 

After creating the dataset and models and running the analysis, it was found that the GNN structure slightly improved performance over non-graph structured data/models, warranting further research into the graph structure for this prediction task. 
