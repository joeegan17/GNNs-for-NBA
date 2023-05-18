
### Missing some inputs for these classes here since I pulled these out of notebooks. If you're looking to replicate, I would just spend a couple minutes filling in the missing pieces and you should be good to go.
### example: the graph structured data input sizes/configurations are hard coded since they came from a notebook but if you follow the other notebooks you should have the right formatting
#Generic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from tqdm import tqdm
import os
import time
import datetime
import pickle
## pytorch packages
from tqdm import auto
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, TopKPooling, SAGPooling, GATConv
from torch_geometric.data import Batch
from torch.nn import BatchNorm1d
from torch_geometric.nn import HANConv, HeteroConv
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_auc_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#### MODELS ####

class TClassifier(torch.nn.Module):
    def __init__(self, team_in_channels, hidden_channels, edge_in_channels,num_classes,  dropout = 0.5):
        super(TClassifier, self).__init__()
        
       
   
        self.leaky_relu = torch.nn.LeakyReLU(0.02)
        self.post_cat = Linear((team_in_channels+team_in_channels+edge_in_channels), hidden_channels)
        self.post_cat_bn = BatchNorm1d(hidden_channels)
        self.final_linear = Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.light_dropout = torch.nn.Dropout(dropout/2)
    
        

        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x_dict, data.edge_index_dict, data.edge_attr_dict[('team','to','team')]
        team_batch = data['team'].batch

        #x_player = global_mean_pool(x['player'], data['player'].batch)
        #print(x_player.shape)

        output_list = []
        for i in range(data.num_graphs):
            x_i = torch.cat((x['team'][2*i], x['team'][2*i+1]), dim=-1)
            edge_attr_i = edge_attr[i]

            x_i = torch.cat((x_i, edge_attr_i), dim=-1)
            #x_i = torch.cat((x_i, x_player[i]), dim=-1)
            output_list.append(x_i)

        x = torch.stack(output_list)
        x = self.post_cat(x)
        x = self.post_cat_bn(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.final_linear(x)
        return x




class TPClassifier(torch.nn.Module):
    def __init__(self, team_in_channels, hidden_channels, edge_in_channels,num_classes,  dropout = 0.5):
        super(TPClassifier, self).__init__()
        
       

        
        self.edge_transform = Linear(edge_in_channels, hidden_channels)
        self.team_linear = Linear(hidden_channels, hidden_channels)
        self.team_bn = BatchNorm1d(hidden_channels)
        self.player_linear = Linear(hidden_channels, hidden_channels)
        self.player_bn = BatchNorm1d(hidden_channels)
        self.teampool = SAGPooling(hidden_channels, ratio=0.5)
        self.playerpool = SAGPooling(hidden_channels, ratio=0.5)
        self.leaky_relu = torch.nn.LeakyReLU(0.02)
        self.post_cat = Linear((team_in_channels+team_in_channels+edge_in_channels+46+46), hidden_channels)
        self.post_cat_bn = BatchNorm1d(hidden_channels)
        self.final_linear = Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.light_dropout = torch.nn.Dropout(dropout/2)

        

        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x_dict, data.edge_index_dict, data.edge_attr_dict[('team','to','team')]
        team_batch = data['team'].batch
        player_batch = data['player'].batch
        player_to_team_edge_index = data.edge_index_dict[('player', 'to', 'team')]

        home_player_indices = (player_to_team_edge_index[1] % 2 == 0).nonzero(as_tuple=True)[0]
        away_player_indices = (player_to_team_edge_index[1] % 2 == 1).nonzero(as_tuple=True)[0]

        # Extract home and away player features
        x_player_home = x['player'][home_player_indices]
        x_player_away = x['player'][away_player_indices]

        # Perform pooling separately for home and away players
        x_player_home_pooled = global_mean_pool(x_player_home, player_batch[home_player_indices])
        x_player_away_pooled = global_mean_pool(x_player_away, player_batch[away_player_indices])

        #print(x_player_home_pooled.shape)
        #print(x_player_away_pooled.shape)


        #x_player = global_mean_pool(x['player'], data['player'].batch)
        #print(x_player.shape)

        output_list = []
        for i in range(data.num_graphs):
            x_i = torch.cat((x['team'][2*i], x['team'][2*i+1]), dim=-1)
            edge_attr_i = edge_attr[i]

            x_i = torch.cat((x_i, edge_attr_i), dim=-1)
            #print(x_i.shape)
            x_i = torch.cat((torch.cat((x_i, x_player_home_pooled[i]), dim=-1), x_player_away_pooled[i]), dim=-1)
            #print(x_i.shape)
            output_list.append(x_i)

        x = torch.stack(output_list)
        x = self.post_cat(x)
        x = self.post_cat_bn(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.final_linear(x)
        return x


class TPHeteroClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_in_channels,num_classes, heads, dropout = 0.5,sag_pool=True):
        super(TPHeteroClassifier, self).__init__()
        def team_aggr(x, x_dict, size):
            return x_dict['team'].mean(dim=1)

        def player_aggr(x, x_dict, size):
            return x_dict['player'].mean(dim=1)
        self.hconv1 = HeteroConv(
            {
                ('team', 'to', 'team'): GCNConv(in_channels['team'], hidden_channels),
                ('player', 'to', 'player'): GCNConv(in_channels['player'], hidden_channels)
            },
            train_graphs[0].metadata()
        )
        
        self.hconv2 = HeteroConv({
                ('team', 'to', 'team'): GCNConv(hidden_channels, hidden_channels),
                ('player', 'to', 'player'): GCNConv(hidden_channels, hidden_channels)
            },
            train_graphs[0].metadata()
        )
 
        
        self.edge_transform = Linear(edge_in_channels, hidden_channels)
        
        self.team_linear = Linear(hidden_channels, hidden_channels)
        self.team_bn = BatchNorm1d(hidden_channels)
        
        self.player_linear = Linear(hidden_channels, hidden_channels)
        self.player_bn = BatchNorm1d(hidden_channels)
        self.teampool = SAGPooling(hidden_channels, ratio=0.5)
        self.playerpool = SAGPooling(hidden_channels, ratio=0.5)
        
        self.leaky_relu = torch.nn.LeakyReLU(0.02)
        
        self.post_cat = Linear(hidden_channels*5, hidden_channels)
        
        self.post_cat_bn = BatchNorm1d(hidden_channels)
        
        self.final_linear = Linear(hidden_channels, num_classes)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.light_dropout = torch.nn.Dropout(dropout/2)
        
        self.sag_pool = sag_pool

        

        self.lin = torch.nn.Linear(hidden_channels, num_classes)
    
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x_dict, data.edge_index_dict, data.edge_attr_dict[('team','to','team')]
        #print(x['team'].device)
        team_batch = data['team'].batch
        #print(team_batch.device)
        player_batch = data['player'].batch
        edge_attr = self.leaky_relu(self.edge_transform(edge_attr))

    
        x= self.hconv1(x, edge_index)

        
        player_to_team_edge_index = data.edge_index_dict[('player', 'to', 'team')]

        home_player_indices = (player_to_team_edge_index[1] % 2 == 0).nonzero(as_tuple=True)[0]
        away_player_indices = (player_to_team_edge_index[1] % 2 == 1).nonzero(as_tuple=True)[0]

        # Extract home and away player features
        x_player_home = x['player'][home_player_indices]
        x_player_away = x['player'][away_player_indices]

        # Perform pooling separately for home and away players
        x_player_home_pooled = global_mean_pool(x_player_home, player_batch[home_player_indices])
        x_player_away_pooled = global_mean_pool(x_player_away, player_batch[away_player_indices])

        output_list = []
        for i in range(data.num_graphs):
            x_i = torch.cat((x['team'][2*i], x['team'][2*i+1]), dim=-1)
            edge_attr_i = edge_attr[i]

            x_i = torch.cat((x_i, edge_attr_i), dim=-1)
            x_i = torch.cat((torch.cat((x_i, x_player_home_pooled[i]), dim=-1), x_player_away_pooled[i]), dim=-1)
            output_list.append(x_i)
        x = torch.stack(output_list)
        #print(x_team.shape)
        
       
        #print(x.shape)
        x = self.post_cat(x)
        x = self.post_cat_bn(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.final_linear(x)
        #x = F.softmax(x, dim=-1)
       
        return x


class TPGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_in_channels,num_classes, heads, dropout = 0.5,sag_pool=True):
        super(TPGraphClassifier, self).__init__()
        
       
        self.conv1 = HANConv(in_channels, hidden_channels, metadata = train_graphs[0].metadata(),heads=heads)
        self.conv2 = HANConv(hidden_channels, hidden_channels, metadata = train_graphs[0].metadata(),heads=heads)
        
        self.edge_transform = Linear(edge_in_channels, hidden_channels)
        
        self.team_linear = Linear(hidden_channels, hidden_channels)
        self.team_bn = BatchNorm1d(hidden_channels)
        
        self.player_linear = Linear(hidden_channels, hidden_channels)
        self.player_bn = BatchNorm1d(hidden_channels)
        self.teampool = SAGPooling(hidden_channels, ratio=0.5)
        self.playerpool = SAGPooling(hidden_channels, ratio=0.5)
        
        self.leaky_relu = torch.nn.LeakyReLU(0.02)
        
        self.post_cat = Linear(hidden_channels*5, hidden_channels)
        
        self.post_cat_bn = BatchNorm1d(hidden_channels)
        
        self.final_linear = Linear(hidden_channels, num_classes)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.light_dropout = torch.nn.Dropout(dropout/2)
        
        self.sag_pool = sag_pool
        #self.test_layer_team =Linear(278, hidden_channels)
        #self.test_layer_player = Linear(46, hidden_channels)
        

        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x_dict, data.edge_index_dict, data.edge_attr_dict[('team','to','team')]
        #print(x['team'].device)
        team_batch = data['team'].batch
        #print(team_batch.device)
        player_batch = data['player'].batch
        edge_attr = self.leaky_relu(self.edge_transform(edge_attr))

    
        x= self.conv1(x, edge_index)
        #x= self.conv2(x, edge_index)
        
        player_to_team_edge_index = data.edge_index_dict[('player', 'to', 'team')]

        home_player_indices = (player_to_team_edge_index[1] % 2 == 0).nonzero(as_tuple=True)[0]
        away_player_indices = (player_to_team_edge_index[1] % 2 == 1).nonzero(as_tuple=True)[0]

        #Extract home and away player features
        x_player_home = x['player'][home_player_indices]
        x_player_away = x['player'][away_player_indices]

        #pooling separately makes more sense and performs better in MLP
        x_player_home_pooled = global_mean_pool(x_player_home, player_batch[home_player_indices])
        x_player_away_pooled = global_mean_pool(x_player_away, player_batch[away_player_indices])

        output_list = []
        for i in range(data.num_graphs):
            x_i = torch.cat((x['team'][2*i], x['team'][2*i+1]), dim=-1)
            edge_attr_i = edge_attr[i]

            x_i = torch.cat((x_i, edge_attr_i), dim=-1)
            x_i = torch.cat((torch.cat((x_i, x_player_home_pooled[i]), dim=-1), x_player_away_pooled[i]), dim=-1)
            output_list.append(x_i)
        x = torch.stack(output_list)
        #print(x_team.shape)
        
       
        #print(x.shape)
        x = self.post_cat(x)
        x = self.post_cat_bn(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.final_linear(x)
        #x = F.softmax(x, dim=-1)
       
        return x
    