import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn.conv import GATConv
from model.se_layer import SELayer

class DrGAT(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(DrGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.conv1 = GATConv(
            nfeat, hidden, heads=nb_heads, dropout=dropout
        )
        self.conv2 = GATConv(hidden * nb_heads, n_output, dropout=dropout)
        self.se1 = SELayer(nfeat, se_channels=int(np.sqrt(nfeat)))
        self.se2 = SELayer(
            hidden * nb_heads, se_channels=int(np.sqrt(hidden * nb_heads))
        )

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        X = self.dropout(X)
		
	X = self.se1(X)

        X = F.elu(self.conv1(X, self.adj))
        X = self.dropout(X)
		
	X = self.se2(X)

        X = F.elu(self.conv2(X, self.adj))
        X_ = X[X_tid]
		
		
        return X_

