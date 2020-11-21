import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from model.se_layer import SELayer

from typing import Optional, Type, Any



class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new model instance."""
        raise NotImplementedError(
            "Models must implement the build_model_from_args method"
        )

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

def spmm(indices, values, b):
    r"""
    Args:
        indices : Tensor, shape=(2, E)
        values : Tensor, shape=(E,)
        shape : tuple(int ,int)
        b : Tensor, shape=(N, )
    """
    output = b.index_select(0, indices[1]) * values.unsqueeze(-1)
    output = torch.zeros_like(b).scatter_add_(0, indices[0].unsqueeze(-1).repeat(1, b.shape[1]), output)
    return output

class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for only sparse region backpropataion layer.
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # in_features: 300
        # out_features: 16
        nn.init.xavier_normal_(self.W.data)

        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()


    def forward(self, input, adj):
        # input: [4458,300]
        # adj: [4458,4458]
        N = input.size()[0]
        edge = torch.LongTensor(adj.nonzero())
        # print('edge',edge.shape)
        # print(edge)
        # [2,59116]

        h = torch.mm(input, self.W)
        # equ 10?
        # h: N x out([4458, 16])
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        # equ 12
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj, edge_attr=None):
        edge_index = torch.LongTensor(adj.nonzero())

        if edge_attr is None:
            # edge_attr = adj.data
            edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
        adj = torch.sparse_coo_tensor(
            edge_index,
            edge_attr,
            (input.shape[0], input.shape[0]),dtype=torch.float
        ).to(input.device)
        support = torch.mm(input, self.weight)
        # print('adj,',adj.dtype)
        # print('support,',support.dtype)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

def add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes):
    N = num_nodes
    row, col = edge_index[0], edge_index[1]
    # print('row,',row)
    # print('col',col)

    mask = row != col
    # print('mask,',mask)

    loop_index = torch.arange(0, N, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    inv_mask = ~mask
    # print('inv_mask, ',inv_mask)
    loop_weight = torch.full(
        (N,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device
    )
    remaining_edge_weight = edge_weight[inv_mask]
    if remaining_edge_weight.numel() > 0:
        loop_weight[row[inv_mask]] = remaining_edge_weight
    edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    return edge_index, edge_weight

class TKipfGCN(BaseModel):
    r"""The GCN model from the `"Semi-Supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    Args:
        num_features (int) : Number of input features.
        num_classes (int) : Number of classes.
        hidden_size (int) : The dimension of node representation.
        dropout (float) : Dropout rate for model training.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.num_features, args.hidden_size, args.num_classes, args.dropout)

    def __init__(self, nfeat, nclass, nhid=64, dropout=0.5):
        super(TKipfGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj_orgin):

        device = x.device
        adj = torch.LongTensor(adj_orgin.nonzero()).to(device)
        # print('adj origin,',adj_orgin.shape)
        # print('adj,',adj.shape)
        # adj_values = torch.ones(adj.shape[1]).to(device)
        adj_values = torch.tensor(adj_orgin.data).float().to(device)

        adj, adj_values = add_remaining_self_loops(adj, adj_values, 1, x.shape[0])
        # print('adj after self loop, ',adj.shape)
        # print('adj_values after self loop, ', adj_values.shape)
        deg = spmm(adj, adj_values, torch.ones(x.shape[0], 1).to(device)).squeeze()
        # print('deg, ',deg.shape)
        deg_sqrt = deg.pow(-1 / 2)
        adj_values = deg_sqrt[adj[1]] * adj_values * deg_sqrt[adj[0]]
        # print('adj_value, ', adj_values)
        x = F.relu(self.gc1(x, adj_orgin))
        # print('x,', x.shape)
        # h1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_orgin)
        # print('x,', x.shape)
        # x = F.relu(x)
        # x = torch.sigmoid(x)
        # return x
        # h2 = x
        return F.log_softmax(x, dim=-1)

    def loss(self, data):
        return F.nll_loss(
            self.forward(data.x, data.edge_index)[data.train_mask],
            data.y[data.train_mask],
        )

    def predict(self, data):
        return self.forward(data.x, data.edge_index)

class GAT(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        # adj.shape[0]
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features = nfeat,
                                                        out_features= hidden,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nb_heads)])
        # self.attentions = nn.ModuleList([TKipfGCN(nfeat=nfeat,
        #                                           nclass=hidden,
        #                                           dropout=dropout) for _ in range(nb_heads)
        # ])
        self.attentions = nn.ModuleList([
            GraphConvolution(in_features=nfeat,
                             out_features=hidden,
                             bias=True) for _ in range(nb_heads)
        ])
        
        # self.out_att = SpGraphAttentionLayer(hidden * nb_heads,
        #                                       n_output,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False)
        # self.out_att = TKipfGCN(nfeat=hidden*nb_heads,
        #                         nclass=n_output,
        #                         dropout=dropout)
        self.out_att = GraphConvolution(in_features=hidden*nb_heads,
                                        out_features=n_output,
                                        bias=False)

    def forward(self, X_tid):
        # print('uV',self.uV)
        # self.uV: 4458
        # print('self.adj',self.adj)
        # print(self.adj.shape)
        # [4458,4458], adj matrix
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        # print('X',X.shape)
        # X: [4458,300]
        X = self.dropout(X)

        X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
        X = self.dropout(X)
        # print('X after SpGraph', X.shape)
        # X: [4458, 128]

        X = F.elu(self.out_att(X, self.adj))
        X_ = X[X_tid]
        return X_

class DrGAT(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(DrGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)
        self.se1 = SELayer(nfeat, se_channels=int(np.sqrt(nfeat)))
        self.se2 = SELayer(
            hidden * nb_heads, se_channels=int(np.sqrt(hidden * nb_heads))
        )

        self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features = nfeat,
                                                        out_features= hidden,
                                                        dropout=dropout,
                                                        alpha=alpha,
                                                        concat=True) for _ in range(nb_heads)])
        
        self.out_att = SpGraphAttentionLayer(hidden * nb_heads,
                                              n_output,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, X_tid):
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        X = self.dropout(X)
        X = self.se1(X)

        X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
        X = self.dropout(X)
        X = self.se2(X)

        X = F.elu(self.out_att(X, self.adj))
        X_ = X[X_tid]
        return X_

class GAT_GCN(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(GAT_GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        # adj.shape[0]
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.attentions = nn.ModuleList([
            GraphConvolution(in_features=nfeat,
                             out_features=hidden,
                             bias=True) for _ in range(nb_heads)
        ])


        self.out_att = GraphConvolution(in_features=hidden * nb_heads,
                                        out_features=n_output,
                                        bias=False)

    def forward(self, X_tid):
        # print('uV',self.uV)
        # self.uV: 4458
        # print('self.adj',self.adj)
        # print(self.adj.shape)
        # [4458,4458], adj matrix
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        # print('X',X.shape)
        # X: [4458,300]
        X = self.dropout(X)

        X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
        X = self.dropout(X)
        # print('X after SpGraph', X.shape)
        # X: [4458, 128]

        X = F.elu(self.out_att(X, self.adj))
        X_ = X[X_tid]
        return X_

class GAT_TKipfGCN(nn.Module):

    def __init__(self, nfeat, uV, adj, hidden=16, nb_heads=8, n_output=300, dropout=0.5, alpha=0.3):
        """Sparse version of GAT."""
        super(GAT_TKipfGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.uV = uV
        # adj.shape[0]
        self.adj = adj
        self.user_tweet_embedding = nn.Embedding(self.uV, 300, padding_idx=0)
        init.xavier_uniform_(self.user_tweet_embedding.weight)

        self.attentions = nn.ModuleList([TKipfGCN(nfeat=nfeat,
                                                  nclass=hidden,
                                                  dropout=dropout) for _ in range(nb_heads)
        ])

        self.out_att = TKipfGCN(nfeat=hidden*nb_heads,
                                nclass=n_output,
                                dropout=dropout)

    def forward(self, X_tid):
        # print('uV',self.uV)
        # self.uV: 4458
        # print('self.adj',self.adj)
        # print(self.adj.shape)
        # [4458,4458], adj matrix
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda())
        # print('X',X.shape)
        # X: [4458,300]
        X = self.dropout(X)

        X = torch.cat([att(X, self.adj) for att in self.attentions], dim=1)
        X = self.dropout(X)
        # print('X after SpGraph', X.shape)
        # X: [4458, 128]

        X = F.elu(self.out_att(X, self.adj))
        X_ = X[X_tid]
        return X_
