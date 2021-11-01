"""Classes for SimGNN modules."""
import torch
import torch.nn.functional as tnfunc
from torch.nn import Module
from torch.nn.parameter import Parameter
import numpy as np
import math

from torch_geometric.nn import GCNConv


class AvePoolingModule(Module):

    def __init__(self, args):
        super(AvePoolingModule, self).__init__()
        self.args = args

    def forward(self, embedding):
        return torch.mean(embedding, dim=0).view(-1, 1).to(self.args.device)


class AttentionModule(Module):

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.embedding_out,
                                                             self.args.embedding_out)).to(self.args.device)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0).to(self.args.device)
        transformed_global = torch.tanh(global_context).to(self.args.device)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1))).to(self.args.device)
        representation = torch.mm(torch.t(embedding), sigmoid_scores).to(self.args.device)
        return representation


class TenorNetworkModule(Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.nfeature_in = self.args.embedding_out
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.nfeature_in,
                                                             self.nfeature_in,
                                                             self.args.tensor_neurons)).to(self.args.device)
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                       2 * self.nfeature_in)).to(self.args.device)
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1)).to(self.args.device)

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.nfeature_in, -1)).to(
            self.args.device)
        scoring = scoring.view(self.nfeature_in, self.args.tensor_neurons).to(self.args.device)

        scoring = torch.mm(torch.t(scoring), embedding_2).to(self.args.device)
        combined_representation = torch.cat((embedding_1, embedding_2)).to(self.args.device)
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation).to(self.args.device)
        scores = tnfunc.relu(scoring + block_scoring + self.bias).to(self.args.device)
        return scores


class NodeGraphMatchingModule(Module):
    def __init__(self, args):
        super(NodeGraphMatchingModule, self).__init__()
        self.args = args
        self.setup_layers()
        self.init_parameters()

    def setup_layers(self):
        # Node-Graph Matching Layer
        # trainable weight matrix for multi-perspective matching function
        self.mp_w = torch.nn.Parameter(
            torch.Tensor(self.args.perspectives, self.args.embedding_out)
        ).to(self.args.device)

        # Aggregation Layer
        self.agg_bilstm = torch.nn.LSTM(input_size=self.args.perspectives, hidden_size=self.args.hidden_size,
                                        num_layers=1,
                                        bidirectional=True, batch_first=True).to(self.args.device)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.mp_w)

    def div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d

    def cosine_attention(self, v1, v2):
        # (batch, len1, len2)
        a = torch.bmm(v1, v2.permute(0, 2, 1)).to(self.args.device)

        v1_norm = v1.norm(p=2, dim=2, keepdim=True).to(self.args.device)  # (batch, len1, 1)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1).to(self.args.device)  # (batch, len2, 1)
        d = v1_norm * v2_norm
        return self.div_with_small_value(a, d)

    def multi_perspective_match_func(self, v1, v2, w):
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0).to(self.args.device)  # (1,      1,  dim, perspectives)
        v1 = w * torch.stack([v1] * self.args.perspectives, dim=3).to(
            self.args.device)  # (batch, len, dim, perspectives)
        v2 = w * torch.stack([v2] * self.args.perspectives, dim=3).to(
            self.args.device)  # (batch, len, dim, perspectives)
        return tnfunc.cosine_similarity(v1, v2, dim=2).to(self.args.device)  # (batch, len, perspectives)

    def forward(self, feature_p, feature_h):
        feature_p = feature_p.unsqueeze(0).to(self.args.device)
        feature_h = feature_h.unsqueeze(0).to(self.args.device)
        # ---------- Node-Graph Matching Layer ----------
        attention = self.cosine_attention(feature_p, feature_h).to(self.args.device)  # (batch, len_p, len_h)

        # (batch, 1, len_h, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_h = feature_h.unsqueeze(1) * attention.unsqueeze(3).to(self.args.device)
        # (batch, len_p, 1, dim) * (batch, len_p, len_h, dim) => (batch, len_p, len_h, dim)
        attention_p = feature_p.unsqueeze(2) * attention.unsqueeze(3).to(self.args.device)

        att_mean_h = self.div_with_small_value(attention_h.sum(dim=2),
                                               attention.sum(dim=2, keepdim=True)).to(
            self.args.device)  # (batch, len_p, dim)
        att_mean_p = self.div_with_small_value(attention_p.sum(dim=1),
                                               attention.sum(dim=1, keepdim=True).permute(0, 2, 1)).to(self.args.device)

        # Matching Layer
        multi_p = self.multi_perspective_match_func(v1=feature_p, v2=att_mean_h, w=self.mp_w).to(self.args.device)
        multi_h = self.multi_perspective_match_func(v1=feature_h, v2=att_mean_p, w=self.mp_w).to(self.args.device)

        match_p = multi_p
        match_h = multi_h

        # Aggregation Layer
        _, (agg_p_last, _) = self.agg_bilstm(match_p)  # (batch, seq_len, l) -> (2, batch, hidden_size)
        agg_p = agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2).to(self.args.device)

        _, (agg_h_last, _) = self.agg_bilstm(match_h)
        agg_h = agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2).to(self.args.device)

        x = torch.cat([agg_p, agg_h], dim=1).to(self.args.device)
        return x


class MT_NEGCN(Module):

    def __init__(self, args, in_features_v, in_features_e):
        super(MT_NEGCN, self).__init__()
        self.args = args
        self.in_features_e = in_features_e
        self.in_features_v = in_features_v
        self.setup_layers()

    def setup_layers(self):
        self.node_GCN_1 = GCNConv(self.in_features_v, self.args.node_nhid_1)
        self.edge_GCN_1 = GCNConv(self.in_features_e, self.args.edge_nhid_1)

        self.share_layer_1 = GCNConv(self.args.node_nhid_1, self.args.share_nhid_1)
        self.share_layer_2 = GCNConv(self.args.share_nhid_1, self.args.share_nhid_2)

        self.node_GCN_2 = GCNConv(self.args.node_nhid_1, self.args.node_nhid_2)
        self.node_GCN_3 = GCNConv(self.args.node_nhid_2, self.args.node_nhid_3)

        self.edge_GCN_2 = GCNConv(self.args.edge_nhid_1, self.args.edge_nhid_2)
        self.edge_GCN_3 = GCNConv(self.args.edge_nhid_2, self.args.edge_nhid_3)

    def forward(self, feature_v, edge_index, feature_e, trans_edge_index):
        feature_v = self.node_GCN_1(feature_v, edge_index)
        feature_v = tnfunc.relu(feature_v)
        feature_e = self.edge_GCN_1(feature_e, trans_edge_index)
        feature_e = tnfunc.relu(feature_e)

        feature_v_share = self.share_layer_1(feature_v, edge_index)
        feature_v_share = tnfunc.relu(feature_v_share)
        feature_e_share = self.share_layer_1(feature_e, trans_edge_index)
        feature_e_share = tnfunc.relu(feature_e_share)

        feature_v_share = self.share_layer_2(feature_v_share, edge_index)
        feature_v_share = tnfunc.relu(feature_v_share)
        feature_e_share = self.share_layer_2(feature_e_share, trans_edge_index)
        feature_e_share = tnfunc.relu(feature_e_share)

        feature_v = self.node_GCN_2(feature_v, edge_index)
        feature_v = tnfunc.relu(feature_v)
        feature_v = self.node_GCN_3(feature_v, edge_index)
        feature_v = tnfunc.relu(feature_v)

        feature_e = self.edge_GCN_2(feature_e, trans_edge_index)
        feature_e = tnfunc.relu(feature_e)
        feature_e = self.edge_GCN_3(feature_e, trans_edge_index)
        feature_e = tnfunc.relu(feature_e)

        feature_v = torch.cat((feature_v, feature_v_share), dim=1)
        feature_e = torch.cat((feature_e, feature_e_share), dim=1)

        return feature_v, feature_e
