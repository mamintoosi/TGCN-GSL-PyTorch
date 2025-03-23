import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class GCN(nn.Module):
    def __init__(self, adj, seq_len: int, hidden_dim: int = 64, **kwargs):
        super(GCN, self).__init__()
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = seq_len  # seq_len for each input series
        self._output_dim = hidden_dim
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # Move dimension: (num_nodes, batch_size, seq_len)
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        # Flatten: (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # Multiply by Laplacian: (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # Reshape: (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # Flatten again: (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # Multiply by param: (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # Reshape: (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # Permute back: (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "hidden_dim": self._output_dim,
        }
