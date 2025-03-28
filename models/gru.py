import torch
import torch.nn as nn


class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + 1, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        # inputs shape: (batch_size, num_nodes)
        batch_size, num_nodes = inputs.shape
        # Reshape inputs to (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # Reshape hidden_state to (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        # Concatenate along last dim: [x, h] -> (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # Flatten for matmul: (batch_size * num_nodes, num_gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))
        # Linear operation
        outputs = concatenation @ self.weights + self.biases
        # Reshape to (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # Flatten for return if needed: (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs


class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.linear1 = GRULinear(self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.linear2 = GRULinear(self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid([x, h]W + b)
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        # Split into r, u
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh([x, (r * h)]W + b)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class GRU(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 64, **kwargs):
        super(GRU, self).__init__()
        self._input_dim = num_nodes
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for i in range(seq_len):
            output, hidden_state = self.gru_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        return last_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
