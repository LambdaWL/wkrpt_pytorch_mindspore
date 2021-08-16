import math

from mindspore import Parameter
import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.initializer import HeUniform
from mindspore.common.initializer import Uniform
from mindspore.common.initializer import _calculate_correct_fan

from config import Config


def dense_init(in_channels, out_channels):
    """ PyTorch-like Linear Layer Initialization """
    fan_in = _calculate_correct_fan((in_channels, out_channels), "fan_in")
    bound = 1 / math.sqrt(fan_in)
    weight_init = HeUniform(math.sqrt(5))
    bias_init = Uniform(bound)

    return nn.Dense(
        in_channels=in_channels,
        out_channels=out_channels,
        weight_init=weight_init,
        bias_init=bias_init,
    )


def param_init(shape, stdv):
    """ PyTorch-like GRU Weight Initialization """
    return Parameter(initializer(
        init=Uniform(stdv),
        shape=shape,
    ))


# NOTE: Since Mindspore does not have module equivalent to torch.nn.GRUCell, 
#   we implement our own versions, with PyTorch-like weights initialization
class GRUCell(nn.Cell):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        num_chunks = 3
        stdv = 1.0 / math.sqrt(hidden_size)

        self.weight_ih = param_init((num_chunks, hidden_size, input_size), stdv)
        self.weight_hh = param_init((num_chunks, hidden_size, hidden_size), stdv)
        self.bias_ih = param_init((num_chunks, hidden_size), stdv)
        self.bias_hh = param_init((num_chunks, hidden_size), stdv)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def construct(self, input, hx):
        """ Directly from the formula, may cause performance degradation.
        Args:
            input: Tensor of shape (B, H)
            hx: Tensor of shape (B, H)
        """
        r = self.sigmoid(
            C.matmul(input, self.weight_ih[0]) + self.bias_ih[0] +
            C.matmul(hx, self.weight_hh[0]) + self.bias_hh[0]
        )
        z = self.sigmoid(
            C.matmul(input, self.weight_ih[1]) + self.bias_ih[1] + 
            C.matmul(hx, self.weight_hh[1]) + self.bias_hh[1]
        )
        n = self.tanh(
            C.matmul(input, self.weight_ih[2]) + self.bias_ih[2] + 
            r * (C.matmul(hx, self.weight_hh[2]) + self.bias_hh[2])
        )
        return (1 - z) * n + z * hx


class MindsporeModel(nn.Cell):
    def __init__(self, config: Config):
        super(MindsporeModel, self).__init__()

        self.concat_dim1 = P.Concat(1)

        self.pool_feature = dense_init(config.feature_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.gru = GRUCell(config.hidden_dim, config.hidden_dim)
        self.output1 = dense_init(config.hidden_dim, config.hidden_dim)
        self.output2 = dense_init(config.hidden_dim, config.num_class)

        # NOTE: It appears that setting dropout > 0 mysteriously causes NaN losses
        self.dropout = nn.Dropout(keep_prob=1-config.dropout)
    
    def construct(self, features):
        """
        Args:
            features: Tensor of shape (B, qlen, D)
        """
        qlen = features.shape[1]
        inputs = self.relu(self.pool_feature(features))  # (B, qlen, H)

        # Handle the first utterance separately
        prev_h = F.zeros_like(inputs[:, 0, :])
        prev_h = self.dropout(self.gru(inputs[:, 0, :], prev_h))
        hidden_states = F.expand_dims(prev_h, 1)  # (B, 1, H)
        for t in range(1, qlen):
            prev_h = self.dropout(self.gru(inputs[:, t, :], prev_h))
            hidden_states = self.concat_dim1((hidden_states, F.expand_dims(prev_h, 1)))  # (B, t+1, H)
        
        outputs = self.relu(self.output1(hidden_states))  # (B, qlen, H)
        outputs = self.output2(self.dropout(outputs))
        return outputs
