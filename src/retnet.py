#基于transformer修改
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###替代多头自注意力
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)


    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)


        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        #Q = self.xpos(Q, n + 1)
        #K = self.xpos(K, n + 1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        #Q = self.xpos(Q, i * chunk_size)
        #K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        # e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (
                    1 - torch.exp(torch.linspace(math.log(1 / 128), math.log(1 / 512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """

        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
            )
            Y.append(y)
            s_ns.append(s_n)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
            )
            Y.append(y)
            r_is.append(r_i)

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is
###替代多头自注意力
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        #self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    @staticmethod
    def swiglu(input):
        # Apply the Swish activation function (SiLU in PyTorch) and an element-wise multiplication
        return F.silu(input) * input  # Using SiLU (Swish-1) as an approximation to Swish-beta

    def forward(self, x):
        x = self.linear1(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiScaleRetention(hidden_size=d_model, heads=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        '''
        self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_hidden),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model)
            )
        '''
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(x)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x

    # 新增：递归前馈方法
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        x_n: (batch_size, 1, d_model) 当前时间步的输入
        s_n_1s: 上一时间步的状态，具体结构依赖于MultiScaleRetention层
        n: 当前时间步索引
        """
        # 1. compute self attention recurrently
        _x_n = x_n
        x_n, s_ns = self.attention.forward_recurrent(x_n, s_n_1s, n)

        # 2. add and norm
        x_n = self.dropout1(x_n)
        x_n = self.norm1(x_n + _x_n)

        # 3. positionwise feed forward network
        _x_n = x_n
        x_n = self.ffn(x_n)

        # 4. add and norm
        x_n = self.dropout2(x_n)
        x_n = self.norm2(x_n + _x_n)

        # 返回当前时间步的输出和更新的状态
        return x_n, s_ns