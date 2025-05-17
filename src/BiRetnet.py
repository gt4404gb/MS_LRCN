#基于transformer修改的
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.XPOS import XPOS
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 数据放到gpu上还是cpu上
print("device", dev)

###替代多头自注意力
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False,use_XPOS=False):
        super(SimpleRetention, self).__init__()
        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.xpos = XPOS(head_size)
        self.use_XPOS = use_XPOS
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, X):
        Q = (X @ self.W_Q)
        K = (X @ self.W_K)
        if self.use_XPOS:
            Q = self.xpos(Q)
            K = self.xpos(K, downscale=True)

        V = X @ self.W_V

        # 使用_get_D函数生成衰减矩阵D
        D = self._get_D(X.size(1)).to(X.device)

        # 计算注意力得分并应用衰减矩阵D
        attention_scores = Q @ K.permute(0, 2, 1)
        attention_scores = attention_scores * D  # 应用衰减因子

        # 计算最终的输出
        ret = self.swish(attention_scores) @ V

        return ret

    def forward_recurrent(self, x_n, s_n_1, backward=False):
        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)
        if self.use_XPOS:
            Q = self.xpos(Q)
            K = self.xpos(K, downscale=True)

        V = x_n @ self.W_V
        KV = self.swish(K.permute(0, 2, 1)) @ V
        if backward:
            s_n = self.gamma * s_n_1 + KV
        else:
            s_n = self.gamma * s_n_1 + KV
        x_n = Q @ s_n
        return x_n, s_n
    '''
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    '''
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(0)
        m = torch.arange(sequence_length).unsqueeze(1)
        # 使用绝对差值来创建双向衰减矩阵
        D_bi = self.gamma ** torch.abs(n - m)
        return D_bi


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, gamma,head_size,use_XPOS,double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        #self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.use_XPOS =use_XPOS

        #self.gammas = (
        #            1 - torch.exp(torch.linspace(math.log(1 / 128), math.log(1 / 512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, head_size, use_XPOS=self.use_XPOS, double_v_dim=double_v_dim)
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
    def __init__(self, d_model, ffn_hidden, drop_prob,gamma,head_size,use_XPOS):
        super(EncoderLayer, self).__init__()
        self.attention = MultiScaleRetention(hidden_size = d_model, heads=1,head_size=head_size,gamma=gamma,use_XPOS=use_XPOS)
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
        # 初始化一个列表来保存每个时间步的状态
        outputs = []
        for t in range(n):
            # 从x_n中切出当前时间步的数据，形状为[512, 1, 4]
            x_n_t = x_n[:, t:t + 1, :]  # 这将选取第t个时间步长的数据
            # 调用forward_recurrent处理当前时间步长的数据并更新状态
            output, s_n_t = self.attention.forward_recurrent(x_n_t, s_n_1s, t)
            # 保存当前时间步的状态
            outputs.append(output)
            # 更新s_n_1s以便于下一次迭代使用
            s_n_1s = s_n_t

            # 将状态列表沿时间步轴堆叠，形成最终的状态张量
        x_n = torch.cat(outputs, dim=1)

        # 现在s_n_1s的形状应该是[512, 4, 4]，其中每个时间步的状态都被保存

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
        return x_n, s_n_1s

def to_square_matrix(V):
    # 获取V的形状
    batch_size, _, max_dim = V.shape

    # 计算最大维度（考虑最后两个维度的最大值）
    target_dim = max(V.size(-2), V.size(-1))

    # 创建一个全为0的方阵，大小为[batch_size, target_dim, target_dim]
    square_matrix = torch.zeros((batch_size, target_dim, target_dim))

    # 将V的内容复制到方阵的适当位置
    if V.size(-1) == target_dim:
        # 如果最后一个维度是最大的，我们将V复制到方阵的前面几行
        square_matrix[:, :V.size(-2), :] = V
    else:
        # 如果倒数第二个维度是最大的，我们将V复制到方阵的左上角
        square_matrix[:, :, :V.size(-1)] = V

    return square_matrix