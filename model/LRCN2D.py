import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.BiRetnet as tf  # 请确保该文件路径无误
from src.deepBDC import BDC

##############################
# 全局参数与默认配置
##############################
# （可在这里统一修改，避免在中间代码中写死）
DEFAULT_CHANNEL_SIZE = 1  # 初始通道数
DEFAULT_GAMMAS = [0.1, 0.3, 0.5]  # 三个Transformer Encoder/Decoder层的 gamma
DEFAULT_HEAD_SIZES = [1, 2, 4]    # 三个Transformer Encoder/Decoder层的多头数
DEFAULT_FFN_HIDDEN = 2048         # Transformer内部FFN隐层大小
DEFAULT_DROP_PROB = 0.1           # Dropout概率
DEFAULT_USE_XPOS = False          # 是否使用XPOS
DEFAULT_KERNEL_SIZE = 2           # 卷积/反卷积核大小
DEFAULT_STRIDE = 2                # 卷积/反卷积步幅
DEFAULT_PADDING = 0               # 卷积/反卷积padding
DEFAULT_OUTPUT_PADDING = 1        # ConvTranspose时的output_padding（仅示例中第1次上采样用到）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LRCNencoder(nn.Module):
    """
    LRCN的编码器部分：
      1. 通过若干Transformer Encoder Layer提取特征
      2. 采用Conv1d进行多次下采样（替代MaxPooling），并在过程中做BatchNorm
      3. 输出形状经过view变换后，得到 [batch_size, channels, hidden_size, hidden_size]
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 5,
        channel_size: int = DEFAULT_CHANNEL_SIZE,
        gammas: list = DEFAULT_GAMMAS,
        head_sizes: list = DEFAULT_HEAD_SIZES,
        ffn_hidden: int = DEFAULT_FFN_HIDDEN,
        drop_prob: float = DEFAULT_DROP_PROB,
        use_xpos: bool = DEFAULT_USE_XPOS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        # 新增BDC参数
        use_bdc: bool = True,          # 是否启用BDC
        bdc_input_dim: int = 16      # BDC输出维度
    ):
        """
        :param input_size:  输入序列长度（如：204）
        :param hidden_size: 编码器最终会把特征reshape成 hidden_size x hidden_size
        :param channel_size: 初始通道数
        :param gammas:   三个Transformer层对应的gamma值
        :param head_sizes: 三个Transformer层对应的多头数量
        :param ffn_hidden: Transformer层内部FFN的隐层大小
        :param drop_prob: Dropout概率
        :param use_xpos:  是否使用XPOS位置编码
        :param kernel_size: 下采样卷积核大小
        :param stride:  下采样卷积步幅
        :param padding: 下采样卷积padding
        """
        super(LRCNencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.channel_size = channel_size

        # ========== 三个Transformer Encoder层 ==========
        # d_model 的设置逻辑和原代码保持一致
        # encoder_layer1: d_model = input_size
        # encoder_layer2: d_model = floor(input_size * 0.5)
        # encoder_layer3: d_model = floor( ceil(input_size * 0.5) * 0.5 )
        self.encoder_layer1 = tf.EncoderLayer(
            d_model=input_size,
            gamma=gammas[0],
            head_size=head_sizes[0],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )
        self.encoder_layer2 = tf.EncoderLayer(
            d_model=math.floor(input_size * 0.5),
            gamma=gammas[1],
            head_size=head_sizes[1],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )
        self.encoder_layer3 = tf.EncoderLayer(
            d_model=math.floor(math.ceil(input_size * 0.5) * 0.5),
            gamma=gammas[2],
            head_size=head_sizes[2],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )

        # ========== 卷积下采样层 + BN ==========
        # channel 的变化：1 -> 4 -> 8 -> 16， 原代码对应 channel_size, 4*channel_size, ...
        self.conv_downsample1 = nn.Conv1d(
            in_channels=channel_size,
            out_channels=channel_size * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv_downsample2 = nn.Conv1d(
            in_channels=channel_size * 4,
            out_channels=channel_size * 8,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv_downsample3 = nn.Conv1d(
            in_channels=channel_size * 8,
            out_channels=channel_size * 16,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(channel_size * 4)
        self.bn2 = nn.BatchNorm1d(channel_size * 8)
        self.bn3 = nn.BatchNorm1d(channel_size * 16)

        self.use_bdc = use_bdc
        if self.use_bdc:
            self.bdc_module = BDC(
                is_vec=True,
                input_dim=(channel_size*16, hidden_size, hidden_size),  # 输入形状 [C, H, W]
                dimension_reduction=bdc_input_dim,
                activate='relu'
            )
            self.bdc_input_dim = bdc_input_dim
        else:
            self.bdc_module = None
            # Add projection layer for use_bdc=False
            self.projection = nn.Linear(
                (self.channel_size * 16) * self.hidden_size * self.hidden_size,
                136
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, input_size]
        :return:  [batch_size, channel_size*16, hidden_size, hidden_size]
        """
        # 在第2维(通道维)插入channel_size=1，原始 x 维度: [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)

        # ========== 第1层Transformer & Conv下采样 ==========
        x_enc1 = self.encoder_layer1(x, src_mask=None)
        # x_enc1 形状仍是 [B, channel_size=1, length=input_size]
        x = self.conv_downsample1(x_enc1)
        # 下采样后长度变为 length/2
        x = self.bn1(x)

        # ========== 第2层Transformer & Conv下采样 ==========
        x_enc2 = self.encoder_layer2(x, src_mask=None)
        x = self.conv_downsample2(x_enc2)
        x = self.bn2(x)

        # ========== 第3层Transformer & Conv下采样 ==========
        x_enc3 = self.encoder_layer3(x, src_mask=None)
        x = self.conv_downsample3(x_enc3)
        x = self.bn3(x)

        # ========== reshape到 [B, C, hidden_size, hidden_size] ==========
        batch, channels, length = x.shape
        # 最终reshape后的特征图 [B, C, H, W]
        x_reshaped = x.view(batch, channels, self.hidden_size, self.hidden_size)

        # 新增：通过BDC处理特征
        if self.use_bdc:
            x_reshaped = self.bdc_module(x_reshaped)  # 输出形状 [B, bdc_output_dim]
        else:
            x_flatten = x_reshaped.view(batch, -1)  # [batch_size, channel_size*16 * hidden_size * hidden_size]
            x_reshaped = self.projection(x_flatten)  # [batch_size, bdc_input_dim]

        return x_reshaped


class LRCNdecoder(nn.Module):
    """
    LRCN的解码器部分：
      1. 反卷积(ConvTranspose1d)进行多次上采样，并在过程中做BatchNorm
      2. 通过若干Transformer Encoder Layer处理恢复后的序列特征
      3. 最终恢复回 [batch_size, input_size] 的形状（去掉通道维后）
    """
    def __init__(
        self,
        output_size: int,
        hidden_size: int = 5,
        channel_size: int = DEFAULT_CHANNEL_SIZE,
        gammas: list = DEFAULT_GAMMAS,
        head_sizes: list = DEFAULT_HEAD_SIZES,
        ffn_hidden: int = DEFAULT_FFN_HIDDEN,
        drop_prob: float = DEFAULT_DROP_PROB,
        use_xpos: bool = DEFAULT_USE_XPOS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        output_padding: int = DEFAULT_OUTPUT_PADDING,
        # 新增参数
        use_bdc: bool = True,
        bdc_output_dim: int = 136
    ):
        """
        :param output_size: 解码后希望回到的序列长度（与encoder输入的input_size对应）
        :param hidden_size: Encoder输出特征图的 spatial 大小 (hidden_size x hidden_size)
        :param channel_size: 编码器中使用的通道基础
        :param gammas:   三个Transformer层对应的gamma值
        :param head_sizes: 三个Transformer层对应的多头数量
        :param ffn_hidden: Transformer层内部FFN的隐层大小
        :param drop_prob: Dropout概率
        :param use_xpos:  是否使用XPOS位置编码
        :param kernel_size: 上采样反卷积核大小
        :param stride:  上采样反卷积步幅
        :param padding: 上采样反卷积padding
        :param output_padding: 反卷积时的output_padding（仅示例中第一层使用1）
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.channel_size = channel_size

        # ========== 反卷积上采样 ==========
        # channel 的变化：16 -> 8 -> 4 -> 1，与encoder对称
        self.conv_upsample1 = nn.ConvTranspose1d(
            in_channels=channel_size * 16,
            out_channels=channel_size * 8,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding  # 仅在第一层示例中设置1
        )
        self.conv_upsample2 = nn.ConvTranspose1d(
            in_channels=channel_size * 8,
            out_channels=channel_size * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv_upsample3 = nn.ConvTranspose1d(
            in_channels=channel_size * 4,
            out_channels=channel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(channel_size * 8)
        self.bn2 = nn.BatchNorm1d(channel_size * 4)
        self.bn3 = nn.BatchNorm1d(channel_size)

        # ========== 三个Transformer Encoder层 (与Encoder对应，但逆向顺序) ==========
        # 注意这里 d_model 的设置和Encoder对应即可
        self.decoder_layer3 = tf.EncoderLayer(
            d_model=math.floor(math.ceil(output_size * 0.5) * 0.5),
            gamma=gammas[2],
            head_size=head_sizes[2],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )
        self.decoder_layer2 = tf.EncoderLayer(
            d_model=math.floor(output_size * 0.5),
            gamma=gammas[1],
            head_size=head_sizes[1],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )
        self.decoder_layer1 = tf.EncoderLayer(
            d_model=output_size,
            gamma=gammas[0],
            head_size=head_sizes[0],
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            use_XPOS=use_xpos
        )

        if use_bdc:
            # 计算原始Encoder输出的特征图大小（如 [B, 16, 5, 5]）
            self.original_channels = channel_size * 16
            self.original_hidden = hidden_size

            # 全连接层将BDC输出映射回原始形状
            self.fc_recover = nn.Sequential(
                nn.Linear(bdc_output_dim, self.original_channels * hidden_size * hidden_size),
                nn.ReLU(),
                nn.Unflatten(1, (self.original_channels, hidden_size, hidden_size))  # 恢复 [B, C, H, W]
            )
        else:
            self.fc_recover = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, channel_size*16, hidden_size, hidden_size]
        :return:  [batch_size, output_size]
        """
        # 先把 [B, C, H, W] 拉成 [B, C, L=H*W]
        # 新增：若使用BDC，需先通过全连接恢复特征图
        if self.fc_recover is not None:
            x = self.fc_recover(x)  # [B, C, H, W]

        batch, channels, h, w = x.shape
        x = x.view(batch, channels, h * w)  # [B, C, hidden_size * hidden_size]

        # ========== 第1次上采样 + Transformer ==========
        x = self.conv_upsample1(x)  # [B, channel_size*8, L*2]
        x = self.bn1(x)
        x_dec3 = self.decoder_layer3(x, src_mask=None)

        # ========== 第2次上采样 + Transformer ==========
        x = self.conv_upsample2(x_dec3)  # [B, channel_size*4, L*4]
        x = self.bn2(x)
        x_dec2 = self.decoder_layer2(x, src_mask=None)

        # ========== 第3次上采样 + Transformer ==========
        x = self.conv_upsample3(x_dec2)  # [B, channel_size, L*8]
        x = self.bn3(x)
        x_dec1 = self.decoder_layer1(x, src_mask=None)

        # 去掉 channel 维度，返回 [B, length]，length=output_size
        x_recon = x_dec1.squeeze(1)
        return x_recon


class LRCNAutoencoder(nn.Module):
    """
    LRCN自动编码器的整体结构，将Encoder与Decoder结合
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        :param x: [batch_size, input_size]
        :return:  [batch_size, output_size] （等同于input_size）
        """
        encoded = self.encoder(x)
        x_recon = self.decoder(encoded)
        return x_recon


def modelinit(
    input_size: int = 70,
    hidden_size: int = 10,
    # 新增BDC参数
    use_bdc: bool = True,
    bdc_input_dim: int = 16,
    bdc_output_dim: int = 136,
    channel_size: int = DEFAULT_CHANNEL_SIZE,
    gammas: list = DEFAULT_GAMMAS,
    head_sizes: list = DEFAULT_HEAD_SIZES,
    ffn_hidden: int = DEFAULT_FFN_HIDDEN,
    drop_prob: float = DEFAULT_DROP_PROB,
    use_xpos: bool = DEFAULT_USE_XPOS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    stride: int = DEFAULT_STRIDE,
    padding: int = DEFAULT_PADDING,
    output_padding: int = DEFAULT_OUTPUT_PADDING,
    device=DEVICE
) -> LRCNAutoencoder:
    """
    构建LRCN自动编码器模型，方便一次性初始化。
    使用时可一次性指定 input_size, hidden_size, 以及其他参数。
    """
    encoder = LRCNencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        use_bdc=use_bdc,
        bdc_input_dim=bdc_input_dim,
        channel_size=channel_size,
        gammas=gammas,
        head_sizes=head_sizes,
        ffn_hidden=ffn_hidden,
        drop_prob=drop_prob,
        use_xpos=use_xpos,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    ).to(device)

    decoder = LRCNdecoder(
        output_size=input_size,
        hidden_size=hidden_size,
        #use_bdc=use_bdc,
        bdc_output_dim=bdc_output_dim,
        channel_size=channel_size,
        gammas=gammas,
        head_sizes=head_sizes,
        ffn_hidden=ffn_hidden,
        drop_prob=drop_prob,
        use_xpos=use_xpos,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding
    ).to(device)

    model = LRCNAutoencoder(encoder, decoder).to(device)
    return model


##############################
# ========== 测试或使用示例 ==========
##############################
if __name__ == "__main__":
    # 指定输入序列长度和隐藏特征图大小
    input_size = 204
    hidden_size = 5

    # 初始化模型
    model = modelinit(input_size=input_size, hidden_size=hidden_size, device=DEVICE)
    print(model)

    # 随机构造一批输入
    x = torch.rand(8, input_size).to(DEVICE)  # 8条样本，每条长度为 input_size
    x_recon = model(x)

    print("x.shape:", x.shape)           # [8, 204]
    print("x_recon.shape:", x_recon.shape)  # [8, 204]
    loss = F.mse_loss(x_recon, x)
    print("MSE Loss:", loss.item())
