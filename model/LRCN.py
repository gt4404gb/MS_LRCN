import torch.nn as nn
import torch
import src.BiRetnet as tf
import math
import torch.nn.functional as F

channel_size = 1
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", dev)

class TFClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gamma_values=[0.1, 0.3, 0.5]):
        super(TFClassifier, self).__init__()
        self.input_size = input_size

        # Transformer Encoder: 使用参数化的 gamma 值
        self.encoder_layer1 = tf.EncoderLayer(
            d_model=input_size,
            gamma=gamma_values[0],
            head_size=channel_size,
            ffn_hidden=2048,
            drop_prob=0.1,
            use_XPOS=False
        )
        self.encoder_layer2 = tf.EncoderLayer(
            d_model=math.ceil(input_size * 0.25),
            gamma=gamma_values[1],
            head_size=channel_size * 2,
            ffn_hidden=2048,
            drop_prob=0.1,
            use_XPOS=False
        )
        self.encoder_layer3 = tf.EncoderLayer(
            d_model=math.ceil(math.ceil(input_size * 0.25) * 0.25),
            gamma=gamma_values[2],
            head_size=channel_size * 4,
            ffn_hidden=2048,
            drop_prob=0.1,
            use_XPOS=False
        )

        # encoder: 卷积下采样层及 Batch Normalization
        self.conv_downsample1 = nn.Conv1d(channel_size, channel_size * 2, kernel_size=3, stride=4, padding=1)
        self.conv_downsample2 = nn.Conv1d(channel_size * 2, channel_size * 4, kernel_size=3, stride=4, padding=1)
        self.conv_downsample3 = nn.Conv1d(channel_size * 4, output_size, kernel_size=3, stride=4, padding=1)
        self.bn1 = nn.BatchNorm1d(channel_size * 2)
        self.bn2 = nn.BatchNorm1d(channel_size * 4)
        self.bn3 = nn.BatchNorm1d(output_size)

        # Estimation network（分类器）
        self.estimation_net = nn.Sequential(
            nn.Linear(output_size * 4, output_size),
        )

    def forward(self, x):
        # x: [batch_size, features]
        x = x.unsqueeze(1)  # [batch_size, 1, features]

        x_enc1 = self.encoder_layer1(x, src_mask=None)
        x = self.conv_downsample1(x_enc1)
        x = self.bn1(x)

        x_enc2 = self.encoder_layer2(x, src_mask=None)
        x = self.conv_downsample2(x_enc2)
        x = self.bn2(x)

        x_enc3 = self.encoder_layer3(x, src_mask=None)
        x = self.conv_downsample3(x_enc3)
        x = self.bn3(x)

        # 将多维特征展开为 2 维向量
        x_dec = x.reshape(x.size(0), -1)
        y_hat = self.estimation_net(x_dec)

        return x_dec, y_hat

    def init_states(self, batch_size):
        states = {
            'encoder_layer1': torch.zeros(batch_size, 1, self.input_size).to(dev),
            'encoder_layer2': torch.zeros(batch_size, 1, math.ceil(self.input_size * 0.25)).to(dev),
            'encoder_layer3': torch.zeros(batch_size, 1, math.ceil(math.ceil(self.input_size * 0.25) * 0.25)).to(dev),
        }
        return states

    def forward_recurrent(self, x, states):
        x = x.unsqueeze(1)
        x, states['encoder_layer1'] = self.encoder_layer1.forward_recurrent(x, states['encoder_layer1'], 1)
        x = self.conv_downsample1(x)
        x = self.bn1(x)

        x, states['encoder_layer2'] = self.encoder_layer2.forward_recurrent(x, states['encoder_layer2'], 2)
        x = self.conv_downsample2(x)
        x = self.bn2(x)

        x, states['encoder_layer3'] = self.encoder_layer3.forward_recurrent(x, states['encoder_layer3'], 4)
        x = self.conv_downsample3(x)
        x = self.bn3(x)

        x_dec = x.reshape(x.size(0), -1)
        y_hat = self.estimation_net(x_dec)

        return x_dec, y_hat, states

def tf_loss(y, y_hat):
    diff_loss = nn.CrossEntropyLoss()(y_hat, y)
    return diff_loss

class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight if weight is not None else 1.0

    def forward(self, preds, labels):
        labels_one_hot = F.one_hot(labels, num_classes=preds.size(1)).float()
        ce_loss = F.cross_entropy(preds, labels, reduction='none')
        pt = torch.sum(labels_one_hot * preds, dim=1) + 1e-7
        f_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return f_loss.mean()
