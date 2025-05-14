"""
自监督预训练模块
包含：掩码处理、模型定义、训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model.LRCN2D import LRCNAutoencoder, LRCNencoder, LRCNdecoder

def create_model(Config):
    """
    初始化自编码器模型，根据 SELF_SUPERVISED_METHOD 选择不同模型
    返回：encoder, decoder, autoencoder 三元组
    """
    if Config.SELF_SUPERVISED_METHOD == 'lrcn':
        # LRCN 模型（原有实现）
        encoder = LRCNencoder(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            use_bdc=Config.USE_BDC,
            bdc_input_dim=Config.BDC_INPUT_DIMENSION_REDUCTION
        ).to(Config.DEVICE)

        decoder = LRCNdecoder(
            output_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            use_bdc=True, #强行修改
            bdc_output_dim=Config.BDC_OUTPUT_DIMENSION_REDUCTION
        ).to(Config.DEVICE)

        autoencoder = LRCNAutoencoder(encoder, decoder).to(Config.DEVICE)
    elif Config.SELF_SUPERVISED_METHOD == 'transformer':
        # Transformer 模型
        encoder = TransformerEncoder(
            input_dim=Config.INPUT_DIM, latent_dim=Config.LATENT_DIM, d_model=128, use_rff=Config.USE_RFF,
            rff_output_dim=Config.RFF_OUTPUT_DIM, rff_gamma=Config.RFF_GAMMA,
            transformer_layers=3, transformer_heads=2
        )
        decoder = TransformerDecoder(
            latent_dim=Config.LATENT_DIM, output_dim=Config.INPUT_DIM, d_model=128,
            transformer_layers=3, transformer_heads=2
        )

        autoencoder = TransformerAutoencoder(
            encoder=encoder,
            decoder=decoder
        ).to(Config.DEVICE)
    elif Config.SELF_SUPERVISED_METHOD == 'simclr':
        # 使用 MLP 自编码器作为对比方法
        encoder = MLPEncoder(
            input_dim=Config.INPUT_DIM,
            latent_dim=Config.LATENT_DIM,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)
        decoder = MLPDecoder(
            latent_dim=Config.LATENT_DIM,
            output_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)
        autoencoder = MLPAutoencoder(encoder, decoder).to(Config.DEVICE)
    elif Config.SELF_SUPERVISED_METHOD == 'vae':
        # VAE 模型
        init_encoder = MLPEncoder(
            input_dim=Config.INPUT_DIM,
            latent_dim=Config.LATENT_DIM,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)

        encoder = VAEEncoder(init_encoder, Config.INPUT_DIM, Config.LATENT_DIM, Config.HIDDEN_DIM)
        #这里的decoder结构和MLPDecoder一样，不需要再定义
        decoder = MLPDecoder(
            latent_dim=Config.LATENT_DIM,
            output_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)
        autoencoder = VAE(encoder, decoder).to(Config.DEVICE)
    else:
        raise ValueError(f"Unknown SELF_SUPERVISED_METHOD: {Config.SELF_SUPERVISED_METHOD}")

    return encoder, decoder, autoencoder

# ------------------------- 掩码处理 ---------------------------
class MaskProcessor:
    @staticmethod
    def apply_random_mask(x_batch, mask_ratio):
        """
        批次数据掩码处理
        Args:
            x_batch: 输入张量 [batch_size, *feature_dims]
            mask_ratio: 掩码比例 (0-1)
        Returns:
            掩码后的张量，被掩码位置填充-1
        """
        flattened = x_batch.view(x_batch.size(0), -1)
        num_mask = int(flattened.size(1) * mask_ratio)

        # 生成随机掩码索引
        rand_indices = torch.rand(flattened.shape, device=x_batch.device)
        rand_indices = rand_indices.argsort(dim=1)[:, :num_mask]

        # 应用掩码
        masked = flattened.clone()
        masked[torch.arange(masked.size(0)).unsqueeze(1), rand_indices] = -1
        return masked.view_as(x_batch)

# ------------------------- simclr ---------------------------
class ProjectionHead(nn.Module):
    """SimCLR 的投影头"""
    def __init__(self, input_dim=136, hidden_dim=128, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SimCLR(nn.Module):
    """SimCLR 模型，包含编码器和投影头"""
    def __init__(self, encoder, projection_dim=128, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=136, hidden_dim=hidden_dim, output_dim=projection_dim)

    def forward(self, x1, x2):
        # 编码器处理两组增强数据
        h1 = self.encoder(x1)  # [batch_size, 136]
        h2 = self.encoder(x2)  # [batch_size, 136]
        # 投影头映射到低维空间
        z1 = self.projection_head(h1)  # [batch_size, projection_dim]
        z2 = self.projection_head(h2)  # [batch_size, projection_dim]
        return z1, z2
def data_augment_1d(x, mask_ratio=0.1, noise_level=0.01):
    """在 GPU 上进行 1D 数据增强"""
    device = x.device
    batch_size, feature_dim = x.shape

    # 随机掩码
    num_mask = int(feature_dim * mask_ratio)
    mask_indices = torch.rand((batch_size, feature_dim), device=device).topk(num_mask, dim=1).indices
    masked_x = x.clone()
    for i in range(batch_size):
        masked_x[i, mask_indices[i]] = 0

    # 添加噪声
    noise = noise_level * torch.randn_like(x, device=device)
    return masked_x + noise

def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent 损失函数（简化为双样本版本）"""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    sim11 = torch.matmul(z1, z1.T) / temperature
    sim22 = torch.matmul(z2, z2.T) / temperature
    sim12 = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    loss = nn.CrossEntropyLoss()(sim12, labels)
    return loss


def pretrain_simclr(train_loader, encoder, lr=0.001, epochs=100, device='cuda', projection_dim=128):
    """SimCLR 预训练流程"""
    simclr_model = SimCLR(encoder, projection_dim=projection_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=lr)

    with tqdm(range(epochs), desc="Pretraining (SimCLR)") as pbar:
        for epoch in pbar:
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)  # [batch_size, 204]
                # 应用数据增强
                x1 = data_augment_1d(x_batch).to(device)
                x2 = data_augment_1d(x_batch).to(device)

                optimizer.zero_grad()
                z1, z2 = simclr_model(x1, x2)  # [batch_size, projection_dim]
                loss = nt_xent_loss(z1, z2, temperature=0.5)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    return simclr_model.encoder

# ------------------------- MLP 自编码器 ---------------------------
class MLPEncoder(nn.Module):
    """
    MLP 编码器
    输入: [batch_size, INPUT_DIM]，输出: [batch_size, LATENT_DIM]
    """
    def __init__(self, input_dim=204, latent_dim=136, hidden_dim=128):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        z = self.fc2(h)
        return z

class MLPDecoder(nn.Module):
    """
    MLP 解码器
    输入: [batch_size, LATENT_DIM]，输出: [batch_size, INPUT_DIM]
    """
    def __init__(self, latent_dim=136, output_dim=204, hidden_dim=128):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.relu(self.fc1(z))
        recon = self.fc2(h)
        return recon

class MLPAutoencoder(nn.Module):
    """
    MLP 自编码器，封装编码器与解码器
    forward 返回：重构结果和编码器输出
    """
    def __init__(self, encoder, decoder):
        super(MLPAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class VAEEncoder(nn.Module):
    """
    VAE 编码器，将输入映射到隐空间的均值和对数方差，并进行重参数化
    输入: [batch_size, 204]
    输出: mu, logvar, z，其中 z 为重参数化得到的隐变量，形状为 [batch_size, 136]
    """
    def __init__(self, encoder, input_dim=204, latent_dim=136, hidden_dim=128):
        super(VAEEncoder, self).__init__()
        self.fc1 = encoder
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def fit(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # 重参数化
        return mu, logvar, z

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # 重参数化
        return z

class VAE(nn.Module):
    """
    VAE 自编码器，包含编码器和解码器
    输入: [batch_size, 204]
    输出: (重构数据, 重参数化后的隐变量, 均值, 对数方差)
    预训练时只需返回 encoder 部分，即重参数化后的隐变量 z
    """
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def fit(self, x):
        mu, logvar, z = self.encoder.fit(x)
        recon = self.decoder(z)
        return recon, z, mu, logvar

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def pretrain_VAE(train_loader, autoencoder, lr, epochs, device, mask_ratio, kl_weight=1.0):
    """
    VAE 预训练流程
    Args:
        train_loader: 训练数据加载器
        vae: VAE 模型实例（由 create_vae 得到）
        lr: 学习率
        epochs: 训练轮次
        device: 训练设备，如 'cuda' 或 'cpu'
        mask_ratio: 掩码比例（用于数据预处理）
        kl_weight: KL 散度权重系数（默认1.0）
    Returns:
        训练好的 encoder（即重参数化后的隐变量部分）
    """
    vae = autoencoder.to(device)
    vae.train()
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    with tqdm(range(epochs), desc="Pretraining (VAE)") as pbar:
        for epoch in pbar:
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                # 数据预处理：应用随机掩码
                x_masked = MaskProcessor.apply_random_mask(x_batch, mask_ratio)
                optimizer.zero_grad()
                recon, z, mu, logvar = vae.fit(x_masked)
                recon_loss = mse_loss(recon, x_batch)
                # 计算 KL 散度：鼓励隐变量接近标准正态分布
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_weight * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    # 返回训练好的 encoder 部分
    return vae.encoder

# ------------------------- Transformer ---------------------------

# 定义独立的 TransformerEncoder 模块
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=204, latent_dim=136, d_model=128,
                 rff_output_dim=58, rff_gamma=1.0, transformer_layers=3, transformer_heads=2):
        super().__init__()
        encoder_input_dim = input_dim

        # 编码器输入适配层
        self.enc_input_adapter = nn.Linear(encoder_input_dim, d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=transformer_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 编码器输出适配层
        self.enc_output_adapter = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # 输入 x: [batch_size, input_dim]
        x_enc = self.enc_input_adapter(x)  # [batch_size, d_model]
        encoded = self.transformer_encoder(x_enc.unsqueeze(1))  # [batch_size, 1, d_model]
        latent = self.enc_output_adapter(encoded.squeeze(1))  # [batch_size, latent_dim]
        return latent


# 定义独立的 TransformerDecoder 模块
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=136, output_dim=204, d_model=128,
                 transformer_layers=3, transformer_heads=2):
        super().__init__()
        # 解码器输入适配层
        self.dec_input_adapter = nn.Linear(latent_dim, d_model)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=transformer_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # 解码器输出适配层
        self.dec_output_adapter = nn.Linear(d_model, output_dim)

    def forward(self, latent, memory):
        dec_input = self.dec_input_adapter(latent)  # [batch_size, d_model]
        decoded = self.transformer_decoder(dec_input.unsqueeze(1), memory)  # [batch_size, 1, d_model]
        reconstructed = self.dec_output_adapter(decoded.squeeze(1))  # [batch_size, output_dim]
        return reconstructed


# 修改后的 TransformerAutoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # 编码
        latent = self.encoder(x)  # [batch_size, latent_dim]
        # 获取 memory（Transformer 编码器的中间输出）
        memory = self.encoder.transformer_encoder(self.encoder.enc_input_adapter(x).unsqueeze(1))
        # 解码
        reconstructed = self.decoder(latent, memory)  # [batch_size, input_dim]
        return reconstructed, latent

def pretrain_Transformermodel(train_loader, autoencoder, lr, epochs, device, mask_ratio, weight_decay=1e-4):
    """
    Transformer 自编码器预训练流程
    Args:
        train_loader: 训练数据加载器
        autoencoder: TransformerAutoencoder 实例
        lr: 学习率
        epochs: 训练轮次
        device: 训练设备
        mask_ratio: 掩码比例
        weight_decay: 权重衰减参数
    Returns:
        训练好的 autoencoder
    """
    autoencoder = autoencoder.to(device)
    autoencoder.train()
    autoencoder.encoder.train()
    autoencoder.decoder.train()

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = PretrainLoss(weight_decay=weight_decay)

    with tqdm(range(epochs), desc="Pretraining (Transformer)") as pbar:
        for epoch in pbar:
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)  # [batch_size, input_dim]
                # 应用随机掩码
                x_masked = MaskProcessor.apply_random_mask(x_batch, mask_ratio)
                optimizer.zero_grad()
                reconstructed = autoencoder(x_masked)
                loss = loss_fn(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    return autoencoder.encoder

# ============================
# 损失函数定义
# ============================
class PretrainLoss(nn.Module):
    def __init__(self, weight_decay=1e-4):
        super(PretrainLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.weight_decay = weight_decay

    def forward(self, x_hat, x, model_parameters):
        loss = self.criterion(x_hat, x)
        l2_reg = sum(torch.sum(param ** 2) for param in model_parameters)
        # 若需要使用L2正则化，可取消下行注释
        #loss = loss + self.weight_decay * l2_reg
        return loss

# ------------------------- 训练流程 ---------------------------
def pretrain_model(train_loader, model_tuple, lr, epochs, device, mask_ratio, weight_decay=1e-4):
    """
    通用预训练流程
    Args:
        train_loader: 训练数据加载器
        model_tuple: (encoder, decoder, autoencoder) 三元组
        lr: 学习率
        epochs: 训练轮次
        device: 训练设备
        mask_ratio: 掩码比例
        weight_decay: 权重衰减参数
    Returns:
        训练好的encoder
    """
    encoder, decoder, autoencoder = model_tuple
    autoencoder = autoencoder.to(device)
    encoder.train()
    decoder.train()
    autoencoder.train()

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = PretrainLoss(weight_decay=weight_decay)

    with tqdm(range(epochs), desc="Pretraining") as pbar:
        for epoch in pbar:
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                # 数据预处理：应用随机掩码
                x_masked = MaskProcessor.apply_random_mask(x_batch, mask_ratio)
                optimizer.zero_grad()
                outputs = autoencoder(x_masked, x_batch)
                loss = loss_fn(outputs, x_batch, autoencoder.parameters())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    return encoder


