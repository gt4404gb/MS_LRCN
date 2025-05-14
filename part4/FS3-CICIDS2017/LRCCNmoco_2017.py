import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# ============================
# 1. 动量对比学习模型定义
# ============================

class MoCoEncoderKNN(nn.Module):
    """
    改进版：每个类别维护独立队列，构造对比样本，避免数据不平衡问题
    所有配置参数均由外部传入（例如来自主配置文件）
    """
    def __init__(self, base_encoder, feature_dim, num_classes, K, m, T, top_k, device):
        """
        参数：
            base_encoder: 基础编码器（例如预训练后的 encoder）
            feature_dim: 特征维度（如 BDC 输出的向量维度）
            num_classes: 类别数
            K: 每个类别的队列长度（队列大小）
            m: 动量更新参数
            T: 温度参数
            top_k: 负样本选择上限
            device: 运行设备
        """
        super(MoCoEncoderKNN, self).__init__()
        self.device = device
        self.m = m
        self.T = T
        self.top_k = top_k
        self.num_classes = num_classes
        self.K = K

        # 主编码器 (query encoder)
        self.encoder_q = base_encoder.to(self.device)

        # 动量编码器 (key encoder)
        self.encoder_k = copy.deepcopy(base_encoder).to(self.device)
        self.encoder_k.load_state_dict(base_encoder.state_dict())
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # ===== 创建按类别划分的队列 =====
        # feature_queues: Tensor of shape (num_classes, K, feature_dim)
        self.register_buffer("feature_queues", torch.randn(num_classes, K, feature_dim))
        self.feature_queues = F.normalize(self.feature_queues, p=2, dim=2)  # 按特征维度归一化

        # queue_ptrs: 每个类别的队列指针，形状 (num_classes,)
        self.register_buffer("queue_ptrs", torch.zeros(num_classes, dtype=torch.long))

    def forward(self, x_q, return_q=False):
        q_feat = self.encoder_q(x_q)
        logits = self.classifier(q_feat)
        if return_q:
            return logits, q_feat
        else:
            return logits

    @torch.no_grad()
    def update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def momentum_forward(self, x_k):
        self.update_key_encoder()
        return self.encoder_k(x_k)

    def enqueue_dequeue(self, keys, labels):
        """
        改进后的入队逻辑：限制每个类别的入队样本数不超过队列容量 K
        """
        for c in range(self.num_classes):
            mask = (labels == c)
            c_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if c_indices.numel() == 0:
                continue

            # 提取当前类别的特征（限制最多取 K 个样本）
            c_keys = keys[c_indices]
            num_c = len(c_indices)
            ptr = self.queue_ptrs[c].item()

            # 如果样本数超过队列容量，则截断
            if num_c > self.K:
                c_keys = c_keys[:self.K]
                num_c = self.K

            # 队列更新逻辑
            if ptr + num_c <= self.K:
                self.feature_queues[c, ptr:ptr + num_c] = c_keys
                self.queue_ptrs[c] = (ptr + num_c) % self.K
            else:
                remaining = self.K - ptr
                self.feature_queues[c, ptr:] = c_keys[:remaining]
                self.feature_queues[c, :num_c - remaining] = c_keys[remaining:]
                self.queue_ptrs[c] = num_c - remaining

    @torch.no_grad()
    def select_pos_neg_sample(self, q_proj: torch.Tensor, y_batch: torch.Tensor):
        """
        改进后的对比样本选择：
          1. 正样本来自同类队列
          2. 负样本来自其他类队列
        """
        batch_size = y_batch.size(0)
        # 展开所有队列 (num_classes * K, feature_dim)
        all_features = self.feature_queues.view(-1, self.feature_queues.size(-1))
        # 生成队列标签 (num_classes*K,)
        all_labels = torch.arange(self.num_classes, device=self.device).repeat_interleave(self.K)
        # 计算相似度矩阵 (batch_size, num_classes*K)
        cos_sim = torch.einsum('nc,kc->nk', q_proj, all_features)
        # 构建正样本掩码
        pos_mask = (all_labels.unsqueeze(0) == y_batch.unsqueeze(1))
        # 正样本得分（同类中取最大）
        pos_sim = torch.where(pos_mask, cos_sim, torch.tensor(float('-inf')).to(self.device))
        pos_score, _ = pos_sim.max(dim=1, keepdim=True)
        # 负样本得分（跨类取 top_k）
        neg_sim = torch.where(~pos_mask, cos_sim, torch.tensor(float('-inf')).to(self.device))
        neg_topk = min(self.top_k, neg_sim.size(1))
        neg_score, _ = neg_sim.topk(neg_topk, dim=1)
        # 组合 logits 并缩放
        logits_con = torch.cat([pos_score, neg_score], dim=1) / self.T
        return logits_con

# ============================
# 2. 对比学习相关损失（可选）
# ============================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin, weight_decay, temperature):
        """
        参数：
            margin: 对比损失中的 margin
            weight_decay: 权重衰减参数
            temperature: 温度参数
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight_decay = weight_decay
        self.temperature = temperature

    def forward(self, sample_embedding, label_embedding, negative, model_parameters):
        l_pos = torch.einsum('nc,nc->n', [sample_embedding, label_embedding]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [sample_embedding, negative])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss_nce = nn.functional.cross_entropy(logits, labels)
        l2_reg = sum(torch.sum(param ** 2) for param in model_parameters)
        loss = loss_nce + self.weight_decay  # 如需乘以 l2_reg，可修改此处
        return loss

# ============================
# 3. 对比学习训练流程
# ============================
def ContrastiveLearningKNN(
    encoder,
    train_loader,
    epochs,
    device,
    class_number,
    contrastive_rate,
    feature_dim,
    top_k,
    queue_size,
    lr,
    m,
    T
):
    """
    使用 KNN 对比学习（带 BDC），结合 MoCo 动量机制
    参数：
        encoder: 主编码器（预训练/预处理后的 encoder）
        train_loader: 训练数据加载器
        epochs: 训练轮次
        device: 训练设备
        class_number: 类别数
        contrastive_rate: 对比损失权重（与分类损失的权重比例）
        feature_dim: 特征维度（如 BDC 输出维度）
        top_k: 负样本选择的上限
        queue_size: 每个类别队列长度
        lr: 优化器学习率
        m: 动量更新参数
        T: 温度参数
    """
    # 1. 初始化 MoCo 编码器
    moco_encoder = MoCoEncoderKNN(
        base_encoder=encoder,
        feature_dim=feature_dim,
        num_classes=class_number,
        K=queue_size,
        m=m,
        T=T,
        top_k=top_k,
        device=device
    ).to(device)

    # 2. 定义损失（这里使用分类损失，同时构造对比学习 logits）
    criterion_cls = nn.CrossEntropyLoss()

    # 3. 优化器（仅更新 encoder_q 与 classifier 参数）
    parameters_to_optimize = list(moco_encoder.encoder_q.parameters()) + list(moco_encoder.classifier.parameters())
    optimizer = optim.Adam(parameters_to_optimize, lr=lr)

    # 4. 训练循环
    moco_encoder.train()
    progress_bar = tqdm(range(epochs), desc="KNN对比学习 Progress")
    for epoch in progress_bar:
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # 4.1 主编码器前向计算
            logits, q_proj = moco_encoder(x_batch, return_q=True)
            loss_cls = criterion_cls(logits, y_batch)

            # 4.2 动量编码器提取特征，并更新队列
            k_proj = moco_encoder.momentum_forward(x_batch)
            moco_encoder.enqueue_dequeue(k_proj, y_batch)

            # 4.3 获取对比学习 logits 并计算对比损失
            logits_con = moco_encoder.select_pos_neg_sample(q_proj, y_batch)
            if logits_con is not None:
                # 正样本标签固定为 0
                labels_con = torch.zeros(logits_con.size(0), dtype=torch.long, device=device)
                loss_con = criterion_cls(logits_con, labels_con)
                loss = loss_con * contrastive_rate + loss_cls * (1 - contrastive_rate)
            else:
                loss = loss_cls

            # 4.4 优化更新
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        progress_bar.set_postfix(loss=avg_loss)
    return moco_encoder
