from model.LRCN2D import LRCNAutoencoder, LRCNencoder, LRCNdecoder
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import umap
import src.retnet as tf
import model.Transformer as TF
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import umap
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def LRCCNdataload(proportion=0.9, device='cpu', batch_size=1024):
    #——————————————————————————数据集加载————————————————————————————————
    x_train = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_training.csv", low_memory=False)
    num_cols = x_train.shape[1]  # 获取列数
    # 提取出最后一列为y
    y_train = x_train.pop(x_train.columns[-1]).values

    x_test = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_edit_testing.csv", low_memory=False)
    num_cols = x_test.shape[1]  # 获取列数
    y_test = x_test.pop(x_test.columns[-1]).values

    # 将读取的数据转化为np格式方便后续训练
    x_train = np.array(x_train, dtype=np.float32)  # 将数据转换为float32类型
    y_train = np.array(y_train, dtype=np.int64)      # 将数据转换为int64类型
    x_test = np.array(x_test, dtype=np.float32)      # 将数据转换为float32类型
    y_test = np.array(y_test, dtype=np.int64)          # 将数据转换为int64类型

    X_train, _, Y_train, _ = train_test_split(x_train, y_train, train_size=proportion, random_state=42)
    x_train, x_train2, y_train, y_train2 = train_test_split(X_train, Y_train, train_size=0.9, random_state=42)

    # 定义未知类别和已知类别的映射
    unknown_classes = [0, 1, 8, 9]
    known_classes = sorted(set(range(10)) - set(unknown_classes))
    mapping_dict = {orig_label: new_label for new_label, orig_label in enumerate(known_classes)}

    # 对y_train2进行映射：
    # 如果属于已知类别，按mapping_dict映射；否则（未知类别）统一映射为类别6
    y_train2 = np.vectorize(lambda x: mapping_dict[x] if x in mapping_dict else 6)(y_train2)

    print(x_train.shape, x_train2.shape, x_test.shape)

    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    x_train2 = torch.from_numpy(x_train2).float().to(device)
    y_train2 = torch.from_numpy(y_train2).long().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 测试集中，默认将所有类别映射为6，再将已知类别映射为0～5
    y_test_mapped = torch.full_like(y_test, 6)
    for orig, new in mapping_dict.items():
        y_test_mapped[y_test == orig] = new

    # 数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    train_dataset2 = TensorDataset(x_train2, y_train2)
    test_dataset = TensorDataset(x_test, y_test_mapped)

    # 输出映射后的带标签数据集数据分布
    y_values = [label.item() for _, label in train_dataset2]
    y_distribution = pd.Series(y_values).value_counts()
    print("映射后的带标签数据集数据分布：")
    print(y_distribution)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_loader2, test_loader

train_loader, train_loader2, test_loader =LRCCNdataload(proportion=0.9,device='cpu',batch_size=1024)
# In[4]:

epochs = 100
#——————————配置调用设备————————————
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 数据放到gpu上还是cpu上
print("device",dev)
#——————————配置调用设备————————————

#model = LSTMClassifier(input_size=x_train.shape[1], hidden_size=hidden_dim, output_size=class_number).to(dev)
model = TF.TransformerClassifier(input_size=204, hidden_size=128, output_size=7).to(dev)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


# In[5]:


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

# 定义独立的 TransformerEncoder 模块
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=204, latent_dim=136, d_model=128, use_rff=False,
                 rff_output_dim=58, rff_gamma=1.0, transformer_layers=3, transformer_heads=2):
        super().__init__()
        self.use_rff = use_rff
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
        if self.use_rff:
            x = self.rff(x)
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
        if self.encoder.use_rff:
            x_enc = self.encoder.rff(x)
        else:
            x_enc = x
        memory = self.encoder.transformer_encoder(self.encoder.enc_input_adapter(x_enc).unsqueeze(1))
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
    loss_fn = nn.MSELoss()

    with tqdm(range(epochs), desc="Pretraining (Transformer)") as pbar:
        for epoch in pbar:
            total_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)  # [batch_size, input_dim]
                # 应用随机掩码
                x_masked = MaskProcessor.apply_random_mask(x_batch, mask_ratio)
                optimizer.zero_grad()
                reconstructed, latent = autoencoder(x_masked)
                loss = loss_fn(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    return autoencoder.encoder

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

def Evaluation(y_true, y_pred_total,y_hat_total=None,title="Unamed"):
    # 精确度、召回率、F1分数、AUC的计算
    acc = accuracy_score(y_true, y_pred_total)
    f1 = f1_score(y_true, y_pred_total, average='weighted', zero_division=1)
    precision_macro = precision_score(y_true, y_pred_total, average='macro', zero_division=1)
    recall_macro = recall_score(y_true, y_pred_total, average='macro', zero_division=1)
    precision_micro = precision_score(y_true, y_pred_total, average='micro', zero_division=1)
    recall_micro = recall_score(y_true, y_pred_total, average='micro', zero_division=1)

    # 打印结果
    print('Test Accuracy: {:.4f}'.format(acc),"\n")
    print('Test F1 Score: {:.4f}'.format(f1),"\n")
    print('Macro Precision: {:.4f}, Macro Recall: {:.4f}'.format(precision_macro, recall_macro),"\n")
    print('Micro Precision: {:.4f}, Micro Recall: {:.4f}'.format(precision_micro, recall_micro),"\n")


    # 二值化预测和真实标签
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6])
    y_pred_bin = label_binarize(y_pred_total, classes=[0, 1, 2, 3, 4, 5, 6])

    # 分类报告
    print(classification_report(y_true, y_pred_total, zero_division=1, digits=4),"\n")

    # 计算每个类别的AUC及总的AUC
    auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None)
    auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
    auc_micro = roc_auc_score(y_true_bin, y_pred_bin, average='micro')

    # 打印AUC结果
    print("AUC Scores by class:", auc_scores,"\n")
    print("Macro AUC: {:.4f}, Micro AUC: {:.4f}".format(auc_macro, auc_micro),"\n")

    # 添加AUPR计算
    precision_dict = {}
    recall_dict = {}
    aupr_scores = []

    for i, class_label in enumerate([0, 1, 2, 3, 4, 5, 6]):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        precision_dict[class_label] = precision
        recall_dict[class_label] = recall
        aupr = auc(recall, precision)
        aupr_scores.append(aupr)

    aupr_macro = np.mean(aupr_scores)
    print("AUPR Scores by class:", np.round(aupr_scores, 4),"\n")
    print("Macro AUPR: {:.4f}".format(aupr_macro) +"\n")

    # 添加FPR@TPR计算（以TPR=0.90为例）
    fpr_at_tpr = {}
    target_tpr = 0.90

    for i, class_label in enumerate([0, 1, 2, 3, 4, 5, 6]):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        # 找到最接近target_tpr的点
        idx = np.argmin(np.abs(tpr - target_tpr))
        fpr_at_tpr[class_label] = fpr[idx]

    print(f"FPR@TPR={target_tpr} by class:", {k: round(v, 4) for k, v in fpr_at_tpr.items()},"\n")
    print("Mean FPR@TPR=0.95: {:.4f}".format(np.mean(list(fpr_at_tpr.values()))),"\n")

    draw_confusion_matrix(y_true, y_pred_total, title=title + " 混淆矩阵" + f"{f1:.4f}")

    # UMAP 可视化
    if(y_hat_total is not None):
        drawplt(y_hat_total, y_true,title=title)

    return f1  # 返回 F1 分数

def sample_data(X, y, max_samples_per_class=1000):
    """
    从每个类别中最多提取 max_samples_per_class 个样本
    """
    # 确保 y 是 NumPy 数组
    y = np.array(y)

    unique_labels = np.unique(y)
    sampled_X, sampled_y = [], []
    for label in unique_labels:
        # 获取当前类别的索引
        indices = np.where(y == label)[0]
        if len(indices) > max_samples_per_class:
            indices = np.random.choice(indices, max_samples_per_class, replace=False)  # 随机采样
        sampled_X.append(X[indices])
        sampled_y.append(y[indices])
    # 合并采样后的数据
    sampled_X = np.vstack(sampled_X)
    sampled_y = np.hstack(sampled_y)
    return sampled_X, sampled_y

def drawplt(y_hat_total, y_true,title="Unamed"):
    # 采样
    y_pred_sampled, y_true_sampled = sample_data(y_hat_total, y_true, max_samples_per_class=200)
    print("Sampled data shape:", y_pred_sampled.shape, y_true_sampled.shape)
    # UMAP 降维
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=42)
    y_pred_umap = reducer.fit_transform(y_pred_sampled)

    # 确保类别与颜色映射一致
    unique_labels = np.unique(y_true_sampled)
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色映射
    label_to_color = {label: cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    # 绘制散点图
    colors = [label_to_color[label] for label in y_true_sampled]
    scatter = plt.scatter(
        y_pred_umap[:, 0],
        y_pred_umap[:, 1],
        c=colors,
        s=10,  # 点大小
        alpha=0.7
    )

    # 创建类别图示框
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Class {int(label)}",
                          markerfacecolor=color, markersize=8)
               for label, color in label_to_color.items()]
    plt.legend(handles=handles, title="Classes", loc='best')

    # 设置标题和标签
    plt.title(title)
    plt.tight_layout()
    save_dir = "..\\result\\"+title + str(time.time()) + ".png"
    plt.savefig(save_dir)
    plt.show()


def draw_confusion_matrix(y_true, y_pred, title="Confusion Matrix (Normalized %)"):
    """
    绘制归一化混淆矩阵的函数，以百分比的形式显示

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - title: 图表标题（默认："Confusion Matrix (Normalized %)"）
    """
    # 定义类别顺序，与评估函数中保持一致
    labels = [0, 1, 2, 3, 4, 5, 6]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 创建混淆矩阵显示对象，values_format设置为百分比格式
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # 绘制混淆矩阵图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

class Classifier2D(nn.Module):
    def __init__(self, encoder, in_features=136, hidden_size=64, out_features=7):
        """
        encoder: 新的 Encoder 模块，输出形状为 [B, 8, 3, 3]
        in_features: 展平后输入的维度 (8*3*3=72)
        """
        super(Classifier2D, self).__init__()
        self.encoder = encoder

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x):
        # 使用 encoder 获取特征图
        x = self.encoder(x)  # [B, 8, 3, 3]

        # 展平为 [B, 8*3*3]
        #x_reshaped = x.reshape(x.size(0), -1)
        y_hat = self.classifier(x)
        # 输入分类器
        return y_hat,x


#----------------------对比基准分类器-------------------------------------------------------
def baseline_classifier2D(encoder, train_loader, test_loader, hidden_size, epochs = 50, learn_rate = 0.001,device='cpu',title="Unamed"):
    # 动量编码器 (key encoder)
    encoder_test = copy.deepcopy(encoder).to(device)
    encoder_test.load_state_dict(encoder.state_dict())

    # 创建分类器并将其移动到目标设备
    classifier = Classifier2D(encoder_test,hidden_size=hidden_size).to(device)

    # 定义损失函数和优化器，仅优化 classifier.classifier 部分的参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.classifier.parameters(), lr=learn_rate)

    # 微调训练
    classifier.train()
    progress_bar = tqdm(range(epochs), desc="基准对比测试 Progress")

    for epoch in progress_bar:
        total_loss = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _ = classifier(x_batch.to(dev))
            loss = criterion(outputs, y_batch.to(dev))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # 在tqdm进度条中显示损失值
        progress_bar.set_postfix(loss=avg_loss)



    # 4. 模型测试
    y_true = []
    y_pred_total = []
    x_total = torch.tensor([]).cpu()
    z_total = torch.tensor([]).cpu()
    quantized_total = torch.tensor([]).cpu()
    recon_total = torch.tensor([]).cpu()
    y_total = torch.tensor([]).cpu()
    y_hat_total = torch.tensor([]).cpu()

    classifier.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_hat,z_hat = classifier(x_batch.to(dev))
            y_hat_total = torch.cat([y_hat_total.cpu(), y_hat.cpu()], dim=0)
            y_true += y_batch.tolist()
            y_pred = np.argmax(y_hat.cpu(), axis=1)
            y_pred_total += y_pred.tolist()
            x_total = torch.cat([x_total.cpu(), x_batch.cpu()], dim=0)
            quantized_total = torch.cat([quantized_total.cpu(), y_hat.cpu()], dim=0)
            y_total = torch.cat([y_total.cpu(), y_batch.cpu()], dim=0)
            z_total = torch.cat([z_total.cpu(), z_hat.cpu()], dim=0)

    #Evaluation(y_true, y_pred_total,y_hat_total=y_hat_total,title="baseline Classifier")
    #z_total = z_total.reshape(z_total.shape[0], -1)  # shape: [8, 400]
    #Evaluation(y_true, y_pred_total, y_hat_total=z_total,title="z_total")
    #不带作图
    Evaluation(y_true, y_pred_total, title=title+"z_total")
    #----------------------对比基准分类器结束-------------------------------------------------------

print("\n正在进行自监督预训练...")
time.sleep(0.1)

# 2. 模型初始化
# LRCN 模型（原有实现）
encoder = LRCNencoder(
    input_size=204,
    hidden_size=5,
    use_bdc=True,
    bdc_input_dim=16
).to(dev)

decoder = LRCNdecoder(
    output_size=204,
    hidden_size=5,
    use_bdc=True,
    bdc_output_dim=136
).to(dev)

autoencoder = LRCNAutoencoder(encoder, decoder).to(dev)

pretrain_encoder = pretrain_model(
    train_loader,
    (encoder, decoder, autoencoder),
    mask_ratio=0.2,
    lr=0.001,
    epochs=epochs,
    device=dev,
)

print("\n---------- 自监督预训练后的分类器效果 ---------")
time.sleep(0.1)
baseline_classifier2D(
    pretrain_encoder, train_loader2, test_loader,
    hidden_size=136,
    epochs=epochs,
    learn_rate=0.0001,
    device=dev,
    title="自监督预训练后的分类器"
)
