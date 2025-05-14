
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

def Evaluation(y_true, y_pred_total,y_hat_total=None,title="Unamed"):
    # 精确度、召回率、F1分数、AUC的计算
    classes = [0, 1, 2, 3, 4, 5, -1]
    f1 = f1_score(y_true, y_pred_total, average='weighted', zero_division=1)


    # 二值化预测和真实标签
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred_total, classes=classes)

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

    for i, class_label in enumerate(classes):
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

    for i, class_label in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        # 找到最接近target_tpr的点
        idx = np.argmin(np.abs(tpr - target_tpr))
        fpr_at_tpr[class_label] = fpr[idx]

    print(f"FPR@TPR={target_tpr} by class:", {k: round(v, 4) for k, v in fpr_at_tpr.items()},"\n")
    print("Mean FPR@TPR=0.95: {:.4f}".format(np.mean(list(fpr_at_tpr.values()))),"\n")

    draw_confusion_matrix(y_true, y_pred_total, title=title + " 混淆矩阵" + f"{f1:.4f}",classes=classes)

    # UMAP 可视化
    if(y_hat_total is not None):
        drawplt(y_true, y_hat_total, title=title)

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

def drawplt(y_true, y_hat_total,title="Unamed"):
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

def draw_confusion_matrix(y_true, y_pred, title="Confusion Matrix (Normalized %)",classes=[0, 1, 2, 3, 4, 5, -1]):
    """
    绘制归一化混淆矩阵的函数，以百分比的形式显示

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - title: 图表标题（默认："Confusion Matrix (Normalized %)"）
    """
    # 定义类别顺序，与评估函数中保持一致
    labels = classes

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

class Classifier(nn.Module):
    def __init__(self, encoder, in_features=136, hidden_size=64, out_features=9):
        """
        encoder: 新的 Encoder 模块，输出形状为 [B, 8, 3, 3]
        in_features: 展平后输入的维度 (8*3*3=72)
        """
        super(Classifier, self).__init__()
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
def baseline_classifier(encoder, train_loader, test_loader, in_features=136, hidden_size=64, epochs = 50, learn_rate = 0.001, device='cpu', title="Unamed"):
    # 动量编码器 (key encoder)
    encoder_test = copy.deepcopy(encoder).to(device)
    encoder_test.load_state_dict(encoder.state_dict())

    # 创建分类器并将其移动到目标设备
    classifier = Classifier(encoder_test,in_features=in_features, hidden_size=hidden_size).to(device)

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
            outputs, _ = classifier(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # 在tqdm进度条中显示损失值
        progress_bar.set_postfix(loss=avg_loss)

        end_time = time.time()


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
            y_hat,z_hat = classifier(x_batch)
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