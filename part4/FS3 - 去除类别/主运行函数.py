#[1]
#!/usr/bin/env python
# coding: utf-8
import sys
import os

# 假设上级目录是'..'
sys.path.append(os.path.abspath('..'))

print("")

import torch
import time
import random
import numpy as np
import logging
import sys
from datetime import datetime
import pprint

# 导入各模块（请根据实际项目目录调整导入路径）
from LRCCNdataloader import LRCCNdataload
from LRCCNpretrain import pretrain_model, pretrain_simclr, pretrain_Transformermodel, pretrain_VAE, create_model
from LRCCNbaseline import baseline_classifier
from LRCCNmoco import ContrastiveLearningKNN, MoCoEncoderKNN
from LRCCNFinalClassifier import LOFtest, KNNtest, LOF_KNN_test, ODINtest, ConfigurableOODtest
import torch.nn as nn
from model.LRCN2D import LRCNAutoencoder, LRCNencoder, LRCNdecoder

# 固定随机种子
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

# 定义基础配置类
class Config:
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据参数
    DATA_PROPORTION = 0.5        # 训练数据使用比例
    BATCH_SIZE = 2048            # 批次大小
    CHANNEL_SIZE = 1             # 输入通道数

    # 自监督预训练参数
    SELF_SUPERVISED_METHOD = 'lrcn'  # 'lrcn' 或 'simclr'，或 'transformer'，或 'vae'
    PROJECTION_DIM = 128  # SimCLR投影头的输出维度

    INPUT_DIM = 204              # 输入特征维度
    HIDDEN_DIM = 5               # 隐藏层维度
    LATENT_DIM = 136             # TransformerAutoencoder 的隐藏层维度
    CLASS_NUM = 7                # 分类类别数
    USE_BDC = True               # 启用 BDC 模块
    BDC_INPUT_DIMENSION_REDUCTION = 16   # BDC 降维维度（需与 Encoder 输出匹配）
    BDC_OUTPUT_DIMENSION_REDUCTION = 136   # BDC 降维维度（需与 Encoder 输出匹配）
    DIMENSION_REDUCTION = 16     # 特征降维维度

    # 训练参数
    BASE_LR = 0.0001             # 分类器训练学习率
    PRETRAIN_LR = 0.0001         # 自监督预训练学习率
    PRETRAIN_EPOCHS = 100        # 自监督预训练轮次
    CONTRASTIVE_EPOCHS = 100     # 对比学习轮次

    # 自监督预训练参数
    PRETRAIN_MASK_RATIO = 0.2    # 输入掩码比例
    PRETRAIN_WEIGHT_DECAY = 1e-4 # 权重衰减参数

    # Transformer 参数
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 2

    # 对比学习参数
    FEATURE_DIM = BDC_OUTPUT_DIMENSION_REDUCTION  # 对比学习中使用的特征维度
    QUEUE_SIZE = 100         # 每个类别的队列长度
    MOMENTUM = 0.99          # 动量更新参数 m
    TEMPERATURE = 0.1        # 温度参数 T
    TOP_K = 70               # 负样本选择上限
    CONTRASTIVE_RATE = 0.4   # 对比损失与分类损失的权重比例
    CONTRASTIVE_LR = 1e-4    # 对比学习优化器学习率

    # LOF+KNN 参数
    LOF_K = 5                # LOFOODClassifier 中 KNN 的近邻数
    LOF_KNN_P = 2            # LOFOODClassifier 中 KNN 的距离度量参数（默认欧氏距离）
    LOF_N_NEIGHBORS = 15     # LOFOODClassifier 中 LOF 的近邻数
    LOF_CONTAMINATION = 0.5  # LOFOODClassifier 中 LOF 的异常点比例
    LOF_THRESHOLD = 0.5      # LOFOODClassifier 中 LOF 的 decision_function 阈值

    # 实验开关
    ENABLE_PRETRAIN = True     # 是否启用自监督预训练
    ENABLE_CONTRASTIVE = True  # 是否启用对比学习

    # 预训练模型路径
    TRAINED_MODEL_PATH = "contrastive_encoder" + str(time.time()) + ".pth"

# 日志配置
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"../log/log_mask{Config.PRETRAIN_MASK_RATIO}_qs{Config.QUEUE_SIZE}_mo{Config.MOMENTUM}_t{Config.TEMPERATURE}_k{Config.TOP_K}_lofn{Config.LOF_N_NEIGHBORS}_{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger('').addHandler(console)

# 重定向 print 到日志
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self._buffer = ''
    def write(self, message):
        if message != '\n':
            self._buffer += message
            if '\n' in self._buffer:
                for line in self._buffer.splitlines():
                    self.level(line)
                self._buffer = ''
    def flush(self):
        if self._buffer:
            self.level(self._buffer)
            self._buffer = ''
sys.stdout = LoggerWriter(logging.info)

# 输出初始配置
config_dict = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
logging.info("当前日志文件名：%s", log_filename)
logging.info("初始配置参数：\n%s", pprint.pformat(config_dict, indent=4))


#[2]
# 数据相关配置（可根据需要修改）
Config.DATA_PROPORTION = 0.5  # 测试数据集使用比例从0.5改为1
Config.BATCH_SIZE = 2048

# 数据加载函数
def load_data():
    """
    加载训练和测试数据集
    返回：train_loader, train_loader2, test_loader
    """
    return LRCCNdataload(
        proportion=Config.DATA_PROPORTION,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE
    )

# 执行数据加载
train_loader, train_loader2, test_loader = load_data()
print(f"数据加载完成，使用比例: {Config.DATA_PROPORTION}, 批次大小: {Config.BATCH_SIZE}")

#[3]
# 模型相关配置（可根据需要修改）
Config.INPUT_DIM = 204
Config.HIDDEN_DIM = 5
Config.LATENT_DIM = 136

# 初始化模型
encoder, decoder, autoencoder = create_model(Config)
print("模型初始化完成")


#[4]
# 基准分类器配置（可根据需要修改）
Config.BDC_OUTPUT_DIMENSION_REDUCTION = 136
Config.PRETRAIN_EPOCHS = 1
Config.BASE_LR = 0.0001

# 测试原始分类器
print("\n---------- 原始分类器效果 ---------")
time.sleep(0.1)
baseline_classifier(
    encoder, train_loader2, test_loader,
    hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION,
    epochs=Config.PRETRAIN_EPOCHS,
    learn_rate=Config.BASE_LR,
    device=Config.DEVICE,
    title="原始分类器"
)


#[5]
# 自监督预训练配置（可根据需要修改）
Config.ENABLE_PRETRAIN = True
Config.SELF_SUPERVISED_METHOD = 'lrcn'  # 可选 'lrcn', 'simclr', 'transformer', 'vae'
Config.PRETRAIN_LR = 0.0001
Config.PRETRAIN_EPOCHS = 1
Config.PRETRAIN_MASK_RATIO = 0.2
Config.PRETRAIN_WEIGHT_DECAY = 1e-4
Config.PROJECTION_DIM = 128

# 自监督预训练
if Config.ENABLE_PRETRAIN:
    print("\n正在进行自监督预训练...")
    time.sleep(0.1)
    if Config.SELF_SUPERVISED_METHOD == 'lrcn':
        pretrain_encoder = pretrain_model(
            train_loader,
            (encoder, decoder, autoencoder),
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE,
            mask_ratio=Config.PRETRAIN_MASK_RATIO,
            weight_decay=Config.PRETRAIN_WEIGHT_DECAY
        )
    elif Config.SELF_SUPERVISED_METHOD == 'simclr':
        pretrain_encoder = pretrain_simclr(
            train_loader=train_loader,
            encoder=encoder,
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE,
            projection_dim=Config.PROJECTION_DIM
        )
    elif Config.SELF_SUPERVISED_METHOD == 'transformer':
        pretrain_encoder = pretrain_Transformermodel(
            autoencoder=autoencoder,
            mask_ratio=Config.PRETRAIN_MASK_RATIO,
            train_loader=train_loader,
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE
        )
    elif Config.SELF_SUPERVISED_METHOD == 'vae':
        pretrain_encoder = pretrain_VAE(
            autoencoder=autoencoder,
            mask_ratio=Config.PRETRAIN_MASK_RATIO,
            train_loader=train_loader,
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE
        )
    else:
        raise ValueError(f"未知的自监督方法: {Config.SELF_SUPERVISED_METHOD}")

    # 测试预训练后的分类器
    print("\n---------- 自监督预训练后的分类器效果 ---------")
    time.sleep(0.1)
    baseline_classifier(
        pretrain_encoder, train_loader2, test_loader,
        in_features=Config.BDC_OUTPUT_DIMENSION_REDUCTION,
        hidden_size=Config.HIDDEN_DIM,
        epochs=Config.PRETRAIN_EPOCHS,
        learn_rate=Config.BASE_LR,
        device=Config.DEVICE,
        title="自监督预训练后的分类器"
    )

    # 保存模型
    torch.save(pretrain_encoder.state_dict(), "pretrain_encoder.pth")
    print("预训练模型已保存至 pretrain_encoder.pth")
else:
    pretrain_encoder = encoder


#[6]
# 对比学习配置（可根据需要修改）
Config.ENABLE_CONTRASTIVE = True
Config.CONTRASTIVE_EPOCHS = 1
Config.CONTRASTIVE_RATE = 0.4
Config.FEATURE_DIM = Config.BDC_OUTPUT_DIMENSION_REDUCTION
Config.TOP_K = 70
Config.QUEUE_SIZE = 100
Config.CONTRASTIVE_LR = 1e-4
Config.MOMENTUM = 0.99
Config.TEMPERATURE = 0.1

# 对比学习
if Config.ENABLE_CONTRASTIVE:
    print("\n正在进行对比学习...")
    time.sleep(0.1)
    contrastive_encoder = ContrastiveLearningKNN(
        encoder=pretrain_encoder,
        train_loader=train_loader2,
        epochs=Config.CONTRASTIVE_EPOCHS,
        device=Config.DEVICE,
        class_number=Config.CLASS_NUM,
        contrastive_rate=Config.CONTRASTIVE_RATE,
        feature_dim=Config.FEATURE_DIM,
        top_k=Config.TOP_K,
        queue_size=Config.QUEUE_SIZE,
        lr=Config.CONTRASTIVE_LR,
        m=Config.MOMENTUM,
        T=Config.TEMPERATURE
    )

    # 测试对比学习后的分类器
    print("\n---------- 对比学习后的分类器效果 ---------")
    time.sleep(0.1)
    baseline_classifier(
        contrastive_encoder.encoder_q, train_loader2, test_loader,
        in_features=Config.BDC_OUTPUT_DIMENSION_REDUCTION,
        hidden_size=Config.HIDDEN_DIM,
        epochs=Config.PRETRAIN_EPOCHS,
        learn_rate=Config.BASE_LR,
        device=Config.DEVICE,
        title="对比学习后的分类器"
    )

    # 保存模型
    torch.save(contrastive_encoder.state_dict(), Config.TRAINED_MODEL_PATH)
    print(f"对比学习模型已保存至 {Config.TRAINED_MODEL_PATH}")
else:
    contrastive_encoder = pretrain_encoder


#[7]
# 最终分类配置（可根据需要修改）
Config.LOF_K = 5
Config.LOF_KNN_P = 2
Config.LOF_N_NEIGHBORS = 15
Config.LOF_CONTAMINATION = 0.5
Config.LOF_THRESHOLD = 0.5

# LOF 测试
print("\n---------- LOF 分类器测试 ---------")
time.sleep(0.1)
lof_params = {
    'k': Config.LOF_K,
    'knn_p': Config.LOF_KNN_P,
    'n_neighbors': Config.LOF_N_NEIGHBORS,
    'contamination': Config.LOF_CONTAMINATION
}
LOFtest(
    contrastive_encoder, train_loader2, test_loader,
    class_number=Config.CLASS_NUM,
    device=Config.DEVICE,
    lof_params=lof_params
)


# IForest 测试
print("\n---------- IForest 分类器测试 ---------")
time.sleep(0.1)
ConfigurableOODtest(
    contrastive_encoder, train_loader2, test_loader,
    class_number=Config.CLASS_NUM,
    device=Config.DEVICE,
    outlier_detector_str="IForest",
    contamination=Config.LOF_CONTAMINATION
)

# ODIN 测试
print("\n---------- ODIN 分类器测试 ---------")
time.sleep(0.1)
f1_score = ODINtest(
    encoder=contrastive_encoder,
    test_loader=test_loader,
    device=Config.DEVICE,
    temperature=50,
    epsilon=0.0014,
    threshold=0.152
)

# 保存带有 F1 分数的模型
f1_str = f"{f1_score:.4f}".replace('.', '_')
model_path_with_f1 = f"contrastive_encoder_f1_{f1_str}.pth"
torch.save(contrastive_encoder.state_dict(), model_path_with_f1)
print(f"模型已保存至 {model_path_with_f1}，F1 分数: {f1_score:.4f}")


#[8]
# 加载已保存的模型
model_path = "contrastive_encoder_f1_0_8765.pth"  # 可修改为其他路径
encoder = MoCoEncoderKNN(
    base_encoder=LRCNencoder(input_size=Config.INPUT_DIM, hidden_size=Config.HIDDEN_DIM),
    feature_dim=Config.FEATURE_DIM,
    num_classes=Config.CLASS_NUM,
    K=Config.QUEUE_SIZE,
    m=Config.MOMENTUM,
    T=Config.TEMPERATURE,
    top_k=Config.TOP_K,
    device=Config.DEVICE
).to(Config.DEVICE)
encoder.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

# LOF+KNN 测试
print("\n---------- LOF+ KNN 分类器测试 ---------")
time.sleep(0.1)
lof_params = {
    'k': Config.LOF_K,
    'knn_p': Config.LOF_KNN_P,
    'n_neighbors': Config.LOF_N_NEIGHBORS,
    'contamination': Config.LOF_CONTAMINATION
}
LOF_KNN_test(
    encoder, train_loader2, test_loader,
    class_number=Config.CLASS_NUM,
    device=Config.DEVICE,
    lof_params=lof_params,
    threshold=Config.LOF_THRESHOLD
)