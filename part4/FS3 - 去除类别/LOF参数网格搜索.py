import torch
import time
import random
import numpy as np
import logging
import sys
from datetime import datetime
import pprint
from itertools import product

from LRCCNdataloader import LRCCNdataload
from LRCCNpretrain import pretrain_model
from LRCCNbaseline import baseline_classifier
from LRCCNmoco import ContrastiveLearningKNN,MoCoEncoderKNN
from LRCCNFinalClassifier import LOFtest, LOF_KNN_test
from model.LRCN2D import LRCNencoder, LRCNdecoder, LRCNAutoencoder


# ------------------------- 全局配置 -------------------------
class Config:
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据参数
    DATA_PROPORTION = 0.5        # 训练数据使用比例
    BATCH_SIZE = 1024            # 批次大小
    CHANNEL_SIZE = 1             # 输入通道数

    # 模型参数
    INPUT_DIM = 204              # 输入特征维度
    HIDDEN_DIM = 5               # 隐藏层维度
    CLASS_NUM = 9                # 分类类别数
    USE_BDC = True               # 启用 BDC 模块
    BDC_INPUT_DIMENSION_REDUCTION = 16   # BDC 降维维度（需与 Encoder 输出匹配）
    BDC_OUTPUT_DIMENSION_REDUCTION = 136   # BDC 降维维度（需与 Encoder 输出匹配）
    DIMENSION_REDUCTION = 16     # 特征降维维度
    MODEL_TYPE = '2D'            # 模型架构类型 ('1D' 或 '2D')
    USE_RFF = False              # 是否使用随机傅里叶特征

    # 训练参数
    BASE_LR = 0.0001             # 分类器训练学习率
    PRETRAIN_LR = 0.0001          # 自监督预训练学习率
    PRETRAIN_EPOCHS = 100          # 自监督预训练轮次
    CONTRASTIVE_EPOCHS = 100       # 对比学习轮次

    # 自监督预训练参数
    PRETRAIN_MASK_RATIO = 0.1    # 输入掩码比例
    PRETRAIN_WEIGHT_DECAY = 1e-4 # 权重衰减参数

    # Transformer / RFF 参数（仅在 1D 模型或使用 RFF 时使用）
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 2
    RFF_OUTPUT_DIM = 58
    RFF_GAMMA = 1.0

    # 对比学习参数
    FEATURE_DIM = BDC_OUTPUT_DIMENSION_REDUCTION  # 对比学习中使用的特征维度（例如 BDC 输出维度）
    QUEUE_SIZE = 100         # 每个类别的队列长度
    MOMENTUM = 0.99         # 动量更新参数 m
    TEMPERATURE = 0.2        # 温度参数 T
    TOP_K = 50               # 负样本选择上限
    CONTRASTIVE_RATE = 0.4   # 对比损失与分类损失的权重比例
    CONTRASTIVE_LR = 1e-4    # 对比学习优化器学习率

    # LOF+KNN 参数
    LOF_K = 5                # LOFOODClassifier 中 KNN 的近邻数
    LOF_KNN_P = 2            # LOFOODClassifier 中 KNN 的距离度量参数（默认欧氏距离）
    LOF_N_NEIGHBORS = 15
    LOF_CONTAMINATION = 0.05 # LOFOODClassifier 中 LOF 的异常点比例
    LOF_THRESHOLD = -0.5      # LOFOODClassifier 中 LOF 的 decision_function 阈值

    # 实验开关
    ENABLE_PRETRAIN = True     # 是否启用自监督预训练
    ENABLE_CONTRASTIVE = True  # 是否启用对比学习

    # 预训练模型路径
    TRAINED_MODEL_PATH = "contrastive_encoder 0.88F1得分.pth"


# ------------------------- 网格搜索参数范围 -------------------------
GRID_SEARCH_PARAMS = {
    'LOF_CONTAMINATION': [ 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
    'LOF_N_NEIGHBORS': [5, 10, 15, 20, 25, 30]
}


# ------------------------- 日志配置 -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"..\\log\\grid_search_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


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

config_dict = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
logging.info("当前日志文件名： %s", log_filename)
logging.info("基础配置参数：\n%s", pprint.pformat(config_dict, indent=4))
logging.info("网格搜索参数范围：\n%s", pprint.pformat(GRID_SEARCH_PARAMS, indent=4))


# ------------------------- 数据加载 -------------------------
def load_data():
    return LRCCNdataload(
        proportion=Config.DATA_PROPORTION,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE
    )


# ------------------------- 训练模型（仅需一次） -------------------------
def train_and_save_model(train_loader, train_loader2, test_loader):
    encoder = LRCNencoder(input_size=Config.INPUT_DIM, hidden_size=Config.HIDDEN_DIM).to(Config.DEVICE)
    decoder = LRCNdecoder(output_size=Config.INPUT_DIM, hidden_size=Config.HIDDEN_DIM).to(Config.DEVICE)
    autoencoder = LRCNAutoencoder(encoder, decoder).to(Config.DEVICE)

    # 自监督预训练
    if Config.ENABLE_PRETRAIN:
        print("\n正在进行自监督预训练...")
        pretrain_encoder = pretrain_model(
            train_loader,
            (encoder, decoder, autoencoder),
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE,
            mask_ratio=Config.PRETRAIN_MASK_RATIO,
            weight_decay=Config.PRETRAIN_WEIGHT_DECAY
        )
    else:
        pretrain_encoder = encoder

    # 对比学习
    if Config.ENABLE_CONTRASTIVE:
        contrastive_encoder = ContrastiveLearningKNN(
            encoder=pretrain_encoder,
            train_loader=train_loader2,
            epochs=Config.CONTRASTIVE_EPOCHS,
            device=Config.DEVICE,
            class_number=Config.CLASS_NUM,
            contrastive_rate=Config.CONTRASTIVE_RATE,
            feature_dim=Config.FEATURE_DIM,
            top_k=50,
            queue_size=Config.QUEUE_SIZE,
            lr=Config.CONTRASTIVE_LR,
            m=Config.MOMENTUM,
            T=Config.TEMPERATURE
        )
    else:
        contrastive_encoder = pretrain_encoder

    # 保存训练好的模型
    torch.save(contrastive_encoder.state_dict(), Config.TRAINED_MODEL_PATH)
    print(f"模型已保存至 {Config.TRAINED_MODEL_PATH}")


# ------------------------- 仅进行LOF+KNN测试 -------------------------
def evaluate_lof_knn( train_loader2, test_loader, params):
    # 读取训练好的模型
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
    encoder.load_state_dict(torch.load(Config.TRAINED_MODEL_PATH, map_location=Config.DEVICE))
    encoder.eval()

    LOF_CONTAMINATION = params['LOF_CONTAMINATION']
    LOF_N_NEIGHBORS = params['LOF_N_NEIGHBORS']

    logging.info(f"\n测试参数: LOF_CONTAMINATION={LOF_CONTAMINATION}, LOF_N_NEIGHBORS={LOF_N_NEIGHBORS}")

    lof_params = {
        'k': Config.LOF_KNN_P,
        'knn_p': Config.LOF_KNN_P,
        'n_neighbors': LOF_N_NEIGHBORS,
        'contamination': LOF_CONTAMINATION
    }

    # 运行LOF测试
    LOFtest(
        encoder, train_loader2, test_loader,
        class_number=Config.CLASS_NUM,
        device=Config.DEVICE,
        lof_params=lof_params
    )



# ------------------------- 主函数 -------------------------
def main():
    train_loader, train_loader2, test_loader = load_data()

    # 仅训练一次
    #train_and_save_model(train_loader, train_loader2, test_loader)

    # 生成所有参数组合
    param_combinations = [dict(zip(GRID_SEARCH_PARAMS.keys(), values)) for values in product(*GRID_SEARCH_PARAMS.values())]
    logging.info(f"总共需要测试 {len(param_combinations)} 组参数组合")

    for i, params in enumerate(param_combinations, 1):
        logging.info(f"\n=== 网格搜索进度: {i}/{len(param_combinations)} ===")
        evaluate_lof_knn(train_loader2, test_loader, params)

    logging.info("\n网格搜索完成！请查看日志文件以比较不同参数组合的性能")


if __name__ == "__main__":
    main()
