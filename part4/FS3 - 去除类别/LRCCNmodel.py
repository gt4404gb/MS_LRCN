"""
主程序文件：基于LR-CCN模型的训练和评估流程
包含：基础分类器、自监督预训练、对比学习以及最终的 LOF+KNN 分类器
"""
print("测试数据集使用比例从0.5变成1，带标签量从0.1增至0.2")
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
from LRCCNpretrain import pretrain_model,pretrain_simclr, pretrain_Transformermodel, pretrain_VAE, create_model  # 自监督预训练模块
from LRCCNbaseline import baseline_classifier
from LRCCNmoco import ContrastiveLearningKNN,MoCoEncoderKNN
from LRCCNFinalClassifier import LOFtest,KNNtest,LOF_KNN_test,ODINtest, ConfigurableOODtest,INFLOtest,GODINtest
import torch.nn as nn
from model.LRCN2D import LRCNAutoencoder, LRCNencoder, LRCNdecoder

# ------------------------- 全局配置 -------------------------
class Config:
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据参数
    DATA_PROPORTION = 0.5        # 训练数据使用比例
    BATCH_SIZE = 1024            # 批次大小
    CHANNEL_SIZE = 1             # 输入通道数

    # 自监督预训练参数
    SELF_SUPERVISED_METHOD = 'lrcn'  # 'lrcn' 或 'simclr'，或 'transformer'，或 'vae'，选择自监督方法
    PROJECTION_DIM = 128  # SimCLR投影头的输出维度

    INPUT_DIM = 204              # 输入特征维度
    HIDDEN_DIM = 5               # 隐藏层维度
    LATENT_DIM = 136             # TransformerAutoencoder 的隐藏层维度
    CLASS_NUM = 7                # 分类类别数
    USE_BDC = False               # 启用 BDC 模块
    BDC_INPUT_DIMENSION_REDUCTION = 16   # BDC 降维维度（需与 Encoder 输出匹配）
    BDC_OUTPUT_DIMENSION_REDUCTION = 136   # BDC 降维维度（需与 Encoder 输出匹配）
    DIMENSION_REDUCTION = 16     # 特征降维维度

    # 训练参数
    BASE_LR = 0.0001             # 分类器训练学习率
    PRETRAIN_LR = 0.0001          # 自监督预训练学习率
    PRETRAIN_EPOCHS = 100          # 自监督预训练轮次
    CONTRASTIVE_EPOCHS = 100       # 对比学习轮次

    # 自监督预训练参数
    PRETRAIN_MASK_RATIO = 0.1    # 输入掩码比例
    PRETRAIN_WEIGHT_DECAY = 1e-4 # 权重衰减参数

    # Transformer 参数
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 2

    # 对比学习参数
    FEATURE_DIM = BDC_OUTPUT_DIMENSION_REDUCTION  # 对比学习中使用的特征维度（例如 BDC 输出维度）
    QUEUE_SIZE = 100         # 每个类别的队列长度
    MOMENTUM = 0.999         # 动量更新参数 m
    TEMPERATURE = 0.3        # 温度参数 T
    TOP_K = 70               # 负样本选择上限
    CONTRASTIVE_RATE = 0.4   # 对比损失与分类损失的权重比例
    CONTRASTIVE_LR = 1e-4    # 对比学习优化器学习率

    # LOF+KNN 参数
    LOF_K = 5                # LOFOODClassifier 中 KNN 的近邻数
    LOF_KNN_P = 2            # LOFOODClassifier 中 KNN 的距离度量参数（默认欧氏距离）
    LOF_N_NEIGHBORS = 15     # LOFOODClassifier 中 LOF 的近邻数
    LOF_CONTAMINATION = 0.5 # LOFOODClassifier 中 LOF 的异常点比例
    LOF_THRESHOLD = 0.5      # LOFOODClassifier 中 LOF 的 decision_function 阈值

    # 实验开关
    ENABLE_PRETRAIN = True     # 是否启用自监督预训练
    ENABLE_CONTRASTIVE = True  # 是否启用对比学习

    # 预训练模型路径
    TRAINED_MODEL_PATH = "contrastive_encoder" + str(time.time())+".pth"

# ------------------------- 固定随机种子 -------------------------
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


# ------------------------- 动态生成日志文件名 -------------------------
# 例如：log_dp0.5_bs1024_20250304_153000.log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"..\\log\\log_mask{Config.PRETRAIN_MASK_RATIO}_qs{Config.QUEUE_SIZE}_mo{Config.MOMENTUM}_t{Config.TEMPERATURE}_k{Config.TOP_K}_lofn{Config.LOF_N_NEIGHBORS}_{timestamp}.log"

# 配置 logging，将日志写入文件和控制台
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

# ------------------------- LoggerWriter 类 -------------------------
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

# 重定向 sys.stdout 到 logging.info，这样所有 print 输出都会写入日志文件
sys.stdout = LoggerWriter(logging.info)

# 只提取非内置属性（不以双下划线开头）的配置参数
config_dict = {k: v for k, v in vars(Config).items() if not k.startswith("__")}
formatted_config = pprint.pformat(config_dict, indent=4)

logging.info("当前日志文件名： %s", log_filename)
logging.info("所有配置参数：\n%s", formatted_config)


# ------------------------- 数据加载 -------------------------
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


# ------------------------- 主训练流程 -------------------------
def train(train_loader, train_loader2):

    # 2. 模型初始化
    encoder, decoder, autoencoder = create_model(Config)

    # 3. 基准分类器测试（使用原始 encoder）
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

    # 4. 自监督预训练阶段
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
                device=Config.DEVICE,
            )
        elif Config.SELF_SUPERVISED_METHOD == 'vae':
            pretrain_encoder = pretrain_VAE(
                autoencoder=autoencoder,
                mask_ratio=Config.PRETRAIN_MASK_RATIO,
                train_loader=train_loader,
                lr=Config.PRETRAIN_LR,
                epochs=Config.PRETRAIN_EPOCHS,
                device=Config.DEVICE,
            )
        else:
            raise ValueError(f"Unknown SELF_SUPERVISED_METHOD: {Config.SELF_SUPERVISED_METHOD}")

        print("\n---------- 自监督预训练后的分类器效果 ---------")
        time.sleep(0.1)
        baseline_classifier(
            pretrain_encoder, train_loader2, test_loader,
            in_features = Config.BDC_OUTPUT_DIMENSION_REDUCTION,
            hidden_size=Config.HIDDEN_DIM,
            epochs=Config.PRETRAIN_EPOCHS,
            learn_rate=Config.BASE_LR,
            device=Config.DEVICE,
            title="自监督预训练后的分类器"
        )
    else:
        pretrain_encoder = encoder

    # 5. 对比学习阶段
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

        print("\n---------- 对比学习后的分类器效果 ---------")
        time.sleep(0.1)
        baseline_classifier(
            pretrain_encoder, train_loader2, test_loader,
            in_features = Config.BDC_OUTPUT_DIMENSION_REDUCTION,
            hidden_size=Config.HIDDEN_DIM,
            epochs=Config.PRETRAIN_EPOCHS,
            learn_rate=Config.BASE_LR,
            device=Config.DEVICE,
            title="自监督预训练后的分类器"
        )
    else:
        contrastive_encoder = pretrain_encoder

    # 保存训练好的模型
    torch.save(contrastive_encoder.state_dict(), Config.TRAINED_MODEL_PATH)
    print(f"模型已保存至 {Config.TRAINED_MODEL_PATH}")


# ------------------------- 仅进行LOF+KNN测试 -------------------------
def evaluate_lof_knn(train_loader2, test_loader,model_path = Config.TRAINED_MODEL_PATH):
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
    encoder.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

    lof_params = {
        'k': Config.LOF_KNN_P,
        'knn_p': Config.LOF_KNN_P,
        'n_neighbors': Config.LOF_N_NEIGHBORS,
        'contamination': Config.LOF_CONTAMINATION,
        'threshold' : Config.LOF_THRESHOLD,
    }

    # # 6. 最终分类测试（LOF）
    # print("\n---------- LOF 分类器测试 ---------")
    # time.sleep(0.1)
    # f1_score =LOFtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     lof_params=lof_params
    # )
    #
    # # 构建包含 F1 分数的模型文件名
    # f1_str = f"{f1_score:.4f}".replace('.', '_')  # 将小数点替换为下划线，避免文件名问题
    # model_path_with_f1 = f"contrastive_encoder_f1_{f1_str}.pth"
    #
    # # 额外保存一个不包含 LOF 的模型
    # torch.save(encoder.state_dict(), model_path_with_f1)
    # print(f"模型已额外保存至 {model_path_with_f1}，F1 分数: {f1_score:.4f}")
    #
    #
    # print("\n---------- IForest 分类器测试 ---------")
    # time.sleep(0.1)
    # ConfigurableOODtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     outlier_detector_str="OCSVM",
    #     contamination=Config.LOF_CONTAMINATION
    # )

    # # 6. 最终对比实验（KNN）
    # print("\n---------- KNN 对比分类器测试 ---------")
    # time.sleep(0.1)
    # # 构建 LOF 配置参数字典
    # knn_params = {
    #     'k': Config.LOF_K,
    #     'knn_p': Config.LOF_KNN_P,
    # }
    # KNNtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    # )
    #
    # print("\n---------- LOF+ KNN 分类器测试 ---------")
    # time.sleep(0.1)
    # # 构建 LOF 配置参数字典
    # lof_params = {
    #     'k': Config.LOF_K,
    #     'knn_p': Config.LOF_KNN_P,
    #     'n_neighbors': Config.LOF_N_NEIGHBORS,
    #     'contamination': Config.LOF_CONTAMINATION
    # }
    # LOF_KNN_test(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     lof_params=lof_params,
    #     threshold=Config.LOF_THRESHOLD
    # )

    print("\n---------- GODIN 分类器测试 ---------")
    time.sleep(0.1)
    GODINtest(
        encoder=encoder,
        test_loader=test_loader,
        device=Config.DEVICE,
        threshold=0.1
    )

    print("\n---------- ODIN 分类器测试 ---------")
    time.sleep(0.1)
    ODINtest(
        encoder=encoder,
        test_loader=test_loader,
        device=Config.DEVICE,
        temperature=50,
        epsilon=0.0014,
        threshold=0.154
    )




    # print("\n---------- INFLO 分类器测试 ---------")
    # time.sleep(0.1)
    # f1_score =INFLOtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     inflo_params=lof_params
    # )


if __name__ == "__main__":
    # 1. 数据加载
    train_loader, train_loader2, test_loader = load_data()

    #train(train_loader, train_loader2)
    evaluate_lof_knn(train_loader2, test_loader,model_path="contrastive_encoder_f1_0_8884.pth")
