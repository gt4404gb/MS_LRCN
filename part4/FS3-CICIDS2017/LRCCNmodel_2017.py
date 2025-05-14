"""
主程序文件：基于LR-CCN模型的训练和评估流程
包含：基础分类器、自监督预训练、对比学习以及最终的 LOF+KNN 分类器
"""

import torch
import time
import random
import numpy as np
import logging
import sys
from datetime import datetime
import pprint

# 导入各模块（请根据实际项目目录调整导入路径）
from LRCCNdataloader_2017 import LRCCNdataload
from LRCCNpretrain_2017 import pretrain_model,pretrain_simclr, pretrain_Transformermodel, TransformerAutoencoder,TransformerEncoder,TransformerDecoder  # 自监督预训练模块
from LRCCNbaseline_2017 import baseline_classifier
from LRCCNmoco_2017 import ContrastiveLearningKNN,MoCoEncoderKNN
from LRCCNFinalClassifier_2017 import LOFtest,KNNtest,LOF_KNN_test,ODINtest, ConfigurableOODtest
from deepBDC import BDC  # 如有需要
import torch.nn as nn
from model.LRCN2D_CICIDS2017 import LRCNAutoencoder, LRCNencoder, LRCNdecoder

# ------------------------- 全局配置 -------------------------
class Config:
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据参数
    DATA_PROPORTION = 0.5        # 训练数据使用比例
    BATCH_SIZE = 1024            # 批次大小
    CHANNEL_SIZE = 1             # 输入通道数

    # 自监督预训练参数
    SELF_SUPERVISED_METHOD = 'lrcn'  # 'lrcn' 或 'simclr'，或 'transformer'，选择自监督方法
    PROJECTION_DIM = 128  # SimCLR投影头的输出维度

    INPUT_DIM = 78              # 输入特征维度
    HIDDEN_DIM = 3               # 隐藏层维度
    LATENT_DIM = 136             # TransformerAutoencoder 的隐藏层维度
    CLASS_NUM = 5                # 分类类别数
    USE_BDC = True               # 启用 BDC 模块
    BDC_INPUT_DIMENSION_REDUCTION = 16   # BDC 降维维度（需与 Encoder 输出匹配）
    BDC_OUTPUT_DIMENSION_REDUCTION = 136   # BDC 降维维度（需与 Encoder 输出匹配）
    DIMENSION_REDUCTION = 16     # 特征降维维度
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
    LOF_N_NEIGHBORS = 15     # LOFOODClassifier 中 LOF 的近邻数
    LOF_CONTAMINATION = 0.001 # LOFOODClassifier 中 LOF 的异常点比例
    LOF_THRESHOLD = -0.5      # LOFOODClassifier 中 LOF 的 decision_function 阈值

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

# ------------------------- 模型初始化 -------------------------
class TransformerEncoderWrapper(nn.Module):
    """从 TransformerAutoencoder 中提取编码器部分的包装器"""
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        _, latent = self.autoencoder(x)
        return latent

class TransformerDecoderWrapper(nn.Module):
    """从 TransformerAutoencoder 中提取解码器部分的包装器"""
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, latent):
        # TransformerAutoencoder 的解码器需要编码器输出作为记忆，这里直接调用完整 forward
        # 假设输入 latent 是编码器输出，重建原始输入
        reconstructed, _ = self.autoencoder.decoder(latent.unsqueeze(1), latent.unsqueeze(1))
        return reconstructed.squeeze(1)

def create_model():
    """
    初始化自编码器模型，根据 SELF_SUPERVISED_METHOD 选择不同模型
    返回：encoder, decoder, autoencoder 三元组
    """
    if Config.SELF_SUPERVISED_METHOD in ['lrcn', 'simclr']:
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
            use_bdc=Config.USE_BDC,
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
    else:
        raise ValueError(f"Unknown SELF_SUPERVISED_METHOD: {Config.SELF_SUPERVISED_METHOD}")

    return encoder, decoder, autoencoder

# ------------------------- 主训练流程 -------------------------
def train(train_loader, train_loader2):

    # 2. 模型初始化
    encoder, decoder, autoencoder = create_model()

    # 3. 基准分类器测试（使用原始 encoder）
    print("\n---------- 原始分类器效果 ---------")
    time.sleep(0.1)
    baseline_classifier(
        encoder, train_loader2, test_loader,
        hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION,
        epochs=1,
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
        else:
            raise ValueError(f"Unknown SELF_SUPERVISED_METHOD: {Config.SELF_SUPERVISED_METHOD}")

        print("\n---------- 自监督预训练后的分类器效果 ---------")
        time.sleep(0.1)
        baseline_classifier(
            pretrain_encoder, train_loader2, test_loader,
            hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION,
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
            contrastive_encoder.encoder_q, train_loader2, test_loader,
            hidden_size=Config.HIDDEN_DIM,
            epochs=Config.PRETRAIN_EPOCHS,
            learn_rate=Config.BASE_LR,
            device=Config.DEVICE,
            title="对比学习后的分类器"
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
        #base_encoder=TransformerEncoder(input_dim=Config.INPUT_DIM, latent_dim=Config.LATENT_DIM, d_model=128,transformer_layers=3, transformer_heads=2),
        feature_dim=Config.FEATURE_DIM,
        num_classes=Config.CLASS_NUM,
        K=Config.QUEUE_SIZE,
        m=Config.MOMENTUM,
        T=Config.TEMPERATURE,
        top_k=Config.TOP_K,
        device=Config.DEVICE
    ).to(Config.DEVICE)
    encoder.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))

    # 6. 最终分类测试（LOF）


    print("\n---------- ODIN 分类器测试 ---------")
    time.sleep(0.1)
    start_time = time.time()
    ODINtest(
        encoder=encoder,
        test_loader=test_loader,
        device=Config.DEVICE,
        temperature=1000,
        epsilon=0.0014,
        threshold=0.20241
    )
    end_time = time.time()
    print("ODIN 耗时: {:.4f}秒".format(end_time - start_time))


    print("\n---------- IForest 分类器测试 ---------")
    time.sleep(0.1)
    start_time = time.time()
    ConfigurableOODtest(
        encoder, train_loader2, test_loader,
        class_number=Config.CLASS_NUM,
        device=Config.DEVICE,
        outlier_detector_str="IForest",
        contamination=0.2
    )

    end_time = time.time()
    print("IForest 耗时: {:.4f}秒".format(end_time - start_time))

    # print("\n---------- LOF 分类器测试 ---------")
    # time.sleep(0.1)
    # # 构建 LOF 配置参数字典
    # lof_params = {
    #     'k': Config.LOF_K,
    #     'knn_p': Config.LOF_KNN_P,
    #     'n_neighbors': Config.LOF_N_NEIGHBORS,
    #     'contamination': Config.LOF_CONTAMINATION
    # }
    # start_time = time.time()
    # LOFtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     lof_params=lof_params
    # )
    # end_time = time.time()
    # print("LOFtest 耗时: {:.4f}秒".format(end_time - start_time))

    # print("\n---------- OCSVM 分类器测试 ---------")
    # time.sleep(0.1)
    # start_time = time.time()
    # ConfigurableOODtest(
    #     encoder, train_loader2, test_loader,
    #     class_number=Config.CLASS_NUM,
    #     device=Config.DEVICE,
    #     outlier_detector_str="OCSVM",
    #     contamination=0.2
    # )

    end_time = time.time()
    print("OCSVM 耗时: {:.4f}秒".format(end_time - start_time))




if __name__ == "__main__":
    # 1. 数据加载
    train_loader, train_loader2, test_loader = load_data()

    #train(train_loader, train_loader2)
    evaluate_lof_knn(train_loader2, test_loader,model_path="contrastive_encoder_f1_0_9121.pth")
