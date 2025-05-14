"""
主程序文件：基于LR-CCN模型的训练和评估流程
包含：基础分类器、自监督预训练、对比学习以及最终的 LOF+KNN 分类器
"""
print("重跑网格搜索")
import torch
import time
import random
import numpy as np
import logging
import sys
from datetime import datetime
import pprint
from itertools import product

# 导入各模块（请根据实际项目目录调整导入路径）
from LRCCNdataloader import LRCCNdataload
from LRCCNpretrain import pretrain_model,pretrain_simclr, pretrain_Transformermodel, pretrain_VAE, create_model  # 自监督预训练模块
from LRCCNbaseline import baseline_classifier
from LRCCNmoco import ContrastiveLearningKNN,MoCoEncoderKNN
from LRCCNFinalClassifier import LOFtest,KNNtest,LOF_KNN_test,ODINtest, ConfigurableOODtest,INFLOtest
import torch.nn as nn
from model.LRCN2D import LRCNAutoencoder, LRCNencoder, LRCNdecoder

# ------------------------- 全局配置 -------------------------
class Config:
    # 硬件配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据参数
    DATA_PROPORTION = 0.1        # 训练数据使用比例
    BATCH_SIZE = 1024            # 批次大小
    CHANNEL_SIZE = 1             # 输入通道数

    # 自监督预训练参数
    SELF_SUPERVISED_METHOD = 'lrcn'  # 'lrcn' 或 'simclr'，或 'transformer'，或 'vae'，选择自监督方法
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
    PRETRAIN_LR = 0.0001          # 自监督预训练学习率
    PRETRAIN_EPOCHS = 50          # 自监督预训练轮次
    CONTRASTIVE_EPOCHS = 50       # 对比学习轮次

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

# ------------------------- 网格搜索参数范围 (MODIFIED) -------------------------
GRID_SEARCH_PARAMS = {
    'PRETRAIN_MASK_RATIO': [0.01, 0.1, 0.2, 0.3],    # Added for grid search
    'TEMPERATURE': [0.01, 0.1, 0.2, 0.3],          # Added for grid search
    'TOP_K': [30, 50, 70, 90]                      # Added for grid search
}


# ------------------------- 固定随机种子 -------------------------
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


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


# ------------------------- 数据加载和模型初始化 -------------------------
def load_data():
    logging.info("开始加载数据...")
    start_time = time.time()
    loaders = LRCCNdataload(
        proportion=Config.DATA_PROPORTION,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE
    )
    logging.info(f"数据加载完成, 耗时: {time.time() - start_time:.2f} 秒")
    return loaders


def create_model():
    logging.info("创建模型架构...")
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
    logging.info("模型创建完成.")
    return encoder, decoder, autoencoder


# ------------------------- 训练和评估单组参数 (MODIFIED) -------------------------
def train_and_evaluate(train_loader, train_loader2, test_loader, params):
    # **每次评估都需要重新创建和训练模型，以隔离不同参数组合的影响**
    encoder, decoder, autoencoder = create_model()

    # --- 获取当前参数组合 (MODIFIED) ---
    PRETRAIN_MASK_RATIO = params['PRETRAIN_MASK_RATIO']
    TEMPERATURE = params['TEMPERATURE']
    TOP_K = params['TOP_K']

    # --- 日志记录当前参数 (MODIFIED) ---
    current_params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    logging.info(f"\n--- 开始测试参数组合: {current_params_str} ---")

    run_start_time = time.time()

    # # --- 1. 基准分类器 (使用新创建的 Encoder) ---
    # logging.info("\n---------- 1. 原始分类器效果 ---------")
    # baseline_start_time = time.time()
    # baseline_classifier2D(
    #     encoder, train_loader2, test_loader,
    #     # 注意：这里的 hidden_size 应该与 Encoder 输出特征维度匹配
    #     hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION if Config.USE_BDC else Config.HIDDEN_DIM * 2, # Adjust if BDC isn't used
    #     epochs=Config.PRETRAIN_EPOCHS, # Using PRETRAIN_EPOCHS for baseline training duration
    #     learn_rate=Config.BASE_LR,
    #     device=Config.DEVICE
    # )
    # logging.info(f"原始分类器训练与评估完成, 耗时: {time.time() - baseline_start_time:.2f} 秒")
    # *重要*: 复制一份 Encoder 用于后续训练，避免修改原始基准测试用的 Encoder
    current_encoder = LRCNencoder(
        input_size=Config.INPUT_DIM,
        hidden_size=Config.HIDDEN_DIM,
        use_bdc=Config.USE_BDC,
        bdc_input_dim=Config.BDC_INPUT_DIMENSION_REDUCTION
    ).to(Config.DEVICE)
    current_encoder.load_state_dict(encoder.state_dict()) # Start from same initial weights as baseline


    # --- 2. 自监督预训练 ---
    if Config.ENABLE_PRETRAIN:
        logging.info("\n---------- 2. 自监督预训练 ---------")
        pretrain_start_time = time.time()
        # 需要使用新的 decoder 和 autoencoder 实例配合 current_encoder
        current_decoder = LRCNdecoder(
            output_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            use_bdc=Config.USE_BDC,
            bdc_output_dim=Config.BDC_OUTPUT_DIMENSION_REDUCTION
        ).to(Config.DEVICE)
        current_autoencoder = LRCNAutoencoder(current_encoder, current_decoder).to(Config.DEVICE)

        # 使用从 params 传入的 PRETRAIN_MASK_RATIO (MODIFIED)
        pretrain_encoder = pretrain_model(
            train_loader,
            (current_encoder, current_decoder, current_autoencoder),
            lr=Config.PRETRAIN_LR,
            epochs=Config.PRETRAIN_EPOCHS,
            device=Config.DEVICE,
            mask_ratio=PRETRAIN_MASK_RATIO, # Use parameter from grid search
            weight_decay=Config.PRETRAIN_WEIGHT_DECAY
        )
        logging.info(f"自监督预训练完成, 耗时: {time.time() - pretrain_start_time:.2f} 秒")

    #     # --- 3. 预训练后分类器评估 ---
    #     logging.info("\n---------- 3. 自监督预训练后的分类器效果 ---------")
    #     post_pretrain_eval_start_time = time.time()
    #     baseline_classifier2D(
    #         pretrain_encoder, train_loader2, test_loader,
    #         hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION if Config.USE_BDC else Config.HIDDEN_DIM * 2,
    #         epochs=Config.PRETRAIN_EPOCHS, # Consistent training duration for comparison
    #         learn_rate=Config.BASE_LR,
    #         device=Config.DEVICE
    #     )
    #     logging.info(f"预训练后分类器评估完成, 耗时: {time.time() - post_pretrain_eval_start_time:.2f} 秒")
    # else:
    #     logging.info("\n跳过自监督预训练 (ENABLE_PRETRAIN=False)")
    #     pretrain_encoder = current_encoder # Use the encoder directly if pretraining is disabled


    # --- 4. 对比学习 ---
    if Config.ENABLE_CONTRASTIVE:
        logging.info("\n---------- 4. 对比学习 ---------")
        contrastive_start_time = time.time()
        # 使用从 params 传入的 TOP_K 和 TEMPERATURE (MODIFIED)
        contrastive_learner = ContrastiveLearningKNN(
            encoder=pretrain_encoder, # Use the potentially pre-trained encoder
            train_loader=train_loader2,
            epochs=Config.CONTRASTIVE_EPOCHS,
            device=Config.DEVICE,
            class_number=Config.CLASS_NUM,
            contrastive_rate=Config.CONTRASTIVE_RATE,
            feature_dim=Config.BDC_OUTPUT_DIMENSION_REDUCTION if Config.USE_BDC else Config.HIDDEN_DIM * 2,
            top_k=TOP_K,             # Use parameter from grid search
            queue_size=Config.QUEUE_SIZE,
            lr=Config.CONTRASTIVE_LR,
            m=Config.MOMENTUM,
            T=TEMPERATURE          # Use parameter from grid search
        )
        # The ContrastiveLearningKNN class likely modifies the encoder in-place or returns the trained one
        # Assuming it returns the trained query encoder:
        final_encoder = contrastive_learner.encoder_q
        logging.info(f"对比学习完成, 耗时: {time.time() - contrastive_start_time:.2f} 秒")


        # --- 5. 对比学习后分类器评估 ---
        logging.info("\n---------- 5. 对比学习后的分类器效果 ---------")
        post_contrastive_eval_start_time = time.time()
        # Note: Hidden size for classifier after contrastive learning might need adjustment
        # depending on whether contrastive learning changes output dimension. Assuming it doesn't.
        baseline_classifier(
            final_encoder, train_loader2, test_loader,
            hidden_size=Config.BDC_OUTPUT_DIMENSION_REDUCTION if Config.USE_BDC else Config.HIDDEN_DIM * 2,
            epochs=Config.PRETRAIN_EPOCHS, # Consistent training duration
            learn_rate=Config.BASE_LR,
            device=Config.DEVICE
        )
        logging.info(f"对比学习后分类器评估完成, 耗时: {time.time() - post_contrastive_eval_start_time:.2f} 秒")
    else:
        logging.info("\n跳过对比学习 (ENABLE_CONTRASTIVE=False)")
        final_encoder = pretrain_encoder # Use the encoder from the previous step

    # --- 6. 最终分类/异常检测测试 (ODIN in this case) ---
    # Note: The original code includes ODINtest but the grid search parameter was LOF_CONTAMINATION.
    # This suggests either the final test should be LOF_KNN_test or the parameter name is misleading.
    # Keeping ODINtest as per the original snippet provided for modification.
    # If LOF/KNN is the intended final step, replace ODINtest with LOF_KNN_test and pass relevant params.
    logging.info("\n---------- 6. ODIN 分类器测试 ---------")
    odin_start_time = time.time()
    time.sleep(0.1) # Small delay as in original code
    # ODIN test uses the 'final_encoder' after all training steps
    # ODIN parameters are currently hardcoded in the call below, not part of grid search.
    ODINtest(
        encoder=final_encoder,
        test_loader=test_loader,
        device=Config.DEVICE,
        temperature=50, # Example value, consider making it configurable or part of grid search if needed
        epsilon=0.0014, # Example value
        threshold=0.16  # Example value
    )
    logging.info(f"ODIN 测试完成, 耗时: {time.time() - odin_start_time:.2f} 秒")


    logging.info(f"--- 参数组合 {current_params_str} 测试完成, 总耗时: {time.time() - run_start_time:.2f} 秒 ---")
    sys.stdout.flush() # Ensure all logs are written


# ------------------------- 主函数 -------------------------
def main():
    overall_start_time = time.time()
    # --- 加载数据 (只需加载一次) ---
    try:
        train_loader, train_loader2, test_loader = load_data()
    except Exception as e:
        logging.error(f"数据加载失败: {e}", exc_info=True)
        return # Exit if data loading fails

    # --- 生成所有参数组合 ---
    param_names = list(GRID_SEARCH_PARAMS.keys())
    param_values = list(GRID_SEARCH_PARAMS.values())
    param_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

    num_combinations = len(param_combinations)
    logging.info(f"\n总共需要测试 {num_combinations} 组参数组合.")
    logging.warning("网格搜索可能需要很长时间，具体取决于参数范围和训练轮数。")
    logging.info("-" * 50)


    # --- 遍历所有参数组合 ---
    for i, params in enumerate(param_combinations, 1):
        logging.info(f"\n{'='*20} 网格搜索进度: {i}/{num_combinations} {'='*20}")
        try:
            # **关键**: 每次迭代都使用加载好的数据，但模型重新初始化和训练
            train_and_evaluate(train_loader, train_loader2, test_loader, params)
        except Exception as e:
            current_params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            logging.error(f"\n!!!!!! 参数组合 {current_params_str} (进度 {i}/{num_combinations}) 执行时发生错误 !!!!!!")
            logging.error(f"错误信息: {e}", exc_info=True) # Log traceback
            logging.info("继续执行下一个参数组合...") # Option to continue or break
            # break # Uncomment to stop grid search on first error

        # Optional: Add delay between runs if needed (e.g., GPU cooldown)
        # time.sleep(5)

    overall_end_time = time.time()
    logging.info("\n网格搜索完成！")
    logging.info(f"总耗时: {overall_end_time - overall_start_time:.2f} 秒 ({datetime.timedelta(seconds=overall_end_time - overall_start_time)})")
    logging.info(f"详细结果请查看日志文件: {log_filename}")


if __name__ == "__main__":
    main()