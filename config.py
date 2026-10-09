"""
比特币价格预测 - 配置文件
可以通过命令行参数覆盖这些默认值
"""

class Config:
    # 数据文件路径
    DATA_FILE = 'Bitcoin_Historical_Data_10年.xlsx'
    
    # 模型参数
    TIME_STEP = 30          # 时间步长（使用多少天的数据来预测）
    TRAIN_RATIO = 0.85      # 训练集比例
    
    # 训练参数
    EPOCHS = 50             # 训练轮数
    BATCH_SIZE = 32         # 批次大小
    LEARNING_RATE = 0.001   # 学习率
    
    # LSTM网络结构
    LSTM_UNITS_1 = 50       # 第一层LSTM单元数
    LSTM_UNITS_2 = 50       # 第二层LSTM单元数（可选）
    DROPOUT = 0.2           # Dropout比例
    
    # 预测参数
    PRED_DAYS = 3           # 预测未来多少天
    
    # 训练优化
    EARLY_STOP_PATIENCE = 15  # 早停耐心值
    VAL_SPLIT = 0.1           # 验证集比例
    
    # 设备设置
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'，会自动检测
    
    # 保存路径
    MODEL_SAVE_PATH = 'bitcoin_lstm_model.pth'
    SCALER_SAVE_PATH = 'scaler.pkl'
    RESULTS_CSV = 'future_predictions.csv'
    
    # 随机种子
    RANDOM_SEED = 42
    
    def __init__(self, **kwargs):
        """允许通过关键字参数覆盖默认配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        """打印配置"""
        config_str = "=" * 70 + "\n"
        config_str += "配置参数\n"
        config_str += "=" * 70 + "\n"
        for key in dir(self):
            if not key.startswith('_') and key.isupper():
                config_str += f"{key:25s}: {getattr(self, key)}\n"
        config_str += "=" * 70 + "\n"
        return config_str