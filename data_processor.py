"""
比特币价格预测 - 数据处理模块
负责数据加载、清洗、归一化和数据集创建
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle


class BitcoinDataset(Dataset):
    """PyTorch数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_data(self, file_path=None):
        """加载Excel数据"""
        if file_path is None:
            file_path = self.config.DATA_FILE
            
        print(f"\n{'='*70}")
        print("📂 加载数据")
        print(f"{'='*70}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"✓ 成功加载文件: {file_path}")
            print(f"  - 数据行数: {df.shape[0]}")
            print(f"  - 数据列数: {df.shape[1]}")
        except Exception as e:
            print(f"❌ 读取文件时出错: {str(e)}")
            raise
        
        # 重命名列（如果需要）
        if 'Price' in df.columns and 'Close' not in df.columns:
            df = df.rename(columns={'Price': 'Close'})
        
        # 确保日期格式正确
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"⚠️ 日期转换警告: {str(e)}")
        
        # 处理Close列的数据类型
        if df['Close'].dtype == 'object':
            df['Close'] = df['Close'].str.replace(',', '').astype(float)
        
        # 确保数据从旧到新排序
        if df['Date'].iloc[0] > df['Date'].iloc[-1]:
            df = df.sort_values('Date', ascending=True).reset_index(drop=True)
            print("✓ 数据已按时间正序排列")
        
        # 清洗数据
        df = df.dropna(subset=['Close'])
        df = df[df['Close'] > 0].reset_index(drop=True)
        
        self.data = df
        
        print(f"\n✓ 数据清洗完成")
        print(f"  - 有效数据: {len(df)} 天")
        print(f"  - 日期范围: {df['Date'].iloc[0].date()} 到 {df['Date'].iloc[-1].date()}")
        print(f"\n📊 价格统计:")
        print(f"  - 最高价: ${df['Close'].max():,.2f}")
        print(f"  - 最低价: ${df['Close'].min():,.2f}")
        print(f"  - 平均价: ${df['Close'].mean():,.2f}")
        print(f"  - 当前价: ${df['Close'].iloc[-1]:,.2f}")
        
        return df
    
    def normalize_data(self):
        """归一化数据"""
        close_prices = self.data[['Close']].values
        self.scaled_data = self.scaler.fit_transform(close_prices)
        return self.scaled_data
    
    def create_sequences(self, data, time_step):
        """创建时间序列数据
        
        Args:
            data: 归一化后的数据
            time_step: 时间步长
            
        Returns:
            X, y: 特征和标签数组
        """
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """准备训练和测试数据"""
        print(f"\n{'='*70}")
        print("🔧 准备训练数据")
        print(f"{'='*70}")
        
        # 归一化
        scaled_data = self.normalize_data()
        
        # 划分训练集和测试集
        training_size = int(len(scaled_data) * self.config.TRAIN_RATIO)
        train_data = scaled_data[0:training_size, :]
        test_data = scaled_data[training_size:, :]
        
        print(f"\n✓ 数据集划分:")
        print(f"  - 训练集: {training_size} 天 ({self.config.TRAIN_RATIO*100:.0f}%)")
        print(f"  - 测试集: {len(test_data)} 天 ({(1-self.config.TRAIN_RATIO)*100:.0f}%)")
        
        # 创建序列
        X_train, y_train = self.create_sequences(train_data, self.config.TIME_STEP)
        X_test, y_test = self.create_sequences(test_data, self.config.TIME_STEP)
        
        # 重塑为LSTM输入格式: (samples, time_steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        print(f"\n📐 数据形状:")
        print(f"  - X_train: {X_train.shape} (样本数, 时间步, 特征数)")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, test_data
    
    def create_dataloaders(self, X_train, y_train, X_test, y_test):
        """创建PyTorch DataLoader"""
        train_dataset = BitcoinDataset(X_train, y_train)
        test_dataset = BitcoinDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def inverse_transform(self, data):
        """反归一化"""
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, path=None):
        """保存scaler"""
        if path is None:
            path = self.config.SCALER_SAVE_PATH
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler已保存: {path}")
    
    def load_scaler(self, path=None):
        """加载scaler"""
        if path is None:
            path = self.config.SCALER_SAVE_PATH
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler已加载: {path}")


if __name__ == "__main__":
    # 测试数据处理器
    from config import Config
    
    config = Config()
    processor = DataProcessor(config)
    
    # 加载数据
    df = processor.load_data()
    
    # 准备数据
    X_train, y_train, X_test, y_test, test_data = processor.prepare_data()
    
    # 创建DataLoader
    train_loader, test_loader = processor.create_dataloaders(X_train, y_train, X_test, y_test)
    
    print(f"\n✓ 数据处理测试完成！")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 测试批次数: {len(test_loader)}")