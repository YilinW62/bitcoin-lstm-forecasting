"""
比特币价格预测 - 多对多 (Seq2Seq) LSTM模型
真正的时间序列预测模式
"""

import torch
import torch.nn as nn


class BitcoinLSTM_Seq2Seq(nn.Module):
    """
    多对多LSTM模型 - 直接预测未来N天
    输入: [batch, time_step, 1]
    输出: [batch, pred_days, 1] ← 一次性输出多天预测
    """

    def __init__(self,
                 input_size=1,
                 hidden_size=50,
                 num_layers=1,
                 dropout=0.2,
                 pred_days=3):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            pred_days: 预测多少天（输出序列长度）
        """
        super(BitcoinLSTM_Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_days = pred_days

        # Encoder LSTM（编码器）
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder LSTM（解码器）
        self.decoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        前向传播 - Seq2Seq架构

        Args:
            x: 输入张量 [batch_size, time_step, 1]

        Returns:
            predictions: [batch_size, pred_days, 1]
        """
        batch_size = x.size(0)

        # 1. Encoder: 处理输入序列
        _, (hidden, cell) = self.encoder_lstm(x)
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]

        # 2. Decoder: 生成预测序列
        # 初始输入是最后一个时间步的值
        decoder_input = x[:, -1:, :]  # [batch_size, 1, 1]

        predictions = []

        for _ in range(self.pred_days):
            # 解码一步
            decoder_output, (hidden, cell) = self.decoder_lstm(
                decoder_input, (hidden, cell)
            )
            # decoder_output: [batch_size, 1, hidden_size]

            # 预测当前步
            pred = self.fc(decoder_output)  # [batch_size, 1, 1]
            predictions.append(pred)

            # 下一步的输入是当前预测（训练时）或真实值（推理时）
            decoder_input = pred

        # 拼接所有预测
        predictions = torch.cat(predictions, dim=1)  # [batch_size, pred_days, 1]

        return predictions

    def predict(self, x):
        """
        预测模式（等同于forward，但更明确）
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class BitcoinLSTM_Direct(nn.Module):
    """
    直接多输出LSTM模型 - 更简单的多对多实现
    输入: [batch, time_step, 1]
    输出: [batch, pred_days] ← 直接输出N个值
    """

    def __init__(self,
                 input_size=1,
                 hidden_size=50,
                 num_layers=1,
                 dropout=0.2,
                 pred_days=3):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            pred_days: 预测多少天
        """
        super(BitcoinLSTM_Direct, self).__init__()

        self.hidden_size = hidden_size
        self.pred_days = pred_days

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层：直接输出pred_days个值
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, pred_days)  # ← 输出层大小=预测天数

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, time_step, 1]

        Returns:
            predictions: [batch_size, pred_days]
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        out = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Dropout
        out = self.dropout(out)

        # 全连接层
        out = self.fc1(out)  # [batch_size, 50]
        out = self.relu(out)
        out = self.fc2(out)  # [batch_size, pred_days]

        return out

    def predict(self, x):
        """预测模式"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)  # [batch_size, pred_days]
            return predictions.unsqueeze(-1)  # [batch_size, pred_days, 1]


def create_seq2seq_model(config, model_type='direct'):
    """
    创建多对多模型

    Args:
        config: 配置对象
        model_type: 'seq2seq' 或 'direct'

    Returns:
        PyTorch模型
    """
    if model_type == 'seq2seq':
        model = BitcoinLSTM_Seq2Seq(
            input_size=1,
            hidden_size=config.LSTM_UNITS_1,
            num_layers=1,
            dropout=config.DROPOUT,
            pred_days=config.PRED_DAYS
        )
    elif model_type == 'direct':
        model = BitcoinLSTM_Direct(
            input_size=1,
            hidden_size=config.LSTM_UNITS_1,
            num_layers=1,
            dropout=config.DROPOUT,
            pred_days=config.PRED_DAYS
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    from config import Config

    config = Config(TIME_STEP=60, PRED_DAYS=3)

    print("="*70)
    print("测试Seq2Seq LSTM模型（编码器-解码器）")
    print("="*70)
    model_seq2seq = create_seq2seq_model(config, model_type='seq2seq')
    print(model_seq2seq)
    print(f"\n总参数数量: {count_parameters(model_seq2seq):,}")

    # 测试前向传播
    batch_size = 8
    x = torch.randn(batch_size, config.TIME_STEP, 1)
    output = model_seq2seq(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}  ← 一次性输出{config.PRED_DAYS}天预测")

    print("\n" + "="*70)
    print("测试Direct LSTM模型（直接多输出）")
    print("="*70)
    model_direct = create_seq2seq_model(config, model_type='direct')
    print(model_direct)
    print(f"\n总参数数量: {count_parameters(model_direct):,}")

    output = model_direct(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}  ← 直接输出{config.PRED_DAYS}个值")

    print("\n✓ 两种模型都是真正的多对多模式！")
    print("  - Seq2Seq: 编码器-解码器架构，更适合长序列预测")
    print("  - Direct: 直接输出N个值，实现更简单")