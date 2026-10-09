"""
比特币价格预测 - 多对多模式训练脚本
使用Direct LSTM: 一次性预测未来N天
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from model import create_seq2seq_model, count_parameters

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Seq2SeqDataProcessor(DataProcessor):
    """多对多模型的数据处理器"""

    def create_sequences_seq2seq(self, data, time_step, pred_days):
        """
        创建多对多的数据序列

        Args:
            data: 归一化后的数据
            time_step: 输入时间步长
            pred_days: 预测天数（输出长度）

        Returns:
            X: [样本数, time_step, 1]
            y: [样本数, pred_days] ← 每个样本有pred_days个目标值
        """
        X, y = [], []
        for i in range(len(data) - time_step - pred_days + 1):
            X.append(data[i:(i + time_step), 0])
            # y包含未来pred_days天的值
            y.append(data[(i + time_step):(i + time_step + pred_days), 0])

        return np.array(X), np.array(y)

    def prepare_data_seq2seq(self):
        """准备多对多训练数据"""
        print(f"\n{'='*70}")
        print("🔧 准备多对多训练数据")
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
        X_train, y_train = self.create_sequences_seq2seq(
            train_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )
        X_test, y_test = self.create_sequences_seq2seq(
            test_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )

        # 重塑为LSTM输入格式
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        # y不需要reshape，保持 [样本数, pred_days]

        print(f"\n📐 数据形状:")
        print(f"  - X_train: {X_train.shape} (样本数, 时间步, 特征数)")
        print(f"  - y_train: {y_train.shape} (样本数, 预测天数) ← 多对多输出")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")

        return X_train, y_train, X_test, y_test, test_data


class Seq2SeqTrainer:
    """多对多模型训练器"""

    def __init__(self, config):
        self.config = config

        if config.DEVICE == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        print(f"\n🖥️  使用设备: {self.device}")

        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 前向传播
            outputs = model(X_batch)  # [batch, pred_days]
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        """验证"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, model, train_loader, val_loader):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print(f"🚀 开始训练多对多模型")
        print(f"{'='*70}")

        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=self.config.LEARNING_RATE)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self.validate(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{self.config.EPOCHS}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\n⚠️  早停触发！验证损失已 {patience_counter} 轮未改善")
                model.load_state_dict(torch.load('best_model_temp.pth'))
                break

        print(f"\n✓ 训练完成！最佳验证损失: {best_val_loss:.6f}")

        return train_losses, val_losses

    def evaluate(self, model, data_loader, data_processor, pred_days):
        """评估模型"""
        model.eval()
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)  # [batch, pred_days]

                all_predictions.extend(outputs.cpu().numpy())
                all_actuals.extend(y_batch.numpy())

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)  # [样本数, pred_days]
        all_actuals = np.array(all_actuals)  # [样本数, pred_days]

        # 反归一化（需要reshape）
        pred_reshaped = all_predictions.reshape(-1, 1)
        actual_reshaped = all_actuals.reshape(-1, 1)

        pred_inverse = data_processor.inverse_transform(pred_reshaped)
        actual_inverse = data_processor.inverse_transform(actual_reshaped)

        # 计算指标
        rmse = math.sqrt(mean_squared_error(actual_inverse, pred_inverse))
        mae = mean_absolute_error(actual_inverse, pred_inverse)
        r2 = r2_score(actual_inverse, pred_inverse)

        # reshape回去
        pred_inverse = pred_inverse.reshape(-1, pred_days)
        actual_inverse = actual_inverse.reshape(-1, pred_days)

        return pred_inverse, actual_inverse, rmse, mae, r2

    def predict_future(self, model, test_data, data_processor):
        """预测未来N天（一次性输出）"""
        print(f"\n{'='*70}")
        print(f"🔮 预测未来 {self.config.PRED_DAYS} 天（一次性输出）")
        print(f"{'='*70}")

        model.eval()

        # 使用最后TIME_STEP天的真实数据
        last_sequence = test_data[-self.config.TIME_STEP:].reshape(1, self.config.TIME_STEP, 1)
        last_sequence = torch.FloatTensor(last_sequence).to(self.device)

        # 一次性预测
        with torch.no_grad():
            predictions = model(last_sequence)  # [1, pred_days]

        # 反归一化
        predictions = predictions.cpu().numpy().reshape(-1, 1)
        predictions = data_processor.inverse_transform(predictions)

        print(f"✓ 预测完成（无误差累积！）")

        return predictions


def plot_training_history(train_losses, val_losses, save_path='training_history_seq2seq.png'):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='训练损失', linewidth=2, color='blue')
    plt.plot(val_losses, label='验证损失', linewidth=2, color='orange')
    plt.title('多对多模型训练历史', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失值 (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练历史图已保存: {save_path}")
    plt.close()


def plot_test_predictions(test_actual, test_pred, test_rmse, test_mae, test_r2, pred_days, save_path='test_predictions.png'):
    """
    绘制测试集预测对比图

    Args:
        test_actual: 实际价格 [样本数, pred_days]
        test_pred: 预测价格 [样本数, pred_days]
        test_rmse: RMSE指标
        test_mae: MAE指标
        test_r2: R²指标
        pred_days: 预测天数
        save_path: 保存路径
    """
    # 展平数据用于绘图
    actual_flat = test_actual.flatten()
    pred_flat = test_pred.flatten()

    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制实际价格和预测价格
    ax.plot(actual_flat, label='实际价格', linewidth=2, alpha=0.8, color='blue')
    ax.plot(pred_flat, label='预测价格', linewidth=2, alpha=0.7, color='orange')

    ax.set_title('测试集预测对比', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('时间（天）', fontsize=14)
    ax.set_ylabel('价格 (USD)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 添加性能指标文本框
    textstr = f'RMSE: ${test_rmse:,.2f}\nMAE: ${test_mae:,.2f}\nR²: {test_r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 测试集预测图已保存: {save_path}")
    plt.close()


def plot_future_predictions(historical_data, future_predictions, time_step, pred_days,
                           current_date, save_path='future_predictions.png'):
    """
    绘制未来预测图

    Args:
        historical_data: 最近time_step天的历史数据 [time_step]
        future_predictions: 未来预测 [pred_days, 1]
        time_step: 时间步长
        pred_days: 预测天数
        current_date: 当前日期
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # X轴：历史数据和预测数据
    hist_days = np.arange(1, time_step + 1)
    pred_days_x = np.arange(time_step + 1, time_step + pred_days + 1)

    # 绘制历史数据
    ax.plot(hist_days, historical_data,
            label=f'最近 {time_step} 天实际价格',
            marker='o', linewidth=2.5, markersize=5, color='blue')

    # 绘制预测数据
    future_flat = future_predictions.flatten()
    ax.plot(pred_days_x, future_flat,
            label=f'未来 {pred_days} 天预测价格',
            marker='s', linewidth=3, markersize=8, color='red', linestyle='--')

    # 标注每个预测值
    for i, (day, price) in enumerate(zip(pred_days_x, future_flat)):
        ax.annotate(f'${price:,.0f}',
                    xy=(day, price),
                    xytext=(8, 10),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 添加预测起点线
    ax.axvline(x=time_step, color='green', linestyle=':', linewidth=2.5,
               label='预测起点', alpha=0.8)

    ax.set_title(f'比特币价格：最近 {time_step} 天 vs 未来 {pred_days} 天预测',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('天数', fontsize=13)
    ax.set_ylabel('价格 (USD)', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 未来预测图已保存: {save_path}")
    plt.close()


def print_predictions(data_processor, future_predictions, pred_days):
    """打印预测结果"""
    current_price = data_processor.data['Close'].iloc[-1]
    current_date = data_processor.data['Date'].iloc[-1]

    print(f"\n{'='*70}")
    print(f"📈 未来 {pred_days} 天价格预测（多对多模式 - 无误差累积）")
    print(f"{'='*70}")
    print(f"\n当前价格: ${current_price:,.2f} ({current_date.date()})\n")

    for i in range(pred_days):
        pred_price = future_predictions[i][0]
        change = pred_price - current_price
        change_pct = (change / current_price) * 100

        future_date = current_date + pd.Timedelta(days=i+1)

        print(f"第 {i+1} 天 ({future_date.date()}):")
        print(f"  预测价格: ${pred_price:,.2f}")
        print(f"  价格变化: ${change:+,.2f} ({change_pct:+.2f}%)")
        print()

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='比特币价格预测 - 多对多模式')
    parser.add_argument('--time_step', type=int, default=60,
                        help='时间步长')
    parser.add_argument('--pred_days', type=int, default=3,
                        help='预测天数')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lstm_units', type=int, default=50,
                        help='LSTM单元数')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='运行设备')
    parser.add_argument('--data_file', type=str, default='Bitcoin_Historical_Data_10年.xlsx',
                        help='数据文件路径')

    args = parser.parse_args()

    config = Config(
        TIME_STEP=args.time_step,
        PRED_DAYS=args.pred_days,
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LSTM_UNITS_1=args.lstm_units,
        LEARNING_RATE=args.learning_rate,
        DEVICE=args.device,
        DATA_FILE=args.data_file
    )

    print("="*70)
    print("🚀 比特币价格预测系统 - 多对多模式 (无误差累积)")
    print("="*70)
    print(config)

    # 1. 数据处理
    data_processor = Seq2SeqDataProcessor(config)
    data_processor.load_data()
    X_train, y_train, X_test, y_test, test_data = data_processor.prepare_data_seq2seq()

    # 划分验证集
    val_size = int(len(X_train) * config.VAL_SPLIT)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    train_loader, _ = data_processor.create_dataloaders(X_train, y_train, X_test, y_test)
    val_loader, test_loader = data_processor.create_dataloaders(X_val, y_val, X_test, y_test)

    print(f"\n✓ 数据集最终划分:")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 验证集: {len(X_val)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")

    # 2. 创建模型
    print(f"\n{'='*70}")
    print(f"🏗️  构建多对多模型")
    print(f"{'='*70}")

    model = create_seq2seq_model(config, model_type='direct')

    trainer = Seq2SeqTrainer(config)
    model = model.to(trainer.device)

    print(model)
    print(f"\n总参数数量: {count_parameters(model):,}")

    # 3. 训练
    train_losses, val_losses = trainer.train(model, train_loader, val_loader)

    # 4. 绘制训练历史
    plot_training_history(train_losses, val_losses)

    # 5. 评估
    print(f"\n{'='*70}")
    print("📊 评估模型性能")
    print(f"{'='*70}")

    print("\n【测试集性能】")
    test_pred, test_actual, test_rmse, test_mae, test_r2 = trainer.evaluate(
        model, test_loader, data_processor, config.PRED_DAYS
    )
    print(f"  - RMSE: ${test_rmse:,.2f}")
    print(f"  - MAE:  ${test_mae:,.2f}")
    print(f"  - R²:   {test_r2:.4f}")

    # 6. 预测未来
    future_predictions = trainer.predict_future(model, test_data, data_processor)

    # 7. 打印结果
    print_predictions(data_processor, future_predictions, config.PRED_DAYS)

    # 8. 生成可视化图表
    print(f"\n{'='*70}")
    print("📈 生成可视化图表")
    print(f"{'='*70}\n")

    # 8.1 测试集预测对比图
    plot_test_predictions(
        test_actual,
        test_pred,
        test_rmse,
        test_mae,
        test_r2,
        config.PRED_DAYS,
        save_path='test_predictions.png'
    )

    # 8.2 未来预测图
    # 获取最近time_step天的历史数据（用于绘图）
    last_n_days_scaled = test_data[-config.TIME_STEP:]
    last_n_days = data_processor.inverse_transform(last_n_days_scaled).flatten()
    current_date = data_processor.data['Date'].iloc[-1]

    plot_future_predictions(
        last_n_days,
        future_predictions,
        config.TIME_STEP,
        config.PRED_DAYS,
        current_date,
        save_path='future_predictions.png'
    )

    # 9. 保存
    print(f"\n{'='*70}")
    print("💾 保存结果")
    print(f"{'='*70}\n")

    torch.save(model.state_dict(), 'bitcoin_lstm_seq2seq_model.pth')
    print(f"✓ 模型已保存: bitcoin_lstm_seq2seq_model.pth")

    data_processor.save_scaler()

    print(f"\n{'='*70}")
    print("✅ 预测完成！")
    print(f"{'='*70}")
    print("\n📊 生成的文件:")
    print("  1. training_history_seq2seq.png  - 训练历史图")
    print("  2. test_predictions.png          - 测试集预测对比图 ⭐")
    print("  3. future_predictions.png        - 未来预测图 ⭐")
    print("  4. bitcoin_lstm_seq2seq_model.pth - 训练好的模型")
    print("  5. scaler.pkl                    - 数据归一化器")
    print("\n✨ 多对多模式优势：")
    print("  ✓ 一次性预测N天，无误差累积")
    print("  ✓ 预测更稳定可靠")
    print("  ✓ 适合短期预测（3-7天）")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()