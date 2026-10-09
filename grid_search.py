"""
比特币价格预测 - 网格搜索最佳参数
自动尝试不同的参数组合，找到最优配置
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
from datetime import datetime
import json
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
        X, y = [], []
        for i in range(len(data) - time_step - pred_days + 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[(i + time_step):(i + time_step + pred_days), 0])
        return np.array(X), np.array(y)

    def prepare_data_seq2seq(self):
        scaled_data = self.normalize_data()
        training_size = int(len(scaled_data) * self.config.TRAIN_RATIO)
        train_data = scaled_data[0:training_size, :]
        test_data = scaled_data[training_size:, :]
        
        X_train, y_train = self.create_sequences_seq2seq(
            train_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )
        X_test, y_test = self.create_sequences_seq2seq(
            test_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, y_train, X_test, y_test, test_data


class GridSearchTrainer:
    """网格搜索训练器"""
    
    def __init__(self, device='cuda'):
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"🖥️  使用设备: {self.device}")
    
    def train_model(self, model, train_loader, val_loader, config):
        """训练单个模型"""
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            # 训练
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                break
        
        return best_val_loss
    
    def evaluate(self, model, data_loader, data_processor, pred_days):
        """评估模型"""
        model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_actuals.extend(y_batch.numpy())
        
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        pred_reshaped = all_predictions.reshape(-1, 1)
        actual_reshaped = all_actuals.reshape(-1, 1)
        
        pred_inverse = data_processor.inverse_transform(pred_reshaped)
        actual_inverse = data_processor.inverse_transform(actual_reshaped)
        
        rmse = math.sqrt(mean_squared_error(actual_inverse, pred_inverse))
        mae = mean_absolute_error(actual_inverse, pred_inverse)
        r2 = r2_score(actual_inverse, pred_inverse)
        
        return rmse, mae, r2


def grid_search(param_grid, data_file='Bitcoin_Historical_Data_10年.xlsx', device='cuda'):
    """
    网格搜索主函数
    
    Args:
        param_grid: 参数网格字典
        data_file: 数据文件路径
        device: 运行设备
    """
    print("="*70)
    print("🔍 比特币价格预测 - 网格搜索最佳参数")
    print("="*70)
    
    # 生成所有参数组合
    from itertools import product
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    total_combinations = len(combinations)
    print(f"\n📊 参数网格:")
    for key, vals in param_grid.items():
        print(f"  - {key}: {vals}")
    print(f"\n总共需要尝试 {total_combinations} 种参数组合")
    
    # 存储结果
    results = []
    
    # 初始化训练器
    trainer = GridSearchTrainer(device=device)
    
    # 遍历所有组合
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        print(f"\n{'='*70}")
        print(f"尝试组合 [{idx}/{total_combinations}]")
        print(f"{'='*70}")
        print("参数配置:")
        for key, val in params.items():
            print(f"  - {key}: {val}")
        
        try:
            # 创建配置
            config = Config(
                TIME_STEP=params['time_step'],
                PRED_DAYS=params['pred_days'],
                EPOCHS=params['epochs'],
                BATCH_SIZE=params.get('batch_size', 32),
                LSTM_UNITS_1=params['lstm_units'],
                LEARNING_RATE=params['learning_rate'],
                DROPOUT=params.get('dropout', 0.2),
                DEVICE=device,
                DATA_FILE=data_file
            )
            
            # 数据处理
            data_processor = Seq2SeqDataProcessor(config)
            data_processor.load_data()
            X_train, y_train, X_test, y_test, test_data = data_processor.prepare_data_seq2seq()
            
            # 划分验证集
            val_size = int(len(X_train) * config.VAL_SPLIT)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train_split, y_train_split = X_train[:-val_size], y_train[:-val_size]
            
            train_loader, _ = data_processor.create_dataloaders(X_train_split, y_train_split, X_test, y_test)
            val_loader, test_loader = data_processor.create_dataloaders(X_val, y_val, X_test, y_test)
            
            # 创建模型
            model = create_seq2seq_model(config, model_type='direct')
            model = model.to(trainer.device)
            
            # 训练
            print("\n🚀 开始训练...")
            val_loss = trainer.train_model(model, train_loader, val_loader, config)
            
            # 评估
            print("📊 评估模型...")
            test_rmse, test_mae, test_r2 = trainer.evaluate(
                model, test_loader, data_processor, config.PRED_DAYS
            )
            
            # 记录结果
            result = {
                'idx': idx,
                'params': params,
                'val_loss': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            results.append(result)
            
            print(f"\n✓ 完成！")
            print(f"  - 验证损失: {val_loss:.6f}")
            print(f"  - 测试RMSE: ${test_rmse:,.2f}")
            print(f"  - 测试MAE:  ${test_mae:,.2f}")
            print(f"  - 测试R²:   {test_r2:.4f}")
            
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            result = {
                'idx': idx,
                'params': params,
                'error': str(e)
            }
            results.append(result)
    
    return results


def save_results(results, save_path='grid_search_results.json'):
    """保存网格搜索结果"""
    # 转换为可序列化格式
    results_serializable = []
    for r in results:
        if 'error' not in r:
            results_serializable.append({
                'idx': r['idx'],
                'params': r['params'],
                'val_loss': float(r['val_loss']),
                'test_rmse': float(r['test_rmse']),
                'test_mae': float(r['test_mae']),
                'test_r2': float(r['test_r2'])
            })
        else:
            results_serializable.append({
                'idx': r['idx'],
                'params': r['params'],
                'error': r['error']
            })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 结果已保存: {save_path}")


def analyze_results(results):
    """分析结果并找到最佳参数"""
    print(f"\n{'='*70}")
    print("📊 网格搜索结果分析")
    print(f"{'='*70}")
    
    # 过滤掉错误的结果
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ 没有成功的训练结果！")
        return None
    
    print(f"\n成功完成: {len(valid_results)}/{len(results)} 个组合")
    
    # 按不同指标排序
    by_r2 = sorted(valid_results, key=lambda x: x['test_r2'], reverse=True)
    by_rmse = sorted(valid_results, key=lambda x: x['test_rmse'])
    by_mae = sorted(valid_results, key=lambda x: x['test_mae'])
    
    # 最佳R²
    print(f"\n🏆 最佳R²模型 (R² = {by_r2[0]['test_r2']:.4f}):")
    print("参数配置:")
    for key, val in by_r2[0]['params'].items():
        print(f"  - {key}: {val}")
    print(f"性能指标:")
    print(f"  - RMSE: ${by_r2[0]['test_rmse']:,.2f}")
    print(f"  - MAE:  ${by_r2[0]['test_mae']:,.2f}")
    print(f"  - R²:   {by_r2[0]['test_r2']:.4f}")
    
    # 最小RMSE
    print(f"\n🎯 最小RMSE模型 (RMSE = ${by_rmse[0]['test_rmse']:,.2f}):")
    print("参数配置:")
    for key, val in by_rmse[0]['params'].items():
        print(f"  - {key}: {val}")
    print(f"性能指标:")
    print(f"  - RMSE: ${by_rmse[0]['test_rmse']:,.2f}")
    print(f"  - MAE:  ${by_rmse[0]['test_mae']:,.2f}")
    print(f"  - R²:   {by_rmse[0]['test_r2']:.4f}")
    
    # Top 5结果
    print(f"\n📋 Top 5 模型 (按R²排序):")
    print(f"{'排名':<6} {'R²':<8} {'RMSE':<12} {'MAE':<12} {'参数'}")
    print("-" * 70)
    for i, r in enumerate(by_r2[:5], 1):
        param_str = f"TS={r['params']['time_step']}, LU={r['params']['lstm_units']}, LR={r['params']['learning_rate']}"
        print(f"{i:<6} {r['test_r2']:<8.4f} ${r['test_rmse']:<11,.0f} ${r['test_mae']:<11,.0f} {param_str}")
    
    return by_r2[0]


def plot_grid_search_results(results, save_path='grid_search_visualization.png'):
    """可视化网格搜索结果"""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ 没有可视化数据")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取数据
    indices = [r['idx'] for r in valid_results]
    r2_scores = [r['test_r2'] for r in valid_results]
    rmse_scores = [r['test_rmse'] for r in valid_results]
    mae_scores = [r['test_mae'] for r in valid_results]
    
    # 1. R²分数
    axes[0, 0].bar(indices, r2_scores, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=max(r2_scores), color='red', linestyle='--', linewidth=2, label=f'最佳: {max(r2_scores):.4f}')
    axes[0, 0].set_title('R² 分数对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('参数组合编号', fontsize=12)
    axes[0, 0].set_ylabel('R² 分数', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. RMSE
    axes[0, 1].bar(indices, rmse_scores, color='coral', alpha=0.7)
    axes[0, 1].axhline(y=min(rmse_scores), color='green', linestyle='--', linewidth=2, label=f'最佳: ${min(rmse_scores):,.0f}')
    axes[0, 1].set_title('RMSE 对比', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('参数组合编号', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (USD)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. MAE
    axes[1, 0].bar(indices, mae_scores, color='lightgreen', alpha=0.7)
    axes[1, 0].axhline(y=min(mae_scores), color='darkgreen', linestyle='--', linewidth=2, label=f'最佳: ${min(mae_scores):,.0f}')
    axes[1, 0].set_title('MAE 对比', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('参数组合编号', fontsize=12)
    axes[1, 0].set_ylabel('MAE (USD)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. RMSE vs R²散点图
    axes[1, 1].scatter(r2_scores, rmse_scores, s=100, alpha=0.6, c=indices, cmap='viridis')
    best_idx = r2_scores.index(max(r2_scores))
    axes[1, 1].scatter([r2_scores[best_idx]], [rmse_scores[best_idx]], 
                       s=300, marker='*', color='red', edgecolors='black', linewidths=2, 
                       label='最佳模型', zorder=5)
    axes[1, 1].set_title('RMSE vs R² 关系', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('R² 分数', fontsize=12)
    axes[1, 1].set_ylabel('RMSE (USD)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='比特币价格预测 - 网格搜索')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_file', type=str, default='Bitcoin_Historical_Data_10年.xlsx')
    parser.add_argument('--quick', action='store_true', help='快速模式（较少的参数组合）')
    
    args = parser.parse_args()
    
    # 定义参数网格
    if args.quick:
        # 快速模式：较少的参数组合
        param_grid = {
            'time_step': [30, 60],
            'pred_days': [3, 7],
            'lstm_units': [50, 100],
            'learning_rate': [0.0001, 0.001],
            'epochs': [30]
        }
    else:
        # 完整模式：更多参数组合
        param_grid = {
            'time_step': [30, 60, 90],
            'pred_days': [3, 7, 14],
            'lstm_units': [50, 100, 150],
            'learning_rate': [0.00005, 0.0001, 0.0005, 0.001],
            'dropout': [0.1, 0.2, 0.3],
            'epochs': [50]
        }
    
    # 运行网格搜索
    start_time = datetime.now()
    results = grid_search(param_grid, data_file=args.data_file, device=args.device)
    end_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"⏱️  总耗时: {end_time - start_time}")
    print(f"{'='*70}")
    
    # 保存结果
    save_results(results)
    
    # 分析结果
    best_model = analyze_results(results)
    
    # 可视化
    plot_grid_search_results(results)
    
    # 保存最佳参数配置
    if best_model:
        best_config = {
            '最佳参数配置': best_model['params'],
            '性能指标': {
                'RMSE': float(best_model['test_rmse']),
                'MAE': float(best_model['test_mae']),
                'R²': float(best_model['test_r2'])
            },
            '搜索时间': str(end_time - start_time),
            '搜索日期': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('best_config.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 最佳配置已保存: best_config.json")
        
        # 打印运行命令
        print(f"\n{'='*70}")
        print("🚀 使用最佳参数训练模型的命令:")
        print(f"{'='*70}")
        params = best_model['params']
        cmd = f"python train.py --time_step {params['time_step']} --pred_days {params['pred_days']} --lstm_units {params['lstm_units']} --learning_rate {params['learning_rate']}"
        if 'dropout' in params:
            # Note: 需要在train.py中添加dropout参数
            pass
        print(cmd)
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
