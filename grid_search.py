"""
Bitcoin price forecasting - Grid search for model parameters
Evaluate different parameter combinations automatically, select the best configuration
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
    """Data processor for multi-step forecasting"""

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
    """Grid-search trainer"""
    
    def __init__(self, device='cuda'):
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        print(f"Device: {self.device}")
    
    def train_model(self, model, train_loader, val_loader, config):
        """Train a single model"""
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.EPOCHS):
            # Training
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
            
            # Validation
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
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                break
        
        return best_val_loss
    
    def evaluate(self, model, data_loader, data_processor, pred_days):
        """Evaluate the model"""
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


def grid_search(param_grid, data_file='Bitcoin_Historical_Data_10years.xlsx', device='cuda'):
    """
    Run the grid search
    
    Args:
        param_grid: Parameter grid dictionary
        data_file: Data file path
        device: Compute device
    """
    print("="*70)
    print("Bitcoin price forecasting - Grid search for model parameters")
    print("="*70)
    
    # Generate all parameter combinations
    from itertools import product
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    total_combinations = len(combinations)
    print(f"\nParameter grid:")
    for key, vals in param_grid.items():
        print(f"  - {key}: {vals}")
    print(f"\nTotal combinations to evaluate: {total_combinations} parameter combinations")
    
    # Store results
    results = []
    
    # Initialize the trainer
    trainer = GridSearchTrainer(device=device)
    
    # Iterate over all combinations
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        print(f"\n{'='*70}")
        print(f"Trying combination [{idx}/{total_combinations}]")
        print(f"{'='*70}")
        print("Parameter configuration:")
        for key, val in params.items():
            print(f"  - {key}: {val}")
        
        try:
            # Create the configuration
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
            
            # Data processing
            data_processor = Seq2SeqDataProcessor(config)
            data_processor.load_data()
            X_train, y_train, X_test, y_test, test_data = data_processor.prepare_data_seq2seq()
            
            # Split the validation set
            val_size = int(len(X_train) * config.VAL_SPLIT)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train_split, y_train_split = X_train[:-val_size], y_train[:-val_size]
            
            train_loader, _ = data_processor.create_dataloaders(X_train_split, y_train_split, X_test, y_test)
            val_loader, test_loader = data_processor.create_dataloaders(X_val, y_val, X_test, y_test)
            
            # Create the model
            model = create_seq2seq_model(config, model_type='direct')
            model = model.to(trainer.device)
            
            # Training
            print("\nStart training...")
            val_loss = trainer.train_model(model, train_loader, val_loader, config)
            
            # Evaluation
            print("Evaluate the model...")
            test_rmse, test_mae, test_r2 = trainer.evaluate(
                model, test_loader, data_processor, config.PRED_DAYS
            )
            
            # Record results
            result = {
                'idx': idx,
                'params': params,
                'val_loss': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            results.append(result)
            
            print(f"\nCompleted!")
            print(f"  - Validation loss: {val_loss:.6f}")
            print(f"  - Test RMSE: ${test_rmse:,.2f}")
            print(f"  - Test MAE:  ${test_mae:,.2f}")
            print(f"  - Test R²:   {test_r2:.4f}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            result = {
                'idx': idx,
                'params': params,
                'error': str(e)
            }
            results.append(result)
    
    return results


def save_results(results, save_path='grid_search_results.json'):
    """Save grid-search results"""
    # Convert to serializable values
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
    
    print(f"Results saved: {save_path}")


def analyze_results(results):
    """Analyze results and select the best parameters"""
    print(f"\n{'='*70}")
    print("Grid-search results analysis")
    print(f"{'='*70}")
    
    # Filter out failed runs
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No successful training results!")
        return None
    
    print(f"\nSuccessfully completed: {len(valid_results)}/{len(results)} combinations")
    
    # Sort by different metrics
    by_r2 = sorted(valid_results, key=lambda x: x['test_r2'], reverse=True)
    by_rmse = sorted(valid_results, key=lambda x: x['test_rmse'])
    by_mae = sorted(valid_results, key=lambda x: x['test_mae'])
    
    # Best R²
    print(f"\nBest R² model (R² = {by_r2[0]['test_r2']:.4f}):")
    print("Parameter configuration:")
    for key, val in by_r2[0]['params'].items():
        print(f"  - {key}: {val}")
    print(f"Performance metrics:")
    print(f"  - RMSE: ${by_r2[0]['test_rmse']:,.2f}")
    print(f"  - MAE:  ${by_r2[0]['test_mae']:,.2f}")
    print(f"  - R²:   {by_r2[0]['test_r2']:.4f}")
    
    # Minimum RMSE
    print(f"\nMinimum RMSE model (RMSE = ${by_rmse[0]['test_rmse']:,.2f}):")
    print("Parameter configuration:")
    for key, val in by_rmse[0]['params'].items():
        print(f"  - {key}: {val}")
    print(f"Performance metrics:")
    print(f"  - RMSE: ${by_rmse[0]['test_rmse']:,.2f}")
    print(f"  - MAE:  ${by_rmse[0]['test_mae']:,.2f}")
    print(f"  - R²:   {by_rmse[0]['test_r2']:.4f}")
    
    # Top 5 results
    print(f"\nTop 5 models (sorted by R²):")
    print(f"{'Rank':<6} {'R²':<8} {'RMSE':<12} {'MAE':<12} {'Parameters'}")
    print("-" * 70)
    for i, r in enumerate(by_r2[:5], 1):
        param_str = f"TS={r['params']['time_step']}, LU={r['params']['lstm_units']}, LR={r['params']['learning_rate']}"
        print(f"{i:<6} {r['test_r2']:<8.4f} ${r['test_rmse']:<11,.0f} ${r['test_mae']:<11,.0f} {param_str}")
    
    return by_r2[0]


def plot_grid_search_results(results, save_path='grid_search_visualization.png'):
    """Visualize grid-search results"""
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No data available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    indices = [r['idx'] for r in valid_results]
    r2_scores = [r['test_r2'] for r in valid_results]
    rmse_scores = [r['test_rmse'] for r in valid_results]
    mae_scores = [r['test_mae'] for r in valid_results]
    
    # 1. R² score
    axes[0, 0].bar(indices, r2_scores, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=max(r2_scores), color='red', linestyle='--', linewidth=2, label=f'Best: {max(r2_scores):.4f}')
    axes[0, 0].set_title('R² score comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Parameter combination index', fontsize=12)
    axes[0, 0].set_ylabel('R² score', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. RMSE
    axes[0, 1].bar(indices, rmse_scores, color='coral', alpha=0.7)
    axes[0, 1].axhline(y=min(rmse_scores), color='green', linestyle='--', linewidth=2, label=f'Best: ${min(rmse_scores):,.0f}')
    axes[0, 1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Parameter combination index', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (USD)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. MAE
    axes[1, 0].bar(indices, mae_scores, color='lightgreen', alpha=0.7)
    axes[1, 0].axhline(y=min(mae_scores), color='darkgreen', linestyle='--', linewidth=2, label=f'Best: ${min(mae_scores):,.0f}')
    axes[1, 0].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Parameter combination index', fontsize=12)
    axes[1, 0].set_ylabel('MAE (USD)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. RMSE vs R² scatter plot
    axes[1, 1].scatter(r2_scores, rmse_scores, s=100, alpha=0.6, c=indices, cmap='viridis')
    best_idx = r2_scores.index(max(r2_scores))
    axes[1, 1].scatter([r2_scores[best_idx]], [rmse_scores[best_idx]], 
                       s=300, marker='*', color='red', edgecolors='black', linewidths=2, 
                       label='Best model', zorder=5)
    axes[1, 1].set_title('RMSE vs R² Relationship', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('R² score', fontsize=12)
    axes[1, 1].set_ylabel('RMSE (USD)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Bitcoin price forecasting - Grid search')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_file', type=str, default='Bitcoin_Historical_Data_10years.xlsx')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer parameter combinations)')
    
    args = parser.parse_args()
    
    # Define the parameter grid
    if args.quick:
        # Quick mode: fewer parameter combinations
        param_grid = {
            'time_step': [30, 60],
            'pred_days': [3, 7],
            'lstm_units': [50, 100],
            'learning_rate': [0.0001, 0.001],
            'epochs': [30]
        }
    else:
        # Full mode: more parameter combinations
        param_grid = {
            'time_step': [30, 60, 90],
            'pred_days': [3, 7, 14],
            'lstm_units': [50, 100, 150],
            'learning_rate': [0.00005, 0.0001, 0.0005, 0.001],
            'dropout': [0.1, 0.2, 0.3],
            'epochs': [50]
        }
    
    # Run the grid search
    start_time = datetime.now()
    results = grid_search(param_grid, data_file=args.data_file, device=args.device)
    end_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"Total elapsed time: {end_time - start_time}")
    print(f"{'='*70}")
    
    # Save results
    save_results(results)
    
    # Analyze results
    best_model = analyze_results(results)
    
    # Visualization
    plot_grid_search_results(results)
    
    # Save the selected parameter configuration
    if best_model:
        best_config = {
            'best_parameters': best_model['params'],
            'performance_metrics': {
                'RMSE': float(best_model['test_rmse']),
                'MAE': float(best_model['test_mae']),
                'R²': float(best_model['test_r2'])
            },
            'search_duration': str(end_time - start_time),
            'search_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('best_config.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        print(f"\nBest configuration saved: best_config.json")
        
        # Print the run command
        print(f"\n{'='*70}")
        print("Command to train with the selected parameters:")
        print(f"{'='*70}")
        params = best_model['params']
        cmd = f"python train.py --time_step {params['time_step']} --pred_days {params['pred_days']} --lstm_units {params['lstm_units']} --learning_rate {params['learning_rate']}"
        if 'dropout' in params:
            # Note: Add a dropout argument to train.py to support this option.
            pass
        print(cmd)
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
