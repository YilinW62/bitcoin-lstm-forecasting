"""
Bitcoin price forecasting - Multi-step training script
Use Direct LSTM: predict the next N days
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
    """Data processor for multi-step forecasting"""

    def create_sequences_seq2seq(self, data, time_step, pred_days):
        """
        Create multi-step data sequences

        Args:
            data: Scaled data
            time_step: Input lookback window
            pred_days: Forecast horizon (output length)

        Returns:
            X: [samples, time_step, 1]
            y: [samples, pred_days]  each sample has pred_days target values
        """
        X, y = [], []
        for i in range(len(data) - time_step - pred_days + 1):
            X.append(data[i:(i + time_step), 0])
            # y contains the values for the next pred_days days
            y.append(data[(i + time_step):(i + time_step + pred_days), 0])

        return np.array(X), np.array(y)

    def prepare_data_seq2seq(self):
        """Prepare multi-step training data"""
        print(f"\n{'='*70}")
        print("Prepare multi-step training data")
        print(f"{'='*70}")

        # Scaling
        scaled_data = self.normalize_data()

        # Split training and test sets
        training_size = int(len(scaled_data) * self.config.TRAIN_RATIO)
        train_data = scaled_data[0:training_size, :]
        test_data = scaled_data[training_size:, :]

        print(f"\nDataset split:")
        print(f"  - Training set: {training_size} days ({self.config.TRAIN_RATIO*100:.0f}%)")
        print(f"  - Test set: {len(test_data)} days ({(1-self.config.TRAIN_RATIO)*100:.0f}%)")

        # Create sequences
        X_train, y_train = self.create_sequences_seq2seq(
            train_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )
        X_test, y_test = self.create_sequences_seq2seq(
            test_data, self.config.TIME_STEP, self.config.PRED_DAYS
        )

        # Reshape to LSTM input format
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        # y does not require reshaping, keep [samples, pred_days]

        print(f"\nData shapes:")
        print(f"  - X_train: {X_train.shape} (samples, time steps, features)")
        print(f"  - y_train: {y_train.shape} (samples, Forecast horizon)  multi-step output")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")

        return X_train, y_train, X_test, y_test, test_data


class Seq2SeqTrainer:
    """Trainer for multi-step forecasting"""

    def __init__(self, config):
        self.config = config

        if config.DEVICE == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        print(f"\nDevice: {self.device}")

        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            outputs = model(X_batch)  # [batch, pred_days]
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        """Validation"""
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
        """Run the full training loop"""
        print(f"\n{'='*70}")
        print(f"Start training the multi-step model")
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
                print(f"\nEarly stopping triggered! Validation loss: {patience_counter} epochs without improvement")
                model.load_state_dict(torch.load('best_model_temp.pth'))
                break

        print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

        return train_losses, val_losses

    def evaluate(self, model, data_loader, data_processor, pred_days):
        """Evaluate the model"""
        model.eval()
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)  # [batch, pred_days]

                all_predictions.extend(outputs.cpu().numpy())
                all_actuals.extend(y_batch.numpy())

        # Convert to NumPy arrays
        all_predictions = np.array(all_predictions)  # [samples, pred_days]
        all_actuals = np.array(all_actuals)  # [samples, pred_days]

        # Inverse scaling (requires reshaping)
        pred_reshaped = all_predictions.reshape(-1, 1)
        actual_reshaped = all_actuals.reshape(-1, 1)

        pred_inverse = data_processor.inverse_transform(pred_reshaped)
        actual_inverse = data_processor.inverse_transform(actual_reshaped)

        # Compute metrics
        rmse = math.sqrt(mean_squared_error(actual_inverse, pred_inverse))
        mae = mean_absolute_error(actual_inverse, pred_inverse)
        r2 = r2_score(actual_inverse, pred_inverse)

        # Restore the original shape
        pred_inverse = pred_inverse.reshape(-1, pred_days)
        actual_inverse = actual_inverse.reshape(-1, pred_days)

        return pred_inverse, actual_inverse, rmse, mae, r2

    def predict_future(self, model, test_data, data_processor):
        """Forecast the next N days (output in one pass)"""
        print(f"\n{'='*70}")
        print(f"Forecast the next {self.config.PRED_DAYS} days (output in one pass)")
        print(f"{'='*70}")

        model.eval()

        # Use the last TIME_STEP days of observed data
        last_sequence = test_data[-self.config.TIME_STEP:].reshape(1, self.config.TIME_STEP, 1)
        last_sequence = torch.FloatTensor(last_sequence).to(self.device)

        # predict in one pass
        with torch.no_grad():
            predictions = model(last_sequence)  # [1, pred_days]

        # Inverse scaling
        predictions = predictions.cpu().numpy().reshape(-1, 1)
        predictions = data_processor.inverse_transform(predictions)

        print(f"Forecast completed (no recursive forecast feedback!)")

        return predictions


def plot_training_history(train_losses, val_losses, save_path='training_history_seq2seq.png'):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training loss', linewidth=2, color='blue')
    plt.plot(val_losses, label='Validation loss', linewidth=2, color='orange')
    plt.title('Multi-step model training history', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved: {save_path}")
    plt.close()


def plot_test_predictions(test_actual, test_pred, test_rmse, test_mae, test_r2, pred_days, save_path='test_predictions.png'):
    """
    Plot test-set predictions against observations

    Args:
        test_actual: Actual price [samples, pred_days]
        test_pred: Predicted price [samples, pred_days]
        test_rmse: RMSEmetric
        test_mae: MAEmetric
        test_r2: R²metric
        pred_days: Forecast horizon
        save_path: Output path
    """
    # Flatten data for plotting
    actual_flat = test_actual.flatten()
    pred_flat = test_pred.flatten()

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual and predicted prices
    ax.plot(actual_flat, label='Actual price', linewidth=2, alpha=0.8, color='blue')
    ax.plot(pred_flat, label='Predicted price', linewidth=2, alpha=0.7, color='orange')

    ax.set_title('Test-set prediction comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Time (days)', fontsize=14)
    ax.set_ylabel('Price (USD)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add a performance-metrics text box
    textstr = f'RMSE: ${test_rmse:,.2f}\nMAE: ${test_mae:,.2f}\nR²: {test_r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Test prediction plot saved: {save_path}")
    plt.close()


def plot_future_predictions(historical_data, future_predictions, time_step, pred_days,
                           current_date, save_path='future_predictions.png'):
    """
    Plot future forecasts

    Args:
        historical_data: Last time_step days of historical data [time_step]
        future_predictions: Future forecasts [pred_days, 1]
        time_step: Lookback window
        pred_days: Forecast horizon
        current_date: Latest observed date
        save_path: Output path
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # X axis: Historical and forecast data
    hist_days = np.arange(1, time_step + 1)
    pred_days_x = np.arange(time_step + 1, time_step + pred_days + 1)

    # Plot historical data
    ax.plot(hist_days, historical_data,
            label=f'Last {time_step} days of observed prices',
            marker='o', linewidth=2.5, markersize=5, color='blue')

    # Plot forecast data
    future_flat = future_predictions.flatten()
    ax.plot(pred_days_x, future_flat,
            label=f'Next {pred_days} day forecast prices',
            marker='s', linewidth=3, markersize=8, color='red', linestyle='--')

    # Annotate each predicted value
    for i, (day, price) in enumerate(zip(pred_days_x, future_flat)):
        ax.annotate(f'${price:,.0f}',
                    xy=(day, price),
                    xytext=(8, 10),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Mark the forecast origin
    ax.axvline(x=time_step, color='green', linestyle=':', linewidth=2.5,
               label='Forecast origin', alpha=0.8)

    ax.set_title(f'Bitcoin price: Last {time_step} days vs Next {pred_days} day forecast',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Days', fontsize=13)
    ax.set_ylabel('Price (USD)', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Future forecast plot saved: {save_path}")
    plt.close()


def print_predictions(data_processor, future_predictions, pred_days):
    """Print forecasts"""
    current_price = data_processor.data['Close'].iloc[-1]
    current_date = data_processor.data['Date'].iloc[-1]

    print(f"\n{'='*70}")
    print(f"Next {pred_days} day price forecast (Multi-step forecasting - no recursive forecast feedback)")
    print(f"{'='*70}")
    print(f"\nLatest price: ${current_price:,.2f} ({current_date.date()})\n")

    for i in range(pred_days):
        pred_price = future_predictions[i][0]
        change = pred_price - current_price
        change_pct = (change / current_price) * 100

        future_date = current_date + pd.Timedelta(days=i+1)

        print(f"Day {i+1} ({future_date.date()}):")
        print(f"Predicted price: ${pred_price:,.2f}")
        print(f"Price change: ${change:+,.2f} ({change_pct:+.2f}%)")
        print()

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Bitcoin price forecasting - Multi-step forecasting')
    parser.add_argument('--time_step', type=int, default=60,
                        help='Lookback window')
    parser.add_argument('--pred_days', type=int, default=3,
                        help='Forecast horizon')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lstm_units', type=int, default=50,
                        help='LSTM units')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--data_file', type=str, default='Bitcoin_Historical_Data_10years.xlsx',
                        help='Data file path')

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
    print("Bitcoin price forecasting system - Multi-step forecasting (no recursive forecast feedback)")
    print("="*70)
    print(config)

    # 1. Data processing
    data_processor = Seq2SeqDataProcessor(config)
    data_processor.load_data()
    X_train, y_train, X_test, y_test, test_data = data_processor.prepare_data_seq2seq()

    # Split the validation set
    val_size = int(len(X_train) * config.VAL_SPLIT)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    train_loader, _ = data_processor.create_dataloaders(X_train, y_train, X_test, y_test)
    val_loader, test_loader = data_processor.create_dataloaders(X_val, y_val, X_test, y_test)

    print(f"\nFinal dataset split:")
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples")
    print(f"  - Test set: {len(X_test)} samples")

    # 2. Create the model
    print(f"\n{'='*70}")
    print(f"Build the multi-step model")
    print(f"{'='*70}")

    model = create_seq2seq_model(config, model_type='direct')

    trainer = Seq2SeqTrainer(config)
    model = model.to(trainer.device)

    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # 3. Training
    train_losses, val_losses = trainer.train(model, train_loader, val_loader)

    # 4. Plot training history
    plot_training_history(train_losses, val_losses)

    # 5. Evaluation
    print(f"\n{'='*70}")
    print("Evaluate model performance")
    print(f"{'='*70}")

    print("\n[Test-set performance]")
    test_pred, test_actual, test_rmse, test_mae, test_r2 = trainer.evaluate(
        model, test_loader, data_processor, config.PRED_DAYS
    )
    print(f"  - RMSE: ${test_rmse:,.2f}")
    print(f"  - MAE:  ${test_mae:,.2f}")
    print(f"  - R²:   {test_r2:.4f}")

    # 6. Forecast the next
    future_predictions = trainer.predict_future(model, test_data, data_processor)

    # 7. Print results
    print_predictions(data_processor, future_predictions, config.PRED_DAYS)

    # 8. Generate visualizations
    print(f"\n{'='*70}")
    print("Generate visualizations")
    print(f"{'='*70}\n")

    # 8.1 Test-set prediction comparison plot
    plot_test_predictions(
        test_actual,
        test_pred,
        test_rmse,
        test_mae,
        test_r2,
        config.PRED_DAYS,
        save_path='test_predictions.png'
    )

    # 8.2 Future forecast plot
    # Get the last time_step days of historical data (for plotting)
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

    # 9. Save
    print(f"\n{'='*70}")
    print("Save results")
    print(f"{'='*70}\n")

    torch.save(model.state_dict(), 'bitcoin_lstm_seq2seq_model.pth')
    print(f"Model saved: bitcoin_lstm_seq2seq_model.pth")

    data_processor.save_scaler()

    print(f"\n{'='*70}")
    print("Forecast completed!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  1. training_history_seq2seq.png  - Training history plot")
    print("  2. test_predictions.png          - Test-set prediction comparison plot ")
    print("  3. future_predictions.png        - Future forecast plot ")
    print("  4. bitcoin_lstm_seq2seq_model.pth - Trained model")
    print("  5. scaler.pkl                    - Data scaler")
    print("\nMulti-step forecasting features: ")
    print("  Predict N days in one pass, no recursive forecast feedback")
    print("  Forecast stability and reliability require evaluation")
    print("  Designed for short-horizon forecasts (3-7 days)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()