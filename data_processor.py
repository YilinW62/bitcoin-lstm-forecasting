"""
Bitcoin price forecasting - Data processing module
Handles data loading, cleaning, scaling and dataset construction
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle


class BitcoinDataset(Dataset):
    """PyTorch dataset class"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataProcessor:
    """Data processor"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_data(self, file_path=None):
        """Load Excel data"""
        if file_path is None:
            file_path = self.config.DATA_FILE
            
        print(f"\n{'='*70}")
        print("Load data")
        print(f"{'='*70}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"File loaded: {file_path}")
            print(f"  - Rows: {df.shape[0]}")
            print(f"  - Columns: {df.shape[1]}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise
        
        # Rename columns (if needed)
        if 'Price' in df.columns and 'Close' not in df.columns:
            df = df.rename(columns={'Price': 'Close'})
        
        # Parse the date column
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"Date conversion warning: {str(e)}")
        
        # Convert the Close column to numeric values
        if df['Close'].dtype == 'object':
            df['Close'] = df['Close'].str.replace(',', '').astype(float)
        
        # Ensure data is ordered from oldest to newest
        if df['Date'].iloc[0] > df['Date'].iloc[-1]:
            df = df.sort_values('Date', ascending=True).reset_index(drop=True)
            print("Data sorted in chronological order")
        
        # Clean data
        df = df.dropna(subset=['Close'])
        df = df[df['Close'] > 0].reset_index(drop=True)
        
        self.data = df
        
        print(f"\nData cleaning completed")
        print(f"  - Valid observations: {len(df)} days")
        print(f"  - Date range: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
        print(f"\nPrice statistics:")
        print(f"  - Maximum price: ${df['Close'].max():,.2f}")
        print(f"  - Minimum price: ${df['Close'].min():,.2f}")
        print(f"  - Mean price: ${df['Close'].mean():,.2f}")
        print(f"  - Latest price: ${df['Close'].iloc[-1]:,.2f}")
        
        return df
    
    def normalize_data(self):
        """Scale data"""
        close_prices = self.data[['Close']].values
        self.scaled_data = self.scaler.fit_transform(close_prices)
        return self.scaled_data
    
    def create_sequences(self, data, time_step):
        """Create time-series sequences
        
        Args:
            data: Scaled data
            time_step: Lookback window
            
        Returns:
            X, y: Feature and target arrays
        """
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare training and test data"""
        print(f"\n{'='*70}")
        print("Prepare training data")
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
        X_train, y_train = self.create_sequences(train_data, self.config.TIME_STEP)
        X_test, y_test = self.create_sequences(test_data, self.config.TIME_STEP)
        
        # Reshape to LSTM input format: (samples, time_steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        print(f"\nData shapes:")
        print(f"  - X_train: {X_train.shape} (samples, time steps, features)")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, test_data
    
    def create_dataloaders(self, X_train, y_train, X_test, y_test):
        """Create PyTorch DataLoaders"""
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
        """Inverse scaling"""
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, path=None):
        """Save the scaler"""
        if path is None:
            path = self.config.SCALER_SAVE_PATH
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {path}")
    
    def load_scaler(self, path=None):
        """Load the scaler"""
        if path is None:
            path = self.config.SCALER_SAVE_PATH
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded: {path}")


if __name__ == "__main__":
    # Test the data processor
    from config import Config
    
    config = Config()
    processor = DataProcessor(config)
    
    # Load data
    df = processor.load_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, test_data = processor.prepare_data()
    
    # Create DataLoaders
    train_loader, test_loader = processor.create_dataloaders(X_train, y_train, X_test, y_test)
    
    print(f"\nData processing check completed!")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")