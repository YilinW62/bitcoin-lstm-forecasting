"""
Bitcoin price forecasting - Configuration file
These defaults can be overridden with command-line arguments
"""

class Config:
    # Data file path
    DATA_FILE = 'Bitcoin_Historical_Data_10years.xlsx'
    
    # Model parameters
    TIME_STEP = 30          # Lookback window (number of historical days used for forecasting)
    TRAIN_RATIO = 0.85      # Training fraction
    
    # Training parameters
    EPOCHS = 50             # Training epochs
    BATCH_SIZE = 32         # Batch size
    LEARNING_RATE = 0.001   # Learning rate
    
    # LSTM network architecture
    LSTM_UNITS_1 = 50       # First LSTM layer units
    LSTM_UNITS_2 = 50       # Second LSTM layer units (optional)
    DROPOUT = 0.2           # Dropout rate
    
    # Forecast parameters
    PRED_DAYS = 3           # Number of days to forecast
    
    # Training optimization
    EARLY_STOP_PATIENCE = 15  # Early-stopping patience
    VAL_SPLIT = 0.1           # Validation fraction
    
    # Device settings
    DEVICE = 'cuda'  # 'cuda' or 'cpu', automatically detected
    
    # Output path
    MODEL_SAVE_PATH = 'bitcoin_lstm_model.pth'
    SCALER_SAVE_PATH = 'scaler.pkl'
    RESULTS_CSV = 'future_predictions.csv'
    
    # Random seed
    RANDOM_SEED = 42
    
    def __init__(self, **kwargs):
        """Override default configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        """Display the configuration"""
        config_str = "=" * 70 + "\n"
        config_str += "Configuration\n"
        config_str += "=" * 70 + "\n"
        for key in dir(self):
            if not key.startswith('_') and key.isupper():
                config_str += f"{key:25s}: {getattr(self, key)}\n"
        config_str += "=" * 70 + "\n"
        return config_str