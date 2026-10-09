"""
Bitcoin price forecasting - multi-step (Seq2Seq) LSTM model
Multi-step time-series forecasting
"""

import torch
import torch.nn as nn


class BitcoinLSTM_Seq2Seq(nn.Module):
    """
    Multi-step LSTM model - Directly forecast the next N days
    Input: [batch, time_step, 1]
    Output: [batch, pred_days, 1]  output multi-day forecasts in one pass
    """

    def __init__(self,
                 input_size=1,
                 hidden_size=50,
                 num_layers=1,
                 dropout=0.2,
                 pred_days=3):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pred_days: Forecast horizon in days (output sequence length)
        """
        super(BitcoinLSTM_Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_days = pred_days

        # Encoder LSTM (encoder)
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder LSTM (decoder)
        self.decoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass - Seq2Seq architecture

        Args:
            x: Input tensor [batch_size, time_step, 1]

        Returns:
            predictions: [batch_size, pred_days, 1]
        """
        batch_size = x.size(0)

        # 1. Encoder: Process the input sequence
        _, (hidden, cell) = self.encoder_lstm(x)
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]

        # 2. Decoder: Generate the forecast sequence
        # The initial input is the value at the last time step
        decoder_input = x[:, -1:, :]  # [batch_size, 1, 1]

        predictions = []

        for _ in range(self.pred_days):
            # Decode one step
            decoder_output, (hidden, cell) = self.decoder_lstm(
                decoder_input, (hidden, cell)
            )
            # decoder_output: [batch_size, 1, hidden_size]

            # Predict the current step
            pred = self.fc(decoder_output)  # [batch_size, 1, 1]
            predictions.append(pred)

            # Feed the current prediction into the next decoder step
            decoder_input = pred

        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # [batch_size, pred_days, 1]

        return predictions

    def predict(self, x):
        """
        Prediction mode (equivalent to forward, with a more explicit name)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class BitcoinLSTM_Direct(nn.Module):
    """
    Direct multi-output LSTM model - a simpler multi-step implementation
    Input: [batch, time_step, 1]
    Output: [batch, pred_days]  Directly output N values
    """

    def __init__(self,
                 input_size=1,
                 hidden_size=50,
                 num_layers=1,
                 dropout=0.2,
                 pred_days=3):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pred_days: Forecast horizon in days
        """
        super(BitcoinLSTM_Direct, self).__init__()

        self.hidden_size = hidden_size
        self.pred_days = pred_days

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer: Directly output pred_days values
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, pred_days)  #  Output size = forecast horizon

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, time_step, 1]

        Returns:
            predictions: [batch_size, pred_days]
        """
        # Process the sequence with the LSTM
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        out = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc1(out)  # [batch_size, 50]
        out = self.relu(out)
        out = self.fc2(out)  # [batch_size, pred_days]

        return out

    def predict(self, x):
        """Prediction mode"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)  # [batch_size, pred_days]
            return predictions.unsqueeze(-1)  # [batch_size, pred_days, 1]


def create_seq2seq_model(config, model_type='direct'):
    """
    Create a multi-step forecasting model

    Args:
        config: Configuration object
        model_type: 'seq2seq' or 'direct'

    Returns:
        PyTorch model
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
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    from config import Config

    config = Config(TIME_STEP=60, PRED_DAYS=3)

    print("="*70)
    print("Test Seq2Seq LSTM model (encoder-decoder)")
    print("="*70)
    model_seq2seq = create_seq2seq_model(config, model_type='seq2seq')
    print(model_seq2seq)
    print(f"\nTotal parameters: {count_parameters(model_seq2seq):,}")

    # Test the forward pass
    batch_size = 8
    x = torch.randn(batch_size, config.TIME_STEP, 1)
    output = model_seq2seq(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}   {config.PRED_DAYS}-day forecast in one pass")

    print("\n" + "="*70)
    print("Test Direct LSTM model (direct multi-output)")
    print("="*70)
    model_direct = create_seq2seq_model(config, model_type='direct')
    print(model_direct)
    print(f"\nTotal parameters: {count_parameters(model_direct):,}")

    output = model_direct(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}   Directly output {config.PRED_DAYS} values")

    print("\nBoth models produce multi-step forecasts!")
    print("  - Seq2Seq: encoder-decoder architecture, intended for longer forecast sequences")
    print("  - Direct: Directly output N values, simpler implementation")