import torch
import torch.nn as nn

class HybridLSTMNet(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, static_input_size, hidden_fc_size, output_size):
        super(HybridLSTMNet, self).__init__()
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size + static_input_size, hidden_fc_size)
        self.fc2 = nn.Linear(hidden_fc_size, output_size)

    def forward(self, seq_input, static_input):
        # seq_input shape: (batch, seq_len, lstm_input_size)
        lstm_out, _ = self.lstm(seq_input)
        last_lstm_out = lstm_out[:, -1, :]  # Take last timestep

        # Concatenate LSTM output with static input
        combined = torch.cat((last_lstm_out, static_input), dim=1)

        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        return x