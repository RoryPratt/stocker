import torch
import torch.nn as nn
import torch.optim as optim
from lstm import HybridLSTMNet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

model = HybridLSTMNet(8, 10, 3, 12, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

df = pd.read_pickle("ai_data.pkl")

df, test_df = train_test_split(df, test_size=0.2, random_state=42)


x_seq = df["x_seq"].apply(lambda x: np.array(x, dtype=np.float32))
x_seq_array = np.stack(x_seq.to_numpy())
x = torch.tensor(x_seq_array, dtype=torch.float32)

static_input = np.stack(df["x_norm"].to_numpy())
static = torch.tensor(static_input, dtype=torch.float32)

y = torch.tensor(df["y"].values, dtype=torch.float32).unsqueeze(1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(x, static)
    loss = criterion(output, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model_weights.pth")


x_seq = test_df["x_seq"].apply(lambda x: np.array(x, dtype=np.float32))
x_seq_array = np.stack(x_seq.to_numpy())
x = torch.tensor(x_seq_array, dtype=torch.float32)

static_input = np.stack(test_df["x_norm"].to_numpy())
static = torch.tensor(static_input, dtype=torch.float32)

y = torch.tensor(test_df["y"].values, dtype=torch.float32).unsqueeze(1)

model.eval()
with torch.no_grad():
    preds = model(x, static).detach().numpy()
    true = y.numpy()

    nonzero_mask = true != 0
    mape = np.mean(np.abs((preds[nonzero_mask] - true[nonzero_mask]) / true[nonzero_mask])) * 100

    print(f"MAPE: {mape:.2f}%")


"""
model = MyModel()  # Re-initialize model with same architecture
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
"""