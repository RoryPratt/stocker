import torch
import torch.nn as nn
import torch.optim as optim
from lstm import HybridLSTMNet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

scaler_seq = StandardScaler()
scaler_static = StandardScaler()
y_scaler = StandardScaler()

model = HybridLSTMNet(8, 10, 3, 12, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

df = pd.read_pickle("ai_data.pkl")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df, test_df = train_test_split(df, test_size=0.2, random_state=42)

x_seq = df["x_seq"].apply(lambda x: np.array(x, dtype=np.float32))
x_seq_array = np.stack(x_seq.to_numpy())
b, s, f = x_seq_array.shape
x_seq_scaled = scaler_seq.fit_transform(x_seq_array.reshape(-1, f)).reshape(b, s, f)
x = torch.tensor(x_seq_scaled, dtype=torch.float32)

static_input = np.stack(df["x_norm"].to_numpy())
static_input_scaled = scaler_static.fit_transform(static_input)
static = torch.tensor(static_input_scaled, dtype=torch.float32)

y = df["y"].values.reshape(-1, 1)
y_scaled = y_scaler.fit_transform(y)
y = torch.tensor(y_scaled, dtype=torch.float32)

for epoch in range(1000):
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
b, s, f = x_seq_array.shape
x_seq_scaled = scaler_seq.transform(x_seq_array.reshape(-1, f)).reshape(b, s, f)
x = torch.tensor(x_seq_scaled, dtype=torch.float32)

static_input = np.stack(test_df["x_norm"].to_numpy())
static_input_scaled = scaler_static.transform(static_input)
static = torch.tensor(static_input_scaled, dtype=torch.float32)

y = test_df["y"].values

model.eval()
with torch.no_grad():
    preds = y_scaler.inverse_transform(model(x, static).numpy())

    preds = np.array([item[0] for item in preds])

    true = y.astype(np.float32)

    nonzero_mask = true != 0

    open_list = [d[2][0] for d in test_df["x_seq"].values]

    analyze_depth = 10

    for i in range(analyze_depth):
        print("\n")
        print(f"prediction analysis {i+1}")
        print(f"previous day: {open_list[i]}\nprediction: {preds[i]}\ntrue value: {true[i]}")

        print("\n")

    mae = mean_absolute_error(true, preds)
    print(f"MAE: {mae:.2f}")

    percent_off = np.abs((preds - true) / true) * 100
    avg_percent_off = np.mean(percent_off)
    print(f"Average Percent Off: {avg_percent_off:.2f}%")


"""
model = MyModel()  # Re-initialize model with same architecture
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
"""