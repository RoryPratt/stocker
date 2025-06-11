import os
import torch
import torch.nn as nn
import torch.optim as optim
from lstm import HybridLSTMNet
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Helper function to create or load scaler
def get_or_create_scaler(file_path, data, name="scaler"):
    if os.path.exists(file_path):
        print(f"Loading existing {name}...")
        return joblib.load(file_path)
    else:
        print(f"No {name} found - creating new one...")
        scaler = StandardScaler()
        scaler.fit(data)
        joblib.dump(scaler, file_path)
        return scaler

# Load data
df = pd.read_pickle("ai_data.pkl")
df = df[df["x_seq"].apply(lambda x: len(x) == 15)]

# Prepare data for scaler fitting
x_seq_array = np.stack(df["x_seq"].to_numpy())
price_part = x_seq_array[:, :, :-1].reshape(-1, 7)
volume_part = x_seq_array[:, :, -1].reshape(-1, 1)
static_input = np.stack(df["x_norm"].to_numpy())
y_array = np.stack(df["y"].to_numpy())
y_array_log = np.log1p(y_array)

# Load or create scalers
scaler_seq = get_or_create_scaler("seq_scaler.pkl", price_part, name="seq scaler")
volume_scaler = get_or_create_scaler("volume_scaler.pkl", volume_part, name="volume scaler")
scaler_static = get_or_create_scaler("static_scaler.pkl", static_input, name="static scaler")
y_scaler = get_or_create_scaler("y_scaler.pkl", y_array_log, name="y scaler")

# Apply log to target
df["y"] = df["y"].apply(lambda arr: np.log1p(arr))

# Split train/test
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Apply volume fix to sequence input
def scale_sequence(seq_array):
    seq_array = np.array(seq_array, dtype=np.float32)
    price_part = seq_array[:, :, :-1]  # first 7 features: price-related
    volume_part = seq_array[:, :, -1].reshape(-1, 1)  # last feature: volume

    # Scale price normally
    price_scaled = scaler_seq.transform(price_part.reshape(-1, price_part.shape[-1])).reshape(price_part.shape)

    # Scale volume separately
    volume_scaled = volume_scaler.transform(volume_part).reshape(seq_array.shape[0], seq_array.shape[1], 1)

    # Concatenate back
    return np.concatenate([price_scaled, volume_scaled], axis=2)

# Train dataset
x_seq_array = np.stack(df["x_seq"].to_numpy())
x_seq_scaled = scale_sequence(x_seq_array)
x = torch.tensor(x_seq_scaled, dtype=torch.float32)

static_input = np.stack(df["x_norm"].to_numpy())
static_input_scaled = scaler_static.transform(static_input)
static = torch.tensor(static_input_scaled, dtype=torch.float32)

y_array = np.stack(df["y"].to_numpy())
y_scaled = y_scaler.transform(y_array)
y = torch.tensor(y_scaled, dtype=torch.float32)

# Model setup
model = HybridLSTMNet(8, 300, 3, 500, 8)
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epoch = 0
try:
    while True:
        model.train()
        optimizer.zero_grad()
        output = model(x, static)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        epoch += 1
except KeyboardInterrupt:
    torch.save(model.state_dict(), "model_weights.pth")

# Testing phase
x_seq_array = np.stack(test_df["x_seq"].to_numpy())
x_seq_scaled = scale_sequence(x_seq_array)
x = torch.tensor(x_seq_scaled, dtype=torch.float32)

static_input = np.stack(test_df["x_norm"].to_numpy())
static_input_scaled = scaler_static.transform(static_input)
static = torch.tensor(static_input_scaled, dtype=torch.float32)

y = np.stack(test_df["y"].to_numpy())

model.eval()
with torch.no_grad():
    preds_scaled = model(x, static).numpy()
    preds = y_scaler.inverse_transform(preds_scaled)
    preds = np.expm1(preds)
    true = y.astype(np.float32)
    true = np.expm1(true)

    nonzero_mask = true != 0

    open_list = [d[2][0] for d in test_df["x_seq"].values]

    analyze_depth = 10
    for i in range(analyze_depth):
        print("\n")
        print(f"prediction analysis {i+1}")
        print(f"previous day: {open_list[i]}")
        print(f"prediction: {preds[i][3]}")
        print(f"true value: {true[i][0]}\n")

    percent_off = np.abs((preds[nonzero_mask][3] - true[nonzero_mask][3]) / true[nonzero_mask][3]) * 100
    avg_percent_off = np.mean(percent_off)
    print(f"Average Percent Off: {avg_percent_off:.2f}%")

    accuracy = 0
    length = 0

    total = 0
    max_error = 0

    for actual, prediction_y, prev_open in zip(true, preds, open_list):
        prediction = prediction_y[3]
        actual = actual[3]

        percent_off_prev = abs((prediction - prev_open) / prev_open) * 100
        actual_off_prev = abs((actual - prev_open) / prev_open) * 100

        length += 1
        total += abs(prediction - actual)
        max_error = max(abs(prediction - actual), max_error)

        if percent_off_prev < 3 and actual_off_prev < 3:
            accuracy += 1
            continue
        if (actual - prev_open > 0) == (prediction - prev_open > 0):
            accuracy += 1

    accuracy = (accuracy / length) * 100

    print(f"MAE: {total / length}, max: {max_error}")
    print()
    print(f"Accuracy: {accuracy}")
