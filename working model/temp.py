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

scaler_seq = StandardScaler() #joblib.load("seq_scaler.pkl")
scaler_static = StandardScaler() #joblib.load("static_scaler.pkl")
y_scaler = StandardScaler() #joblib.load("y_scaler.pkl")

model = HybridLSTMNet(8, 80, 3, 100, 1)
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

df = pd.read_pickle("ai_data.pkl")

df = df[df["x_seq"].apply(lambda x: len(x) == 15)]



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

y_array = np.stack(df["y"].to_numpy())
y_scaled = y_scaler.fit_transform(y_array)
y = torch.tensor(y_scaled, dtype=torch.float32)


with open("y_scaler.pkl", "wb") as f:
    joblib.dump(y_scaler, f)

with open("seq_scaler.pkl", "wb") as f:
    joblib.dump(scaler_seq, f)

with open("static_scaler.pkl", "wb") as f:
    joblib.dump(scaler_static, f)
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

x_seq = test_df["x_seq"].apply(lambda x: np.array(x, dtype=np.float32))
x_seq_array = np.stack(x_seq.to_numpy())
b, s, f = x_seq_array.shape
x_seq_scaled = scaler_seq.transform(x_seq_array.reshape(-1, f)).reshape(b, s, f)
x = torch.tensor(x_seq_scaled, dtype=torch.float32)

static_input = np.stack(test_df["x_norm"].to_numpy())
static_input_scaled = scaler_static.transform(static_input)
static = torch.tensor(static_input_scaled, dtype=torch.float32)

y = np.stack(test_df["y"].to_numpy())

model.eval()
with torch.no_grad():
    preds_scaled = model(x, static).numpy()
    preds = y_scaler.inverse_transform(preds_scaled)

    true = y.astype(np.float32)

    nonzero_mask = true != 0

    open_list = [d[2][0] for d in test_df["x_seq"].values]


    accuracy = 0
    length = 0

    for actual, prediction_y, prev_open in zip(true, preds, open_list):
        prediction = prediction_y

        
        percent_off_prev = abs((prediction - prev_open) / prev_open) * 100

        #actual_off_prev = abs((actual - prev_open) / prev_open) * 100

        length += 1

        accuracy += int(actual == int(float(prediction[0]) >= 0.5))

        #if percent_off_prev < 3 and actual_off_prev < 3:
        #    accuracy += 1
        #    continue
        #if actual == int(prediction - prev_open > 0):
        #    accuracy += 1

    accuracy = (accuracy / length) * 100
    print()
    print(f"Accuracy: {accuracy}")


"""
model = MyModel()  # Re-initialize model with same architecture
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
"""