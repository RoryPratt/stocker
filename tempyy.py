import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_pickle("ai_data.pkl")
df = df[df["x_seq"].apply(lambda x: len(x) == 15)]

x_seq_array = np.stack(df["x_seq"].to_numpy())
price_part = x_seq_array[:, :, :-1].reshape(-1, 7)  # 7 price features

scaler_seq = StandardScaler()
scaler_seq.fit(price_part)

joblib.dump(scaler_seq, "seq_scaler.pkl")
print("seq_scaler.pkl updated for price-only scaling.")