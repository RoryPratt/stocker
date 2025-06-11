import torch
import numpy as np
import joblib
from lstm import HybridLSTMNet
from datetime import datetime, timedelta
from wiki_news import clean_text, scrape_wiki_current_events, generate_wiki_url
from naive_classifier import BayesClassifier
from data_collection import Stock


class AI:
    def __init__(self, stock):
        self.stock = stock

        self.model = HybridLSTMNet(8, 300, 3, 500, 8)
        self.model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
        self.model.eval()

        self.seq_scaler = joblib.load("seq_scaler.pkl")
        self.volume_scaler = joblib.load("volume_scaler.pkl")
        self.static_scaler = joblib.load("static_scaler.pkl")
        self.y_scaler = joblib.load("y_scaler.pkl")

        self.bayes = BayesClassifier()

    def predict(self, date_str, days_ahead):
        historical_seq, static_input = self.prepare_inputs(date_str)
        if historical_seq is None:
            return ["Could Not Find Recent Stock Data"]

        predictions = self.recursive_predict(historical_seq, static_input, days_ahead)

        for i, pred in enumerate(predictions, 1):
            print(f"Prediction Day {i}: {pred[3]}")

        return predictions

    def prepare_inputs(self, date_str):
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        for _ in range(3):
            target_date -= timedelta(days=1)
            if target_date.strftime("%Y-%m-%d") in self.stock.historic_data.index:
                break
        else:
            return None, None

        sorted_index = sorted([d.date() for d in self.stock.historic_data.index])
        idx = sorted_index.index(target_date.date())
        if idx < 14:
            return None, None

        dates = [sorted_index[idx - i].strftime("%Y-%m-%d") for i in range(15)]

        seq_data = []
        wiki_data = []
        tick_in_news, ind_in_news = False, False

        for date in dates:
            text = clean_text(scrape_wiki_current_events(generate_wiki_url(date))).lower()
            wiki_sentiment = self.bayes.classify(text)
            wiki_data.append(wiki_sentiment)

            if self.stock.ticker in text:
                tick_in_news = True
            if any(ind.lower() in text for ind in self.stock.industry):
                ind_in_news = True

            daily_data = self.stock.historic_data.loc[date]
            seq_data.append([
                daily_data["Open"],
                daily_data["High"],
                daily_data["Low"],
                daily_data["Close"],
                daily_data["Volume"]
            ])

        full_seq = []
        for stock_row, wiki_row in zip(seq_data, wiki_data):
            full_seq.append(stock_row + list(wiki_row))

        static_input = [self.stock.employees, int(tick_in_news), int(ind_in_news)]

        return full_seq, static_input

    def scale_sequence(self, seq_array):
        seq_array = np.array(seq_array, dtype=np.float32)
        price_part = seq_array[:, :-1]  # (15, 7)
        volume_part = seq_array[:, -1].reshape(-1, 1)  # (15, 1)

        price_scaled = self.seq_scaler.transform(price_part.reshape(-1, price_part.shape[-1])).reshape(price_part.shape)
        volume_scaled = self.volume_scaler.transform(volume_part).reshape(seq_array.shape[0], 1)

        return np.concatenate([price_scaled, volume_scaled], axis=1)

    def recursive_predict(self, seq_input, static_input, future_days):
        seq_array = np.array(seq_input, dtype=np.float32)

        static_array = np.array(static_input, dtype=np.float32)
        static_array = np.log1p(static_array)
        static_scaled = self.static_scaler.transform(static_array.reshape(1, -1))
        static_tensor = torch.tensor(static_scaled, dtype=torch.float32)

        predictions = []

        for _ in range(future_days):
            scaled_seq = self.scale_sequence(seq_array)

            scaled_seq = scaled_seq.reshape(1, 15, 8)
            seq_tensor = torch.tensor(scaled_seq, dtype=torch.float32)

            with torch.no_grad():
                pred_scaled = self.model(seq_tensor, static_tensor).numpy()
                pred_inversed = self.y_scaler.inverse_transform(pred_scaled)
                pred_final = np.expm1(pred_inversed)

            predictions.append(pred_final[0])

            # Prepare new input for next prediction
            new_entry = pred_final[0][:8]
            seq_array = np.vstack([seq_array[1:], new_entry])

        return predictions



#stock = Stock("Microsoft")

#ai = AI(stock)

#print(ai.predict("2025-6-9", 1))