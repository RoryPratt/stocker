from lstm import HybridLSTMNet
from wiki_news import clean_text, scrape_wiki_current_events, generate_wiki_url
from datetime import datetime, timedelta
import torch
import numpy as np
import joblib
from naive_classifier import BayesClassifier

from data_collection import Stock


class AI:
	def __init__(self, stock):
		self.model = HybridLSTMNet(8, 10, 3, 12, 1)
		self.stock = stock

		self.model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
		self.model.eval()

		self.bayes = BayesClassifier(stock.historic_data, stock.ticker)

	def predict(self, date_str):
		prev_day = datetime.strptime(date_str, "%Y-%m-%d")
		found_day = False

		for i in range(3):
			prev_day = prev_day - timedelta(days=1)

			if prev_day.strftime("%Y-%m-%d") in self.stock.historic_data.index:
				found_day = True
				break

		if not found_day: return ["Could Not Find Recent Stock Data"]

		sorted_index = dates_dt = sorted([d.date() for d in self.stock.historic_data.index])

		dates = [prev_day.strftime("%Y-%m-%d")]

		for i in range(1, 3):
			index = sorted_index.index(prev_day.date())
			dates.append(sorted_index[index - i].strftime("%Y-%m-%d"))


		wiki_news_list = []

		stock_data = []

		tick_in_news = False
		ind_in_news = False

		for date in dates:
			text = clean_text(scrape_wiki_current_events(generate_wiki_url(date)))

			if self.stock.ticker in text: tick_in_news = True
			if any(word in text for word in self.stock.industry): ind_in_news = True #LOWERCASE WORD AND TICKER
			
			wiki_news_list.append(self.bayes.classify(text))

			stock_data.append([self.stock.historic_data.loc[date, "Open"], self.stock.historic_data.loc[date, "High"], self.stock.historic_data.loc[date, "Low"], self.stock.historic_data.loc[date, "Close"], self.stock.historic_data.loc[date, "Volume"]])

		seq_input = []
		
		for wiki, stock in zip(wiki_news_list, stock_data):
			stock.extend(wiki)
			seq_input.append(stock)

		seq_input.reverse()

		print(seq_input)

		static_input = [self.stock.employees, int(tick_in_news), int(ind_in_news)]


		seq_scaler = joblib.load("seq_scaler.pkl")
		static_scaler = joblib.load("static_scaler.pkl")
		y_scaler = joblib.load("y_scaler.pkl")


		seq_array = np.array(seq_input, dtype=np.float32)
		seq_scaled = seq_scaler.transform(seq_array)
		scaled_seq = torch.tensor(seq_scaled.reshape(1, *seq_scaled.shape), dtype=torch.float32)

		scaled_static = torch.tensor(static_scaler.transform(np.array(static_input).reshape(1, -1)), dtype=torch.float32)

		print(y_scaler.inverse_transform(self.model(scaled_seq, scaled_static).detach().numpy())[0][0])

		
stock = Stock("Microsoft")
ai = AI(stock)
date = "2025-3-5"


ai.predict(date)