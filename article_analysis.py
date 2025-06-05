import pandas as pd
import re
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class Article:
	def __init__(self, raw):
		self.title = raw["title"]
		self.word_count = self.get_word_count(raw["title"])
		self.stock = raw["stock"]
		self.stock_data = None
		self.date = datetime.strptime(raw["date"].split()[0], "%Y-%m-%d")

	def __str__(self):
		return f"title: {self.title} \nstock: {self.stock} \ndate: {self.date}"

	def get_word_count(self, text):
		words = re.split(r"[a-zA-Z0-9]+", text.lower())

		temp = {}
		for word in words:
			temp[word] = temp.get(word, 0) + 1

		return temp

	def get_stock_data(self):
		time_range = timedelta(weeks=1)

		self.stock_data = yf.download(self.stock, start=(self.date - time_range), end=(self.date + time_range), interval="1d")









article_list = []

df = pd.read_csv("data/analyst_ratings_processed.csv")
df = df.sample(frac=1).reset_index(drop=True)[:10]

for _, article in df.iterrows():
	article_list.append(Article(article))


print(article_list[0])

article_list[0].get_stock_data()
article_list[0].graph_data()