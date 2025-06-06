from naive_classifier import BayesClassifier
#from lstm import LSTMModel
import json
import os
import pickle
import pandas as pd
from data_collection import Stock

x_seq = []
x_norm = []
y = []


if os.path.exists("stocks.pkl"):
    with open("stocks.pkl", "rb") as f:
        stocks = pickle.load(f)
else:
    with open("companies.json", "r") as file:
        data = json.load(file)

    stocks = [Stock(stock["name"]) for stock in data["companies"]] # ADD WAY TO FAIL BAD STOCK

    with open("stocks.pkl", "wb") as f:
        pickle.dump(stocks, f)

for stock in stocks:
    
    if stock.historic_data is None: continue 

    bayes = BayesClassifier(stock.historic_data, stock.ticker)

    dates = stock.historic_data.index

    dates = [date.strftime("%Y-%m-%d %H:%M:%S").split()[0] for date in dates]

    trimmed = dates[:len(dates) - (len(dates) % 4)]
    groups = [trimmed[i:i+4] for i in range(0, len(trimmed), 4)]

    for group in groups:
        new_x = []
        ind_in_news = False
        ticker_in_news = False

        for idx, date in enumerate(group):
            if idx == 3:
                y.append(stock.historic_data.loc[date, "Close"])
                continue

            with open(f"wiki_data/wiki_{date}.txt", "r") as file:
                text = file.read()

            if stock.ticker in text: ticker_in_news = True
            if any(word in text for word in stock.industry): ind_in_news = True

            p, ne, n = bayes.classify(text)

            new_x.append([stock.historic_data.loc[date, "Open"], stock.historic_data.loc[date, "High"], stock.historic_data.loc[date, "Low"], stock.historic_data.loc[date, "Close"], stock.historic_data.loc[date, "Volume"], p, ne, n])
        x_seq.append(new_x)
        x_norm.append([stock.employees, int(ind_in_news), int(ticker_in_news)])


proccessed_data = {
    "x_seq": x_seq,
    "x_norm": x_norm,
    "y": y
}

df = pd.DataFrame(proccessed_data)

df.to_pickle("ai_data.pkl")