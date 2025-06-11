from naive_classifier import BayesClassifier
import json
import os
import pickle
from wiki_news import clean_text
import pandas as pd
from data_collection import Stock
from sentiment import get_sentiment

group_length = 15

x_seq = []
x_norm = []
y = []


group_length += 1

if os.path.exists("stocks.pkl"):
    with open("stocks.pkl", "rb") as f:
        stocks = pickle.load(f)
else:
    with open("companies.json", "r") as file:
        data = json.load(file)

    stocksy = [stock["name"] for stock in data["companies"]]

    stocks = []

    for stock in stocksy:
        try:
            stocks.append(Stock(stock))
        except:
            continue

    with open("stocks.pkl", "wb") as f:
        pickle.dump(stocks, f)


bayes = BayesClassifier()

for stock in stocks:
    
    if stock.historic_data is None: continue

    dates = stock.historic_data.index

    dates = [date.strftime("%Y-%m-%d %H:%M:%S").split()[0] for date in dates]

    trimmed = dates[:len(dates) - (len(dates) % group_length)]
    groups = [trimmed[i:i+group_length] for i in range(0, len(trimmed), group_length)]

    for group in groups:
        new_x = []
        ind_in_news = False
        ticker_in_news = False
        group_good = True
        prev_date = ""

        for idx, date in enumerate(group):
            if idx == 3:
                
                try:
                    with open(f"wiki_data/wiki_{date}.txt", "r") as file:
                        text = file.read()
                except:
                    group_good = False
                    continue

                text = clean_text(text).lower()

                ticker_in_news_y = stock.ticker.lower() in text
                ind_in_news_y = any(industry.lower() in text for industry in stock.industry)


                p, ne, n = bayes.classify(text)


                listy = [stock.historic_data.loc[date, "Open"], stock.historic_data.loc[date, "High"], stock.historic_data.loc[date, "Low"], stock.historic_data.loc[date, "Close"], stock.historic_data.loc[date, "Volume"], p, ne, n]
               

                if group_good:
                    y.append(listy)
                
                continue
            try:
                with open(f"wiki_data/wiki_{date}.txt", "r") as file:
                    text = file.read()
            except:
                group_good = False
                continue

            text = clean_text(text).lower()

            if stock.ticker in text: ticker_in_news = True
            if any(industry in text for industry in stock.industry): ind_in_news = True

            p, ne, n = bayes.classify(text)
            prev_date = date
            listx = [stock.historic_data.loc[date, "Open"], stock.historic_data.loc[date, "High"], stock.historic_data.loc[date, "Low"], stock.historic_data.loc[date, "Close"], stock.historic_data.loc[date, "Volume"], p, ne, n]

            new_x.append(listx)

        if group_good:
            x_seq.append(new_x)
            x_norm.append([stock.employees, int(ind_in_news), int(ticker_in_news)])


proccessed_data = {
    "x_seq": x_seq,
    "x_norm": x_norm,
    "y": y
}

df = pd.DataFrame(proccessed_data)

df.to_pickle("ai_data.pkl")