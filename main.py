from data_collection import Stock
from typing import List, Callable, Tuple, Any, Match
from match import match
import matplotlib.pyplot as plt
from naive_classifier import BayesClassifier
from lstm import LSTMModel

print("Stonks\n")
stock = Stock(input("What stock do you want to analyze? "))

def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt

def switch_stock(name):
    global stock
    stock = Stock(name)
    return [""]

def print_stock(_): return [stock]

def train_model(_):
    global stock
    
    bayes = BayesClassifier(stock.historic_data, stock.ticker)


    #inputs: 
    # open, close, high, low, volume, wiki_sentiment - 3, 
    # employees, industry in wiki file, ticker in wiki file

    x_seq = []
    x_norm = []
    y = []

    dates = stock.historic_data.index

    trimmed = dates[:len(dates) - (len(dates) % 4)]
    groups = [trimmed[i:i+4] for i in range(0, len(trimmed), 4)]

    for group in groups:
        new_x = []
        for idx, date in enumerate(group):
            if idx == 3:
                y.append(stock.historic_data.loc[date, "Close"])
                continue

            with open(f"wiki_data/wiki_{date}.txt", "r") as file:
                text = file.read()

            p, ne, n = bayes.classify(text)

            new_x.append([stock.historic_data.loc[date, "Open"], stock.historic_data.loc[date, "High"], stock.historic_data.loc[date, "Low"], stock.historic_data.loc[date, "Close"], stock.historic_data.loc[date, "Volume"], p, ne, n])
        x_seq.append(new_x)
        x_norm.append
    #outputs


    model = HybridLSTMNet(8, lstm_hidden_size=64, static_input_size=5, hidden_fc_size=32, output_size=1)
    
    return [bayes.classify("how do you do today war war war.")]



def graph_data(matches):
    graph_type = matches[0]

    if graph_type not in ["open", "close", "high", "low", "volume"]: return ["invalid input"]

    y = stock.historic_data[graph_type.capitalize()]
    x = stock.historic_data.index

    plt.plot(x, y, label=f"{graph_type}")

    plt.xlabel("Day")
    plt.ylabel(graph_type)
    plt.title(f"{graph_type} over time for {stock.name}")
    plt.legend()

    plt.show()
    return [""]

Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

pa_list: List[Tuple[Pattern, Action]] = [
    ("analyze %".split(), switch_stock),
    ("print data".split(), print_stock),
    ("graph stock %".split(), graph_data),
    ("train sentiment analysis model".split(), train_model),
    (["bye"], bye_action),
]

def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]

def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


query_loop()