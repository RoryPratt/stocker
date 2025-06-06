from data_collection import Stock
from typing import List, Callable, Tuple, Any, Match
from match import match
import matplotlib.pyplot as plt
from naive_classifier import BayesClassifier

print("Stonks\n")
stock = Stock(input("What stock do you want to analyze? "))

def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt

def switch_stock(name):
    global stock
    stock = Stock(name)
    return [""]

def print_stock(_): return [stock]



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