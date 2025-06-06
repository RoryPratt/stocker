import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from typing import List, Callable, Tuple, Any, Match
import yfinance as yf

def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.find("table", class_="infobox")

    if not infobox:
        raise LookupError("Page has no infobox")

    info = {}
    for row in infobox.find_all("tr"):
        header = row.find("th")
        data = row.find("td")
        if header and data:
            key = header.get_text(" ", strip=True)
            value = data.get_text(" ", strip=True)
            info[key] = value

    return info


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match

class Stock:
    def __init__(self, name):
        if type(name) == str:
            self.name = name
        else:
            self.name = name[0]
        self.raw_wiki = ""
        self.ticker = None
        self.industry = None
        self.employees = None

        try: 
            self.load_stock()
        except:
            self.load_stock()

        self.historic_data = self.get_historic_data()
        #self.recent_articles = self.get_recent_articles()

    def __str__(self):
        return f"Name: {self.ticker} / {self.name} \nIndustry: {self.industry}\nNumber of Employees: {self.employees}"

    def load_stock(self):
        infobox_text = get_first_infobox_text(get_page_html(self.name))

        self.raw_wiki = infobox_text

        error_text = (
            "Page infobox has no information"
        )

        self.ticker = get_match(infobox_text["Traded as"], r":\s*(?P<ticker>[A-Z]+)", error_text + f" {self.name}").group("ticker")
        self.industry = re.findall(r"[A-Z][a-z ]+", infobox_text["Industry"])
        self.industry = [industry[:-1] if industry.endswith(" ") else industry for industry in self.industry]
        self.employees = int(get_match(infobox_text["Number of employees"], r"c*\.* *(?P<employees>[0-9,]+)", error_text).group("employees").replace(",", ""))

    def get_historic_data(self):
        ticker = yf.Ticker(self.ticker)

        return ticker.history(period="1y", interval="1d")