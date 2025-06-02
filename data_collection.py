import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    try:
        results = wikipedia.search(title)
        return WikipediaPage(results[0]).html()
    except:
        results = wikipedia.search(title + " company").html()
        return WikipediaPage(results[0]).html


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


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
        self.name = name
        self.raw_wiki = ""
        self.patterns = [
            r"Traded asNasdaq: (?P<ticker>[A-Z]+)Nas",
            r"Industry(?P<industry>[\w ]+)Pred",
            #r"Number of employees (?P<employees>[0-9,]+)",
            #r"Area served(?P<area_served>\w+)Key"
        ]
        
        self.load_stock()

    def __str__(self):
        return f"{self.ticker} / {self.name} \nIndustry: {self.Industry}"

    def load_stock(self):
        infobox_text = clean_text(get_first_infobox_text(get_page_html(self.name)))
        self.raw_wiki = infobox_text

        #(self.raw_wiki)

        error_text = (
            "Page infobox has no information"
        )

        pattern = "".join(f"(?=.*{p})" for p in self.patterns)
        
        match = get_match(infobox_text, pattern, error_text)

        self.ticker = match.group("ticker")
        self.industry = match.group("industry")
