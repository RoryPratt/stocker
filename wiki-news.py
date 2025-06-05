import requests
from bs4 import BeautifulSoup
import re, string
from datetime import datetime, timedelta

def generate_wikipedia_event_urls(days_back=365):
    base_url = "https://en.wikipedia.org/wiki/Portal:Current_events"
    date_url_dict = {}

    for i in range(days_back):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        formatted = f"{date.year}_{date.strftime('%B')}_{date.day}"
        url = f"{base_url}/{formatted}"
        date_url_dict[date_str] = url

    return date_url_dict


def scrape_wiki_current_events(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return None
    return content.get_text(separator=" ")

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





#url = "https://en.wikipedia.org/wiki/Portal:Current_events/2025_June_4"
#text = clean_text(scrape_wiki_current_events(url))
#print(text) 


#urls = generate_wikipedia_event_urls()

#for date, url in urls.items():
#    text = clean_text(scrape_wiki_current_events(url))
#    
#    with open(f"wiki_data\wiki_{date}.txt", "w", encoding="utf-8") as f:
#        f.write(text)
