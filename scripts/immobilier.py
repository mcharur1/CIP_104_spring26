from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
import pandas as pd

BASE = "https://www.immobilier.ch/en/rent/apartment-house/{slug}/page-1"
SLEEP = 5

CANTONS = {
    "ZH": "Zurich", #?
    "BE": "bern",
    "LU": "lucerne",
    "UR": "uri",
    "SZ": "schwyz",
    "OW": "obwalden",
    "NW": "nidwalden",
    "GL": "glarus",
    "ZG": "zug",
    "FR": "fribourg",
    "SO": "solothurn",
    "BS": "basel-city",
    "BL": "basel-country",
    "SH": "schaffhausen",
    "AR": "appenzell-outerrhodes",
    "AI": "appenzell-innerrhodes",
    "SG": "st-gallen",
    "GR": "graubunden",
    "AG": "aargau",
    "TG": "thurgau",
    "TI": "ticino",
    "VD": "vaud",
    "VS": "valais",
    "NE": "neuchatel",
    "GE": "geneva",
    "JU": "jura",
}

def parse_price(txt):
    m = re.search(r"CHF\s*([\d'., ]+)", txt)
    if not m:
        return None
    return int(re.sub(r"[^\d]", "", m.group(1)))

def parse_area(txt):
    txt = txt.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*m²", txt)
    return float(m.group(1)) if m else None

def parse_rooms(txt):
    txt = txt.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*rooms?", txt, re.IGNORECASE)
    return float(m.group(1)) if m else None

def parse_location(txt):
    m = re.search(r"\b\d{4}\s+[A-Za-zÄÖÜäöü\- ]+\b", txt)
    return m.group(0).strip() if m else None

driver = webdriver.Chrome()
all_rows = []
today = pd.Timestamp.today().date().isoformat() # it shouldn't be today, last tree months at least

for code, slug in CANTONS.items():
    url = BASE.format(slug=slug)
    print("\n", code, url)

    driver.get(url)
    time.sleep(SLEEP)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    prices = soup.select("strong.title")
    print("Found prices:", len(prices))

    for p in prices:
        card = p.find_parent("div")
        card_text = card.get_text(" ", strip=True) if card else p.get_text(" ", strip=True)

        a = p.find_parent("a")
        link = None
        if a and a.get("href"):
            link = a.get("href")
            if link.startswith("/"):
                link = "https://www.immobilier.ch" + link

        all_rows.append({
            "canton": code,
            "date_scraped":today, #check
            "price_chf": parse_price(card_text),
            "living_area_m2": parse_area(card_text),
            "rooms": parse_rooms(card_text),
            "location_text": parse_location(card_text),
            "listing_url": link,
            "source": "immobilier.ch"
        })

driver.quit()

df = pd.DataFrame(all_rows)
df = df[df["price_chf"].notna()].drop_duplicates(subset=["canton", "listing_url"])


df.to_csv("../data/processed/immobilier_all_cantons.csv", index=False)
print("\nSaved -> immobilier_all_cantons.csv | rows:", len(df))
print(df.head(10))

#---------NOTES----------#
# rent, sale both of them
# at least last three months
# all pages
# careful about the formats
# try to join two dataset
## if possible check other wen sites
## if possible find population and tax dateset just in case
