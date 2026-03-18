from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re

# DEBUG CONFIG (tek sayfa, tek kanton)
BASE = "https://www.immobilier.ch/en/rent/apartment-house/{slug}/page-{page}"
TEST_SLUG = "zurich"
TEST_PAGE = 1
SLEEP = 3
N_CARDS = 3   # kaç kart görmek istiyorsun?

# --- PARSE AREA ---
def parse_area(txt):
    txt = txt.replace(",", ".")
    # handles: 59 m 2, 59 m2, 59 m², 59 sqm, 59 sq m
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\s*2|m²|m2|sq\s*m|sqm)\b", txt, re.IGNORECASE)
    return float(m.group(1)) if m else None

# --- DRIVER ---
driver = webdriver.Chrome()

url = BASE.format(slug=TEST_SLUG, page=TEST_PAGE)
print("OPEN:", url)

driver.get(url)
time.sleep(SLEEP)

soup = BeautifulSoup(driver.page_source, "html.parser")
cards = soup.select("div.filter-item[id^='filter-item-']")

print("Cards found:", len(cards))

# --- DEBUG OUTPUT ---
for i, card in enumerate(cards[:N_CARDS], start=1):
    card_text = card.get_text(" ", strip=True)

    area = parse_area(card_text)

    print(f"\n--- CARD {i} ---")
    print("TEXT:", card_text)
    print("AREA:", area)

driver.quit()