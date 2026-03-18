from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re

# DEBUG CONFIG
BASE = "https://www.immobilier.ch/en/rent/apartment-house/{slug}/page-{page}"
TEST_SLUG = "zurich"
TEST_PAGE = 1
SLEEP = 3
N_CARDS = 5

def parse_area(txt):
    txt = txt.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\s*2|m²|m2|sq\s*m|sqm)\b", txt, re.IGNORECASE)
    return float(m.group(1)) if m else None

def parse_rooms(txt):
    txt = txt.replace(",", ".")
    # 1) "Apartment 2.5 rooms" gibi
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*rooms?\b", txt, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # 2) bazen sonda "m 2 2.5" gibi tekrar ediyor -> en sondaki sayıyı al (metrekareden sonra)
    m2 = re.search(r"(?:m\s*2|m²|m2)\s*(\d+(?:\.\d+)?)\b", txt, re.IGNORECASE)
    return float(m2.group(1)) if m2 else None

def parse_location(txt):
    # 1) Öncelik: 4 haneli posta kodu + şehir (varsa)
    m = re.search(r"\b(\d{4}\s+[A-Za-zÀ-ÿ'’\- ]+)\b", txt)
    if m:
        loc = re.sub(r"\s{2,}", " ", m.group(1).strip())
        return loc

    # 2) Fallback: "rooms" kelimesinden sonra gelen ilk konum parçası
    # Örn: "... Apartment 2.5 rooms Zürich, Neunbrunnenstrasse 188 59 m 2 2.5"
    m2 = re.search(r"rooms?\s+([^,]+)", txt, re.IGNORECASE)
    if not m2:
        return None

    loc = m2.group(1).strip()

    # temizle: sondaki gereksiz kelimeler/boşluklar
    loc = re.sub(r"\s{2,}", " ", loc)

    # çok saçma uzun string gelirse kırp (ör. tüm cümleyi alırsa)
    if len(loc) > 40:
        loc = loc[:40].rstrip()

    return loc

driver = webdriver.Chrome()

url = BASE.format(slug=TEST_SLUG, page=TEST_PAGE)
print("OPEN:", url)

driver.get(url)
time.sleep(SLEEP)

soup = BeautifulSoup(driver.page_source, "html.parser")
cards = soup.select("div.filter-item[id^='filter-item-']")

print("Cards found:", len(cards))

for i, card in enumerate(cards[:N_CARDS], start=1):
    card_text = card.get_text(" ", strip=True)

    area = parse_area(card_text)
    rooms = parse_rooms(card_text)
    location = parse_location(card_text)

    print(f"\n--- CARD {i} ---")
    print("TEXT:", card_text)
    print("AREA:", area)
    print("ROOMS:", rooms)
    print("LOCATION:", location)

driver.quit()