from bs4 import BeautifulSoup
from selenium import webdriver
import time
import re
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
BASE = "https://www.immobilier.ch/en/rent/apartment-house/{slug}/page-{page}"
SLEEP = 5
MAX_PAGES_PER_CANTON = 20  # manual limit

CANTONS = {
    "ZH": "zurich",
    "BE": "berne",
    "LU": "lucerne",
    "UR": "uri",
    "SZ": "schwyz",
    "OW": "obwald",
    "NW": "nidwald",
    "GL": "glaris",
    "ZG": "zoug",
    "FR": "fribourg",
    "SO": "soleure",
    "BS": "bale-ville",
    "BL": "bale-campagne",
    "SH": "schaffhouse",
    "AR": "appenzell-rhodes-exterieures",
    "AI": "appenzell-rhodes-interieures",
    "SG": "st-gall",
    "GR": "grisons",
    "AG": "argovie",
    "TG": "thurgovie",
    "TI": "tessin",
    "VD": "vaud",
    "VS": "valais",
    "NE": "neuchatel",
    "GE": "geneva",
    "JU": "jura",
}
# ---------------------------
# PARSERS (your original style)
# ---------------------------
def parse_price(txt):
    m = re.search(r"CHF\s*([\d'., ]+)", txt)
    if not m:
        return None
    return int(re.sub(r"[^\d]", "", m.group(1)))

# --- PARSE AREA ---
def parse_area(txt):
    txt = txt.replace(",", ".")
    # handles: 59 m 2, 59 m2, 59 m², 59 sqm, 59 sq m
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


# ---------------------------
# SCRAPER
# ---------------------------
driver = webdriver.Chrome()

all_rows = []
seen_urls = set()
today = pd.Timestamp.today().date().isoformat()  # scrape date (not publication date)

for code, slug in CANTONS.items():
    print("\n==============================")
    print("CANTON:", code)

    page = 1

    while page <= MAX_PAGES_PER_CANTON:
        url = BASE.format(slug=slug, page=page)
        print(f"\n{code} | page-{page} -> {url}")

        driver.get(url)
        time.sleep(SLEEP)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # listing cards (ads are not div.filter-item)
        cards = soup.select("div.filter-item[id^='filter-item-']")

        if not cards:
            print("No cards found -> stop this canton.")
            break

        print("Cards found:", len(cards))

        # 1) Collect links on THIS page first (for repeat detection)
        links_on_page = []
        card_payloads = []  # keep card_text + link together

        for card in cards:
            card_text = card.get_text(" ", strip=True)

            a = card.select_one("a[id^='link-result-item-']")
            link = None
            if a and a.get("href"):
                link = a.get("href")
                if link.startswith("/"):
                    link = "https://www.immobilier.ch" + link

            if link:
                links_on_page.append(link)
                card_payloads.append((card_text, link))

        # If we couldn't extract any link, stop (something changed in HTML)
        if not links_on_page:
            print("No links extracted -> stop this canton (HTML structure may have changed).")
            break

        # 2) If ALL links on this page were already seen -> repeated/looping page -> stop
        if all(link in seen_urls for link in links_on_page):
            print("Repeated page detected (same listings again) -> stop this canton.")
            break

        # 3) Append only NEW listings
        new_count = 0
        for card_text, link in card_payloads:
            if link in seen_urls:
                continue

            seen_urls.add(link)
            new_count += 1

            all_rows.append({
                "canton": code,
                "date_scraped": today,
                "price_chf": parse_price(card_text),
                "living_area_m2": parse_area(card_text),
                "rooms": parse_rooms(card_text),
                "location_text": parse_location(card_text),
                "listing_url": link,
                "source": "immobilier.ch"
            })

        print("New listings added:", new_count)

        page += 1

driver.quit()

df = pd.DataFrame(all_rows)

# cleaning (same logic as before)
if not df.empty:
    df = df[df["price_chf"].notna()].drop_duplicates(subset=["canton", "listing_url"])
else:
    print("\nWARNING: No rows collected.")

out_file = "../data/processed/immobilier_all_cantons_allpages_snapshot_p20.csv"
df.to_csv(out_file, index=False)

print("\nSaved ->", out_file, "| rows:", len(df))
print(df.head(10))

#--------------read

df = pd.read_csv("/Users/sedadulger/PyCharmProjects/CIP/project/data/processed/"
                 "immobilier_all_cantons_allpages_snapshot_p20.csv")
print(df.dtypes)
print(df.shape)
print(df.isna().sum())
print(df["canton"].nunique())

# OLD DATASET
# shape ≈ (4689, 8)
# living_area_m2    4689
# rooms             very high (mostly missing)
# location_text     very high (mostly missing)
# canton            0
# price_chf         0
# listing_url       0
# source            0

# NEW DATASET
# shape: (4731, 8)
# living_area_m2    653
# rooms             63
# location_text     49
# canton            0
# price_chf         0
# listing_url       0
# source            0