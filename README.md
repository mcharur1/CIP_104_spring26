# CIP_FS2026_104 - GDP and Apartment Rental Prices Across Swiss Cantons 

## Repository Structure
```text
CIP_FS2026_104/
├── data/
│   ├── raw/                                        # Original data files
│   │   ├── orijinal.xlsx                           # Original DGP by canton  # CAN WE REMOVE??
│   │   └── gdp_percapita_canton_2022.xlsx          # GDP data by canton
│   └── processed/                                  # Processed datasets
│       ├── canton_gdp_2022_clean.csv
│       └── immobilier_all_cantons_allpages_snapshot_p20.csv
├── scripts/
│   ├── immobilier.py                  # Initial scraper (subset one canton, one page)
│   ├── immobilier_allpages.py         # Full scraper (all cantons and pages)
│   ├── data_processing.py             # Data cleaning and transformation
│   ├── gdpdataset.py                  # GDP data cleaning
│   ├── population_by_canton_2022.py   # Future outlook population dataset  ## CAN WE REMOVE??
│   ├── model.py                       #  analysis
│   ├── card_preview.py                # Single listing variable extraction functions
│   └── rooms_location_preview.py      # Location-based analysis
├── Figures/                           # Output from data_processing.py included in report
│   ├── figure_1_gdp_vs_log_price_per_m2.png
│   ├── figure_2_distribution_log_price_by_economic_and_structural_type.png
│   └── figure_3_r2_explained_variance.png
├── documentation/
│   ├── Feasibility_Report.docx      
│   ├── CIP_FS2026_104_Project_Documentation.docx  # Final report
│   └── CIP_team_agreement.docx
├── app.py                           
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Data Sources
- **Immobilier.ch**: Property listings scraped using Selenium and BeautifulSoup (March 2026)
- **Swiss Federal Statistical Office**: GDP per capita by canton (2022)

## Technologies Used
- **Python 3.11**
- **Web Scraping**: Selenium, BeautifulSoup4
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical Analysis**: Scikit-learn, SciPy

## Usage

### 1. Data Collection
```bash
# Scrape Immobilier.ch (all cantons, all pages)
python scripts/immobilier_allpages.py
```

### 2. Data Processing and Analysis
```bash
# Merge datasets, clean, and generate machine learning models
python scripts/data_processing.py

# Clean GDP data
python scripts/gdpdataset.py
```

## Data Transformation Steps
1. Dataset merging 
2. Missing data identification and handling
3. Data type validation and conversion
4. Value range verification
5. Outlier detection and treatment

## Individual Contributions

### Margarita Arias - [margarita.arias@stud.hslu.ch]
- Contribution 1
- Contribution 2
- Contribution 3

### Seda Dulger - [havva.duelger@stud.hslu.ch]
- Contribution 1
- Contribution 2
- Contribution 3

### Michael Ryan - [michael.ryan@stud.hslu.ch]
- Contribution 1
- Contribution 2
- Contribution 3

### Svenja Ryf - [svenja.ryf@stud.hslu.ch]
- Contribution 1
- Contribution 2
- Contribution 3

## AI Disclaimer
AI was used in this project to debug Python syntax errors, deepening classroom knowledge, and improving report paragraph construction. AI was <u>_**not**_</u> used to produce invented data, results, interpretations or report content. 

---
*Last updated: April 8 2026*
