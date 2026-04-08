# CIP_FS2026_104 - GDP and Apartment Rental Prices Across Swiss Cantons 

## Repository Structure
CIP_FS2026_104/
├── data/
│   ├── raw/                          # Original data files
│   │   ├── orijinal.xlsx            # Raw Immobilier.ch scraped data
│   │   └── gdp_percapita_canton_2022.xlsx  # GDP data by canton
│   └── processed/                    # Cleaned and merged datasets
│       ├── canton_gdp_2022_clean.csv
│       └── immobilier_all_cantons_allpages_snapshot_p20.csv
├── scripts/
│   ├── immobilier.py                # Initial scraper
│   ├── immobilier_allpages.py       # Full multi-page scraper with Selenium
│   ├── data_processing.py           # Data cleaning and transformation
│   ├── gdpdataset.py                # GDP data processing
│   ├── population_by_canton_2022.py # Population data integration
│   ├── model.py                     # Statistical analysis
│   ├── card_preview.py              # Visualization helper
│   └── rooms_location_preview.py    # Location-based analysis
├── Figures/
│   ├── figure_1_gdp_vs_log_price_per_m2.png
│   ├── figure_2_distribution_log_price_by_economic_and_structural_type.png
│   └── figure_3_r2_explained_variance.png
├── documentation/
│   ├── Feasibility_Report.docx      # Initial project proposal
│   ├── CIP_FS2026_104_Project_Documentation.docx  # Final report
│   └── CIP_team_agreement.docx
├── app.py                           # Main application entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

## Data Sources
- **Immobilier.ch**: Property listings scraped using Selenium and BeautifulSoup
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
# Scrape Immobilier.ch (all pages)
python scripts/immobilier_allpages.py
```

### 2. Data Processing and Analysis
```bash
# Clean and merge datasets and generate visualizations
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
- data_processing.py - Dataset merging and data type validation
- data_processing.py -Figure 2 code
- data_processing.py - Figure saving code
- GitHub Repository creation and structuring
- Report documentation (Data Acquisition section and EDA section)
- ReadMe

### Seda Dulger - [seda.dulger@stud.hslu.ch]
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

### Student 2 - [Name]
- Data cleaning and transformation (`data_processing.py`)
- GDP dataset integration (`gdpdataset.py`)
- Population data processing

### Student 3 - [Name]
- Statistical modeling (`model.py`)
- Visualization development (all figures)
- Final documentation and reporting

## Contact
For questions or collaboration:
- Student 1: [email]
- Student 2: [email]
- Student 3: [email]

---
*Last updated: April 2026*