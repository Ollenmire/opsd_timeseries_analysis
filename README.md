# OPSD Time Series Analysis

This repo walks through a complete exploration of the Open Power System Data “Time Series” package. You’ll see how I handled real-world energy data, cleaned and transformed it, ran some deeper analytics, and then turned everything into four polished charts.

## What We’re Doing

1. **Grabbing the Data**  
   I start by downloading the latest OPSD zip file, checking its integrity, and unpacking it into `data/`. No manual clicks—everything runs from the notebook.

2. **Tidying and Validating**  
   Next, I align timestamps (including daylight-saving quirks), forward-fill any gaps under six hours, flag longer gaps, and clip any wildly impossible values (like negative loads or huge spikes).

3. **Building New Features**  
   To get more insight, I tag weekends and holidays, calculate rolling averages (24 h and 7 d), and compute each hour’s “renewable share” (wind + solar divided by load).

4. **Digging into the Numbers**  
   Here’s where the fun begins. I pull together country-by-year summaries, plot hourly profiles and heatmaps, trace how base-load has drifted since 2015, and chart renewables’ growing slice of the pie.

5. **Advanced Analysis & Reporting**  
   In one script (`src/advanced_analytics.py`), I crunch all these features and spit out a single JSON report (`output/reports/comprehensive_energy_report.json`) that captures every metric and insight.

6. **Final Portfolio-Ready Charts**  
   The final notebook reads that JSON and produces four publication-style PNGs in `output/figures/`—no interactive widgets, just clean static images with captions.

## Why It Matters

- **Grid Flexibility (Duck Curve):** See how ramp-up requirements vary by country—critical for folks planning new storage or demand-response programs.  
- **Renewable Intermittency:** Quantify solar’s predictability versus its volatility.  
- **Demand Forecasting:** A Random Forest model nails an R² over 0.94 using just last hour’s load and weekday features.  
- **Behavior Clusters:** K-Means identifies distinct daily patterns (e.g., winter workdays vs. summer weekends), which can guide targeted grid interventions.

## Repo Layout
````
opsd\_timeseries\_analysis/
├── data/                              # raw OPSD files (git-ignored)
├── notebooks/
│   ├── 01\_download.ipynb              # grabs and unzips the data
│   ├── 02\_clean\_explore.ipynb         # cleaning, feature engineering, EDA
│   ├── 04\_advanced\_insights.ipynb     # runs analytics script, writes JSON
│   └── 05\_final\_visualizations.ipynb  # loads JSON, saves final PNGs
├── src/
│   ├── opsd\_utils.py                  # data-loading & helper functions
│   └── advanced\_analytics.py          # builds the JSON report
├── output/
│   ├── figures/                       # final PNG charts
│   └── reports/
│       └── comprehensive\_energy\_report.json
├── environment.yml                    # conda setup
└── README.md                          # you’re reading it now

````

## Getting Started

1. **Clone it**  
   ```bash
   git clone <repo-URL>
   cd opsd_timeseries_analysis
   ```

2. **Set up your environment**

   ```bash
   conda env create -f environment.yml
   conda activate opsd-timeseries-analysis
   ```
3. **Run the notebooks in order**

   * `01_download.ipynb` → fetch data
   * `02_clean_explore.ipynb` → clean & EDA
   * `04_advanced_insights.ipynb` → generate JSON
   * `05_final_visualizations.ipynb` → export final charts

All the text commentary and interpretations land in the JSON report and are called out in each notebook; the static PNGs live in `output/figures/`.

## Key Findings

After running the full analysis pipeline on Germany (DE), four standalone charts landed in `output/figures/`. Here’s what each one tells us:

1. **Duck Curve Ramp Rate Trends**  
   ![Duck Curve Trends](output/duck_curve_ramp_rate_trends.png)  
   The familiar midday “dip-and-rise” in net load (the duck curve) is getting steeper over time. As solar capacity grows, evening ramp-up rates have climbed significantly—driving home the urgent need for faster-responding reserves or storage to keep the lights on.

2. **Renewable Intermittency Analysis**  
   ![Renewable Intermittency](output/renewable_intermittency_analysis.png)  
   Wind and solar outputs swing unpredictably hour to hour. Some countries (e.g. DE, ES) show much higher volatility and more frequent extreme ramp events than others, demonstrating exactly why grid operators lean on batteries or backup plants when the wind stops or a cloud passes.

3. **ML Demand Forecasting Results**  
   ![ML Forecasting](output/ml_demand_forecasting_results.png)  
   A Random Forest model nails short-term demand with R²≈0.94 using just lag features and day-of-week. Most of the scatter in the “Actual vs. Predicted” plot happens around peak hours, suggesting those are the toughest times to forecast, and where better real-time data could pay dividends.

4. **Load Profile Clustering**  
   ![Load Clustering](output/load_profile_clustering.png)  
   Daily consumption naturally splits into five clusters—think “winter workday,” “summer weekend,” etc. Each cluster has a distinct peak hour, valley hour, and daily range. Utilities can leverage these segments to tailor demand-response or tariff programs to the right customer groups.

> **Run it yourself:** all code lives in the `notebooks/` folder. If you want to verify any of these insights—or drill into another country or time window—just follow the **Getting Started** steps above and let the notebooks do the rest.


## Data Details

* **Source:** OPSD Time Series
* **Download URL:** [https://data.open-power-system-data.org/time\_series/2020-10-06/opsd-time\_series-2020-10-06.zip](https://data.open-power-system-data.org/time_series/2020-10-06/opsd-time_series-2020-10-06.zip)
* **License:** CC BY 4.0
* **Coverage:** 37 European countries, hourly resolution from 2015 onward
