#!/usr/bin/env python3
"""
Visualization Script for Energy Analytics Report
================================================

This script loads the final JSON report and generates a series of high-quality
visualizations to highlight the key findings.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# --- Configuration ---
REPORT_PATH = "output/reports/comprehensive_energy_report.json"
PLOTS_DIR = "output/plots"
TOP_N_COUNTRIES = 4  # Number of countries to show in comparison plots

# --- Plotting Style ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def plot_duck_curve_trends(data: dict, output_dir: Path):
    """Plots the trend of evening ramp rates for the top N countries."""
    print("  Generating Duck Curve Trend plot...")
    
    trends = {}
    for country, analysis in data.items():
        try:
            trends[country] = analysis['flexibility_requirements']['ramp_rate_trend']
        except (KeyError, TypeError):
            continue
            
    if not trends:
        print("    - No duck curve trend data to plot.")
        return

    # Sort by the magnitude of the trend and take the top N
    sorted_trends = sorted(trends.items(), key=lambda item: abs(item[1]), reverse=True)
    top_trends = dict(sorted_trends[:TOP_N_COUNTRIES])

    plt.figure()
    ax = sns.barplot(x=list(top_trends.keys()), y=list(top_trends.values()), palette="viridis")
    ax.set_title(f'Top {TOP_N_COUNTRIES} Countries by Grid Flexibility Demand (Evening Ramp Rate Trend)')
    ax.set_ylabel('Ramp Rate Trend (MW/hour per year)')
    ax.set_xlabel('Country')
    plt.tight_layout()
    plt.savefig(output_dir / "duck_curve_ramp_rate_trends.png", dpi=300)
    plt.close()


def plot_intermittency_comparison(data: dict, output_dir: Path):
    """Compares the intermittency of wind vs. solar generation."""
    print("  Generating Renewable Intermittency plot...")
    
    intermittency_data = []
    for country, analysis in data.items():
        for source, metrics in analysis.items():
            try:
                intermittency_data.append({
                    'country': country,
                    'source': source.capitalize(),
                    'volatility': metrics['ramp_analysis']['ramp_rate_volatility']
                })
            except (KeyError, TypeError):
                continue
    
    if not intermittency_data:
        print("    - No intermittency data to plot.")
        return

    df = pd.DataFrame(intermittency_data)
    
    # Select top N countries by max volatility
    top_countries = df.groupby('country')['volatility'].max().nlargest(TOP_N_COUNTRIES).index
    df_top = df[df['country'].isin(top_countries)]

    plt.figure()
    ax = sns.barplot(data=df_top, x='country', y='volatility', hue='source', palette="magma")
    ax.set_title(f'Renewable Generation Volatility in Top {TOP_N_COUNTRIES} Countries')
    ax.set_ylabel('Ramp Rate Volatility (MW/hour)')
    ax.set_xlabel('Country')
    plt.tight_layout()
    plt.savefig(output_dir / "renewable_intermittency_comparison.png", dpi=300)
    plt.close()


def plot_cluster_profiles(data: dict, output_dir: Path):
    """Plots the mean daily load profile for each identified cluster."""
    # We'll focus on Germany (DE) as the primary example
    country_code = 'DE'
    cluster_data = data.get(country_code)
    
    if not cluster_data:
        print(f"  - No clustering data found for {country_code}.")
        return
        
    print(f"  Generating Load Profile Clusters plot for {country_code}...")

    plt.figure()
    for name, cluster in cluster_data.items():
        try:
            percentage = cluster['percentage']
            profile = cluster['mean_profile']
            plt.plot(profile, label=f"{name.replace('_', ' ').title()} ({percentage:.1f}%)", lw=2.5)
        except (KeyError, TypeError):
            continue

    plt.title(f'Typical Daily Load Profiles Identified by K-Means Clustering ({country_code})')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Load (MW)')
    plt.xticks(ticks=range(0, 25, 2))
    plt.legend(title='Cluster (% of Days)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "load_profile_clusters.png", dpi=300)
    plt.close()


def plot_feature_importance(data: dict, output_dir: Path):
    """Plots the feature importances from the ML demand forecast model."""
    if not data or 'feature_importance' not in data:
        print("  - No ML feature importance data to plot.")
        return
        
    print("  Generating ML Feature Importance plot...")
    
    try:
        df = pd.DataFrame(data['feature_importance'])
        df = df.sort_values('importance', ascending=False).head(10) # Top 10 features
    except (KeyError, TypeError):
        print("    - Malformed feature importance data.")
        return

    plt.figure()
    ax = sns.barplot(data=df, x='importance', y='feature', palette='rocket')
    ax.set_title('Top 10 Feature Importances for Demand Forecast Model (DE)')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_dir / "ml_feature_importance.png", dpi=300)
    plt.close()


def main():
    """Main function to load data and generate all plots."""
    report_file = Path(REPORT_PATH)
    output_dir = Path(PLOTS_DIR)
    
    if not report_file.exists():
        print(f"Error: Report file not found at {REPORT_PATH}")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading comprehensive report...")
    with open(report_file, 'r') as f:
        report_data = json.load(f)
        
    print("Generating visualizations...")
    plot_duck_curve_trends(report_data.get('duck_curve_analysis', {}), output_dir)
    plot_intermittency_comparison(report_data.get('intermittency_analysis', {}), output_dir)
    plot_cluster_profiles(report_data.get('load_profile_clustering', {}), output_dir)
    plot_feature_importance(report_data.get('ml_demand_forecast', {}), output_dir)

    print(f"\nAll visualizations have been saved to the '{PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main() 