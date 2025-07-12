"""
OPSD Time Series Analysis Utilities

This module provides helper functions for downloading, cleaning, and analyzing
the Open Power System Data (OPSD) Time Series dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import os
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import holidays
import pytz
from datetime import datetime, timedelta


def get_latest_opsd_version() -> str:
    """
    Auto-discover the latest OPSD Time Series version from the catalogue page.
    
    Returns:
        Latest version string (e.g., "2020-10-06")
    """
    import re
    
    try:
        catalogue_url = "https://data.open-power-system-data.org/time_series/"
        response = requests.get(catalogue_url, timeout=10)
        response.raise_for_status()
        
        # Look for version patterns (corrected based on actual HTML structure)
        patterns = [
            r'(\d{4}-\d{2}-\d{2})\s+\(latest\)',  # Works based on debug
            r'version\s+(\d{4}-\d{2}-\d{2})',
            r'"version":\s*"(\d{4}-\d{2}-\d{2})"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.text)
            if match:
                version = match.group(1)
                print(f"Auto-discovered latest OPSD version: {version}")
                return version
        
        print("Could not auto-discover version, using fallback")
        return "2020-10-06"  # Known working fallback
            
    except Exception as e:
        print(f"Error auto-discovering version: {e}, using fallback")
        return "2020-10-06"  # Known working fallback


def download_opsd_data(data_dir: str = "data") -> str:
    """
    Download the latest OPSD Time Series dataset with auto-discovery.
    
    Args:
        data_dir: Directory to save the downloaded data
        
    Returns:
        Path to the downloaded zip file
    """
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Auto-discover the latest version
    latest_version = get_latest_opsd_version()
    
    # Build URL for the latest version (corrected format)
    url = f"https://data.open-power-system-data.org/time_series/opsd-time_series-{latest_version}.zip"
    zip_path = os.path.join(data_dir, "time_series.zip")
    
    print(f"Downloading OPSD Time Series data from: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {zip_path}")
        return zip_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from auto-discovered URL: {e}")
        
        # Try DOI redirect as secondary fallback (future-proof)
        try:
            doi_url = "https://doi.org/10.25832/time_series/2020-10-06"
            print(f"Trying DOI redirect: {doi_url}")
            
            # Follow redirects to get the actual download URL
            doi_response = requests.get(doi_url, allow_redirects=True, timeout=10)
            if doi_response.status_code == 200:
                # Use the corrected URL format
                doi_download_url = f"https://data.open-power-system-data.org/time_series/opsd-time_series-{latest_version}.zip"
                
                response = requests.get(doi_download_url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded using DOI redirect: {zip_path}")
                return zip_path
        except Exception as doi_error:
            print(f"DOI redirect also failed: {doi_error}")
        
        # Final fallback to known working URL (corrected format)
        fallback_url = "https://data.open-power-system-data.org/time_series/opsd-time_series-2020-10-06.zip"
        print(f"Trying final fallback URL: {fallback_url}")
        
        response = requests.get(fallback_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded using final fallback: {zip_path}")
        return zip_path


def extract_opsd_data(zip_path: str, data_dir: str = "data") -> List[str]:
    """
    Extract OPSD Time Series zip file.
    
    Args:
        zip_path: Path to the zip file
        data_dir: Directory to extract files to
        
    Returns:
        List of extracted CSV file paths
    """
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.csv'):
                zip_ref.extract(file_info, data_dir)
                extracted_files.append(os.path.join(data_dir, file_info.filename))
                print(f"Extracted: {file_info.filename}")
    
    return extracted_files


def load_opsd_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load and combine OPSD Time Series data from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Combined DataFrame with all time series data
    """
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv') and 'time_series' in file:
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        raise FileNotFoundError(f"No time series CSV files found in {data_dir}")
    
    # Load the main time series file (usually the largest one)
    main_file = max(csv_files, key=os.path.getsize)
    print(f"Loading main time series file: {main_file}")
    
    df = pd.read_csv(main_file, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def clean_timestamps(df: pd.DataFrame, target_tz: str = 'Europe/Berlin') -> pd.DataFrame:
    """
    Clean and standardize timestamps, handling DST transitions.
    
    Args:
        df: DataFrame with datetime index
        target_tz: Target timezone for conversion
        
    Returns:
        DataFrame with cleaned timestamps
    """
    df_clean = df.copy()
    
    # Convert to target timezone if not already timezone-aware
    if df_clean.index.tz is None:
        # If naive datetime, assume UTC first
        df_clean.index = df_clean.index.tz_localize('UTC')
    
    tz = pytz.timezone(target_tz)
    df_clean.index = df_clean.index.tz_convert(tz)
    
    # Handle DST transitions by forward-filling duplicated timestamps
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    
    # Ensure chronological order
    df_clean = df_clean.sort_index()
    
    print(f"Cleaned timestamps. Shape: {df_clean.shape}")
    return df_clean


def fill_missing_values(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    """
    Fill missing values using forward-fill for gaps < max_gap_hours.
    
    Args:
        df: DataFrame with time series data
        max_gap_hours: Maximum gap in hours to forward-fill
        
    Returns:
        DataFrame with filled values and gap flags
    """
    df_filled = df.copy()
    
    # Create a column to track long gaps
    df_filled['long_gap_flag'] = False
    
    for column in df_filled.columns:
        if column == 'long_gap_flag':
            continue
            
        # Find gaps longer than max_gap_hours
        missing_mask = df_filled[column].isna()
        
        if missing_mask.any():
            # Group consecutive missing values
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()
            
            for group_id in missing_groups[missing_mask].unique():
                group_mask = (missing_groups == group_id) & missing_mask
                gap_start = df_filled.index[group_mask].min()
                gap_end = df_filled.index[group_mask].max()
                gap_duration = (gap_end - gap_start).total_seconds() / 3600
                
                if gap_duration > max_gap_hours:
                    # Mark as long gap
                    df_filled.loc[group_mask, 'long_gap_flag'] = True
                else:
                    # Forward-fill short gaps
                    df_filled[column] = df_filled[column].ffill()
    
    print(f"Filled missing values. Long gaps flagged: {df_filled['long_gap_flag'].sum()}")
    return df_filled


def cap_outliers(df: pd.DataFrame, sigma_threshold: float = 5.0) -> pd.DataFrame:
    """
    Cap extreme outliers using sigma threshold.
    
    Args:
        df: DataFrame with time series data
        sigma_threshold: Number of standard deviations for outlier detection
        
    Returns:
        DataFrame with capped outliers
    """
    df_capped = df.copy()
    
    numeric_cols = df_capped.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in ['long_gap_flag']:
            continue
            
        # Calculate statistics
        mean_val = df_capped[col].mean()
        std_val = df_capped[col].std()
        
        # Set bounds
        lower_bound = mean_val - sigma_threshold * std_val
        upper_bound = mean_val + sigma_threshold * std_val
        
        # Handle negative values (especially for load data)
        if 'load' in col.lower():
            lower_bound = max(lower_bound, 0)
        
        # Cap outliers
        outliers_mask = (df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            df_capped.loc[df_capped[col] < lower_bound, col] = lower_bound
            df_capped.loc[df_capped[col] > upper_bound, col] = upper_bound
            print(f"Capped {n_outliers} outliers in {col}")
    
    return df_capped


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features including weekday/weekend and holiday flags.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with additional time features
    """
    df_features = df.copy()
    
    # Basic time features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    
    # Weekend flag
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Holiday flags for major European countries
    countries = ['DE', 'FR', 'GB', 'IT', 'ES', 'NL', 'BE', 'AT', 'CH', 'PL']
    
    for country in countries:
        try:
            country_holidays = holidays.country_holidays(country)
            # Convert datetime index to date for comparison
            date_index = pd.to_datetime(df_features.index).date
            df_features[f'is_holiday_{country}'] = pd.Series(
                date_index, index=df_features.index
            ).isin(country_holidays).astype(int)
        except Exception as e:
            print(f"Warning: Could not load holidays for {country}: {e}")
    
    print(f"Created time features. Shape: {df_features.shape}")
    return df_features


def create_rolling_features(df: pd.DataFrame, 
                          load_cols: List[str] = None, 
                          gen_cols: List[str] = None) -> pd.DataFrame:
    """
    Create rolling mean features for load and generation data.
    
    Args:
        df: DataFrame with time series data
        load_cols: List of load column names
        gen_cols: List of generation column names
        
    Returns:
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    
    # Auto-detect load and generation columns if not provided
    if load_cols is None:
        load_cols = [col for col in df.columns if 'load' in col.lower()]
    
    if gen_cols is None:
        gen_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['wind', 'solar', 'generation', 'pv'])]
    
    # Create rolling features
    for col in load_cols + gen_cols:
        if col in df_rolling.columns:
            # 24-hour rolling mean
            df_rolling[f'{col}_24h_mean'] = df_rolling[col].rolling(window=24).mean()
            
            # 7-day rolling mean
            df_rolling[f'{col}_7d_mean'] = df_rolling[col].rolling(window=24*7).mean()
    
    print(f"Created rolling features. Shape: {df_rolling.shape}")
    return df_rolling


def calculate_renewable_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate renewable energy share as percentage of total load.
    
    Args:
        df: DataFrame with load and renewable generation data
        
    Returns:
        DataFrame with renewable share calculations
    """
    df_renewable = df.copy()
    
    # Find renewable and load columns
    wind_cols = [col for col in df.columns if 'wind' in col.lower()]
    solar_cols = [col for col in df.columns if 'solar' in col.lower() or 'pv' in col.lower()]
    load_cols = [col for col in df.columns if 'load' in col.lower()]
    
    # Calculate renewable share by country/region
    for load_col in load_cols:
        # Extract country/region identifier
        country_code = load_col.split('_')[0] if '_' in load_col else load_col
        
        # Find matching renewable columns
        matching_wind = [col for col in wind_cols if country_code in col]
        matching_solar = [col for col in solar_cols if country_code in col]
        
        if matching_wind or matching_solar:
            # Sum renewable generation
            renewable_sum = 0
            if matching_wind:
                renewable_sum += df_renewable[matching_wind].sum(axis=1)
            if matching_solar:
                renewable_sum += df_renewable[matching_solar].sum(axis=1)
            
            # Calculate percentage
            load_data = df_renewable[load_col]
            renewable_share = (renewable_sum / load_data * 100).clip(0, 100)
            
            df_renewable[f'{country_code}_renewable_share'] = renewable_share
    
    print(f"Calculated renewable share. Shape: {df_renewable.shape}")
    return df_renewable


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics by country and year.
    
    Args:
        df: DataFrame with time series data
        
    Returns:
        DataFrame with summary statistics
    """
    # Prepare data for grouping
    df_stats = df.copy()
    
    # Get country codes from column names
    load_cols = [col for col in df.columns if 'load' in col.lower()]
    countries = list(set([col.split('_')[0] for col in load_cols if '_' in col]))
    
    summary_data = []
    
    for country in countries:
        country_cols = [col for col in df.columns if col.startswith(country)]
        load_col = [col for col in country_cols if 'load' in col.lower()]
        
        if not load_col:
            continue
            
        load_col = load_col[0]
        
        # Group by year
        yearly_data = df_stats.groupby(df_stats.index.year).agg({
            load_col: ['mean', 'max', 'min', 'std'],
        }).round(2)
        
        yearly_data.columns = ['_'.join(col).strip() for col in yearly_data.columns]
        yearly_data['country'] = country
        yearly_data['year'] = yearly_data.index
        
        summary_data.append(yearly_data)
    
    if summary_data:
        summary_df = pd.concat(summary_data, ignore_index=True)
        print(f"Created summary statistics for {len(countries)} countries")
        return summary_df
    else:
        print("No summary statistics could be created")
        return pd.DataFrame()


def save_figure(fig, filename: str, output_dir: str = "output/figures", 
                dpi: int = 300, bbox_inches: str = 'tight') -> str:
    """
    Save matplotlib figure to the specified directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Filename for the saved figure
        output_dir: Directory to save the figure
        dpi: Resolution for the saved figure
        bbox_inches: Bounding box setting
        
    Returns:
        Full path to the saved figure
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Add .png extension if not present
    if not filename.endswith('.png'):
        filename += '.png'
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    
    print(f"Saved figure: {filepath}")
    return filepath


def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        # Fallback to a basic style if seaborn style not available
        plt.style.use('default')
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16 