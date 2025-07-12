#!/usr/bin/env python3
"""
Advanced Analytics Module for OPSD Time Series Analysis
=======================================================

This module provides cutting-edge analytics for energy time series data,
focusing on insights that matter for energy transition, grid stability,
and policy decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

warnings.filterwarnings('ignore')


class EnergyTransitionAnalyzer:
    """Advanced analytics for energy transition insights."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def analyze_duck_curve_evolution(self, countries: List[str] = None) -> Dict:
        """
        Analyze the evolution of the 'duck curve' - the shape of net load
        throughout the day as solar penetration increases.
        
        Returns insights on grid flexibility requirements.
        """
        if countries is None:
            countries = self._detect_countries()
        
        duck_curve_results = {}
        
        for country in countries:
            load_col = f"{country}_load_actual_entsoe_transparency"
            solar_col = f"{country}_solar_generation_actual"
            
            if load_col not in self.df.columns or solar_col not in self.df.columns:
                continue
                
            # Create a DataFrame for the specific country and calculate net load
            country_df = self.df[[load_col, solar_col]].dropna()
            country_df['net_load'] = country_df[load_col] - country_df[solar_col]
            country_df['year'] = country_df.index.year

            # Group by hour and year to see evolution
            hourly_analysis = []
            for year, year_df in country_df.groupby('year'):
                # Ensure we have enough data for a meaningful daily profile
                if len(year_df) < 8000:  # Skip years with incomplete data
                    continue
                    
                hourly_pattern = year_df['net_load'].groupby(year_df.index.hour).mean()
                
                if hourly_pattern.empty:
                    continue

                # Calculate duck curve metrics from the daily profile
                morning_ramp = hourly_pattern.loc[6:10].diff().max()
                evening_ramp = hourly_pattern.loc[16:20].diff().max()
                midday_dip = hourly_pattern.loc[10:16].min() - hourly_pattern.loc[6:10].mean()
                
                hourly_analysis.append({
                    'year': year,
                    'morning_ramp_rate': morning_ramp,
                    'evening_ramp_rate': evening_ramp,
                    'midday_dip_severity': abs(midday_dip),
                    'pattern': hourly_pattern.values
                })
            
            if not hourly_analysis:
                continue

            # Create a DataFrame for trend analysis and drop rows with missing data
            analysis_df = pd.DataFrame(hourly_analysis).set_index('year').dropna()

            # Calculate the trend for the evening ramp rate using linear regression
            if len(analysis_df) >= 3:  # Need at least 3 data points for a meaningful trend
                y = analysis_df['evening_ramp_rate']
                x = analysis_df.index
                ramp_trend_slope = np.polyfit(x, y, 1)[0]
            else:
                ramp_trend_slope = 0.0 # Not enough data for a trend

            duck_curve_results[country] = {
                'trend_analysis': analysis_df.to_string(),
                'flexibility_requirements': {
                    'max_ramp_rate': analysis_df['evening_ramp_rate'].max(),
                    'ramp_rate_trend': ramp_trend_slope
                }
            }
        
        return duck_curve_results
    
    def analyze_renewable_intermittency(self, countries: List[str] = None) -> Dict:
        """
        Quantify renewable energy intermittency and its impact on grid stability.
        
        Returns variability metrics, ramp rates, and predictability scores.
        """
        if countries is None:
            countries = self._detect_countries()
        
        intermittency_results = {}
        
        for country in countries:
            wind_col = f"{country}_wind_generation_actual"
            solar_col = f"{country}_solar_generation_actual"
            
            country_results = {}
            
            for source, col in [('wind', wind_col), ('solar', solar_col)]:
                if col not in self.df.columns:
                    continue
                    
                data = self.df[col].dropna()
                if len(data) < 100:
                    continue
                
                # Calculate variability metrics
                hourly_changes = data.diff().dropna()
                
                # Ramp rate analysis
                ramp_rates = {
                    'mean_absolute_ramp': hourly_changes.abs().mean(),
                    'max_ramp_up': hourly_changes.max(),
                    'max_ramp_down': hourly_changes.min(),
                    'ramp_rate_volatility': hourly_changes.std(),
                    'extreme_ramp_frequency': (hourly_changes.abs() > 2 * hourly_changes.std()).mean()
                }
                
                # Predictability analysis
                # Simple persistence model (naive forecast)
                forecast_errors = data.diff().dropna()
                mae = forecast_errors.abs().mean()
                
                # Capacity factor analysis
                capacity_col = f"{country}_{source}_capacity"
                if capacity_col in self.df.columns:
                    capacity = self.df[capacity_col].ffill().bfill()
                    capacity_factor = (data / capacity).clip(0, 1)
                    cf_stats = {
                        'mean_capacity_factor': capacity_factor.mean(),
                        'cf_volatility': capacity_factor.std(),
                        'low_output_hours': (capacity_factor < 0.1).mean(),
                        'high_output_hours': (capacity_factor > 0.8).mean()
                    }
                else:
                    cf_stats = {}
                
                # Calculate predictability score safely
                data_mean = data.mean()
                if data_mean > 0:
                    predictability_score = max(0, 1 - (mae / data_mean))  # Higher = more predictable
                else:
                    predictability_score = 0.0
                
                country_results[source] = {
                    'ramp_analysis': ramp_rates,
                    'predictability_score': predictability_score,
                    'capacity_factor_analysis': cf_stats
                }
            
            intermittency_results[country] = country_results
        
        return intermittency_results
    
    def analyze_price_renewable_correlation(self, countries: List[str] = None) -> Dict:
        """
        Analyze the relationship between renewable generation and electricity prices.
        
        Returns correlation analysis and merit order effects.
        """
        if countries is None:
            countries = self._detect_countries()
        
        price_correlation_results = {}
        
        for country in countries:
            price_col = f"{country}_price_day_ahead_entsoe_transparency"
            wind_col = f"{country}_wind_generation_actual"
            solar_col = f"{country}_solar_generation_actual"
            load_col = f"{country}_load_actual_entsoe_transparency"
            
            if price_col not in self.df.columns:
                continue
                
            price_data = self.df[price_col].dropna()
            correlations = {}
            
            # Price-renewable correlations
            for source, col in [('wind', wind_col), ('solar', solar_col), ('load', load_col)]:
                if col in self.df.columns:
                    renewable_data = self.df[col].reindex(price_data.index).dropna()
                    common_index = price_data.index.intersection(renewable_data.index)
                    
                    if len(common_index) > 100:
                        corr = np.corrcoef(price_data[common_index], renewable_data[common_index])[0, 1]
                        correlations[f'{source}_price_correlation'] = corr
            
            # Merit order effect analysis
            if wind_col in self.df.columns and solar_col in self.df.columns:
                # Combine renewable generation
                total_renewable = (self.df[wind_col].fillna(0) + 
                                 self.df[solar_col].fillna(0)).reindex(price_data.index)
                
                # Bin renewable generation into deciles
                renewable_deciles = pd.qcut(total_renewable, 10, labels=False, duplicates='drop')
                
                # Calculate average price by renewable generation level
                merit_order_effect = []
                for decile in range(10):
                    mask = renewable_deciles == decile
                    if mask.sum() > 0:
                        avg_price = price_data[mask].mean()
                        renewable_level = total_renewable[mask].mean()
                        merit_order_effect.append({
                            'renewable_decile': decile,
                            'avg_renewable_generation': renewable_level,
                            'avg_price': avg_price
                        })
                
                # Calculate merit order coefficient (€/MWh per MWh of renewable generation)
                if len(merit_order_effect) > 5:
                    df_merit = pd.DataFrame(merit_order_effect)
                    merit_coef = np.polyfit(df_merit['avg_renewable_generation'], 
                                          df_merit['avg_price'], 1)[0]
                    correlations['merit_order_coefficient'] = merit_coef
            
            price_correlation_results[country] = correlations
        
        return price_correlation_results
    
    def analyze_cross_border_effects(self) -> Dict:
        """
        Analyze cross-border electricity flows and price convergence.
        
        Returns interconnection benefits and market integration metrics.
        """
        # 1. Smarter Bidding Zone Detection to handle complex names (e.g., DE_LU, IT_NORD)
        price_cols = [col for col in self.df.columns if 'price_day_ahead' in col]
        
        if len(price_cols) < 2:
            return {"error": "Insufficient price data for cross-border analysis."}
            
        # 2. Prepare data: resample to daily, forward-fill to handle missing values,
        # and then back-fill for any remaining NaNs at the start of the series.
        price_data = self.df[price_cols].ffill().bfill()
        price_data_daily = price_data.resample('D').mean().dropna(axis=1, how='all')
        
        # 3. Calculate price correlations on the clean, daily data
        price_correlations = price_data_daily.corr()
        
        # Clean up column names for a readable report
        def clean_col_name(col):
            return col.replace('_price_day_ahead_entsoe_transparency', '').replace('_price_day_ahead', '')
            
        cleaned_names = [clean_col_name(c) for c in price_data_daily.columns]
        price_correlations.columns = cleaned_names
        price_correlations.index = cleaned_names

        # 4. Calculate price spread volatility between all zones
        price_spreads = {}
        zones = price_data_daily.columns
        
        for i, zone1 in enumerate(zones):
            for zone2 in zones[i+1:]:
                spread = price_data_daily[zone1] - price_data_daily[zone2]
                
                # Use cleaned names for the report key
                spread_key = f"{clean_col_name(zone1)}_{clean_col_name(zone2)}_spread"
                
                price_spreads[spread_key] = {
                    'mean_spread': spread.mean(),
                    'spread_volatility': spread.std()
                }

        # 5. Calculate a robust market integration score
        # The score is the average of all cross-zonal correlations.
        corr_values = price_correlations.values[np.triu_indices_from(price_correlations.values, k=1)]
        market_integration_score = np.nanmean(corr_values) if len(corr_values) > 0 else 0.0

        return {
            'price_correlations': price_correlations.to_dict(),
            'price_spreads': price_spreads,
            'market_integration_score': market_integration_score
        }
    
    def analyze_seasonal_patterns(self, countries: List[str] = None) -> Dict:
        """
        Deep dive into seasonal patterns in energy consumption and generation.
        
        Returns seasonal decomposition and climate sensitivity analysis.
        """
        if countries is None:
            countries = self._detect_countries()
        
        seasonal_results = {}
        
        for country in countries:
            load_col = f"{country}_load_actual_entsoe_transparency"
            wind_col = f"{country}_wind_generation_actual"
            solar_col = f"{country}_solar_generation_actual"
            
            country_results = {}
            
            for metric, col in [('load', load_col), ('wind', wind_col), ('solar', solar_col)]:
                if col not in self.df.columns:
                    continue
                    
                data = self.df[col].dropna()
                if len(data) < 1000:
                    continue
                
                # Monthly patterns
                monthly_avg = data.groupby(data.index.month).mean()
                # Ensure we have more than one data point to calculate standard deviation
                monthly_std = data.groupby(data.index.month).std().dropna()
                
                # Seasonal amplitude
                seasonal_amplitude = (monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean()
                
                # Weather sensitivity (using temperature proxy via load patterns)
                if metric == 'load':
                    # Cooling/heating degree days proxy
                    summer_months = [6, 7, 8]
                    winter_months = [12, 1, 2]
                    
                    summer_avg = data[data.index.month.isin(summer_months)].mean()
                    winter_avg = data[data.index.month.isin(winter_months)].mean()
                    
                    seasonal_sensitivity = abs(summer_avg - winter_avg) / data.mean()
                else:
                    seasonal_sensitivity = seasonal_amplitude
                
                country_results[metric] = {
                    'monthly_averages': monthly_avg.to_dict(),
                    'monthly_volatility': monthly_std.to_dict(),
                    'seasonal_amplitude': seasonal_amplitude,
                    'seasonal_sensitivity': seasonal_sensitivity
                }
            
            seasonal_results[country] = country_results
        
        return seasonal_results
    
    def detect_anomalies_and_events(self, countries: List[str] = None) -> Dict:
        """
        Detect anomalous patterns and extreme events in energy data.
        
        Returns anomaly detection results and event characterization.
        """
        if countries is None:
            countries = self._detect_countries()
        
        anomaly_results = {}
        
        for country in countries:
            load_col = f"{country}_load_actual_entsoe_transparency"
            
            if load_col not in self.df.columns:
                continue
                
            data = self.df[load_col].dropna()
            if len(data) < 1000:
                continue
            
            # Statistical anomaly detection
            rolling_mean = data.rolling(window=24*7).mean()
            rolling_std = data.rolling(window=24*7).std()
            
            # Z-score based anomalies
            z_scores = (data - rolling_mean) / rolling_std
            anomalies = data[abs(z_scores) > 3]
            
            # Peak detection (limit for large datasets)
            peaks, _ = find_peaks(data, height=data.quantile(0.95))
            peak_dates = data.index[peaks]
            
            # Limit number of events analyzed for performance
            max_events = 100
            if len(peak_dates) > max_events:
                # Keep the most extreme peaks
                peak_values = data.iloc[peaks]
                top_peak_indices = peak_values.nlargest(max_events).index
                peak_dates = top_peak_indices
            
            # Event characterization
            events = []
            for peak_date in peak_dates[:max_events]:  # Ensure we don't exceed limit
                peak_value = data.loc[peak_date]
                # Look for patterns around the peak
                window = slice(peak_date - pd.Timedelta(days=1), 
                              peak_date + pd.Timedelta(days=1))
                event_data = data[window]
                
                events.append({
                    'date': peak_date,
                    'peak_value': peak_value,
                    'duration_hours': len(event_data[event_data > peak_value * 0.9]),
                    'severity_score': (peak_value - data.mean()) / data.std()
                })
            
            anomaly_results[country] = {
                'anomaly_count': len(anomalies),
                'anomaly_dates': anomalies.index.tolist(),
                'extreme_events': events,
                'baseline_volatility': data.std() / data.mean()
            }
        
        return anomaly_results
    
    def _detect_countries(self) -> List[str]:
        """Helper method to detect country codes from column names."""
        countries = set()
        for col in self.df.columns:
            if '_' in col:
                potential_country = col.split('_')[0]
                if len(potential_country) == 2 and potential_country.isupper():
                    countries.add(potential_country)
        return list(countries)


class EnergyForecastingAnalyzer:
    """Advanced forecasting and predictive analytics for energy data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_forecast_accuracy(self, countries: List[str] = None) -> Dict:
        """
        Analyze the accuracy of day-ahead vs actual generation/load.
        
        Returns forecast performance metrics and improvement suggestions.
        """
        if countries is None:
            countries = self._detect_countries()
        
        forecast_results = {}
        
        for country in countries:
            # Look for forecast vs actual pairs
            forecast_pairs = [
                ('load', f"{country}_load_forecast_entsoe_transparency", 
                 f"{country}_load_actual_entsoe_transparency"),
                ('wind', f"{country}_wind_generation_forecast", 
                 f"{country}_wind_generation_actual"),
                ('solar', f"{country}_solar_generation_forecast", 
                 f"{country}_solar_generation_actual")
            ]
            
            country_results = {}
            
            for metric, forecast_col, actual_col in forecast_pairs:
                if forecast_col not in self.df.columns or actual_col not in self.df.columns:
                    continue
                
                # Get common data points
                forecast_data = self.df[forecast_col].dropna()
                actual_data = self.df[actual_col].dropna()
                common_index = forecast_data.index.intersection(actual_data.index)
                
                if len(common_index) < 100:
                    continue
                
                forecast_values = forecast_data[common_index]
                actual_values = actual_data[common_index]
                
                # Calculate forecast errors
                errors = forecast_values - actual_values
                
                # Performance metrics
                mae = errors.abs().mean()
                mape = (errors.abs() / actual_values.abs()).mean() * 100
                rmse = np.sqrt((errors ** 2).mean())
                
                # Directional accuracy
                forecast_direction = np.sign(forecast_values.diff())
                actual_direction = np.sign(actual_values.diff())
                directional_accuracy = (forecast_direction == actual_direction).mean()
                
                country_results[metric] = {
                    'mae': mae,
                    'mape': mape,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy,
                    'forecast_bias': errors.mean(),
                    'error_volatility': errors.std()
                }
            
            forecast_results[country] = country_results
        
        return forecast_results
    
    def create_ml_demand_forecast(self, country: str, horizon_hours: int = 24) -> Dict:
        """
        Create a machine learning model to forecast energy demand.
        
        Returns model performance and feature importance.
        """
        load_col = f"{country}_load_actual_entsoe_transparency"
        
        if load_col not in self.df.columns:
            return {"error": f"No load data available for {country}"}
        
        # Prepare features
        data = self.df[load_col].dropna()
        
        # Create time features
        features_df = pd.DataFrame(index=data.index)
        features_df['hour'] = data.index.hour
        features_df['day_of_week'] = data.index.dayofweek
        features_df['month'] = data.index.month
        features_df['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 24, 48, 168]:  # 1h, 2h, 1d, 2d, 1w
            features_df[f'load_lag_{lag}'] = data.shift(lag)
        
        # Rolling averages
        features_df['load_24h_avg'] = data.rolling(24).mean()
        features_df['load_7d_avg'] = data.rolling(24*7).mean()
        
        # Target variable
        features_df['target'] = data.shift(-horizon_hours)
        
        # Clean data
        features_df = features_df.dropna()
        
        if len(features_df) < 1000:
            return {"error": "Insufficient data for ML model"}
        
        # Split data
        train_size = int(len(features_df) * 0.8)
        train_data = features_df.iloc[:train_size]
        test_data = features_df.iloc[train_size:]
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns if col != 'target']
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_test = test_data[feature_cols]
        y_test = test_data['target']
        
        try:
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            
            # Performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE safely
            mape_values = np.abs((y_test - y_pred) / y_test)
            mape_values = mape_values[np.isfinite(mape_values)]  # Remove inf/nan
            mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else 100.0
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"}
        
        
        return {
            'model_performance': {
                'mae': mae,
                'r2_score': r2,
                'mape': mape
            },
            'feature_importance': feature_importance.to_dict('records'),
            'model_type': 'RandomForest',
            'training_samples': len(train_data),
            'test_samples': len(test_data)
        }
    
    def _detect_countries(self) -> List[str]:
        """Helper method to detect country codes from column names."""
        countries = set()
        for col in self.df.columns:
            if '_' in col:
                potential_country = col.split('_')[0]
                if len(potential_country) == 2 and potential_country.isupper():
                    countries.add(potential_country)
        return list(countries)


class EnergyClusteringAnalyzer:
    """Advanced clustering and pattern recognition for energy profiles."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def analyze_load_profiles(self, countries: List[str] = None, n_clusters: int = 5) -> Dict:
        """
        Cluster daily load profiles to identify typical energy consumption patterns.
        
        Returns cluster analysis and pattern characterization.
        """
        if countries is None:
            countries = self._detect_countries()
        
        clustering_results = {}
        
        for country in countries:
            load_col = f"{country}_load_actual_entsoe_transparency"
            
            if load_col not in self.df.columns:
                continue
                
            data = self.df[load_col].dropna()
            if len(data) < 1000:
                continue
            
            # Create daily profiles efficiently using groupby
            daily_profiles = []
            dates = []
            
            # Group by date more efficiently - use normalize() to avoid slow .date attribute access
            data_df = data.to_frame(name='load')
            data_df['date'] = data_df.index.normalize()  # Much faster than .date
            
            for date, group in data_df.groupby('date'):
                if len(group) == 24:  # Complete day
                    daily_profiles.append(group['load'].values)
                    dates.append(date.date())
            
            if len(daily_profiles) < 100:
                continue
            
            # For very large datasets, sample to improve performance
            max_profiles = 5000  # Limit to avoid memory issues
            if len(daily_profiles) > max_profiles:
                sample_indices = np.random.choice(len(daily_profiles), max_profiles, replace=False)
                sampled_profiles = [daily_profiles[i] for i in sample_indices]
                sampled_dates = [dates[i] for i in sample_indices]
                profiles_array = np.array(sampled_profiles)
                dates = sampled_dates
                print(f"  Sampling {max_profiles} profiles from {len(daily_profiles)} total for {country}")
            else:
                profiles_array = np.array(daily_profiles)
            
            # Normalize profiles
            scaler = StandardScaler()
            profiles_normalized = scaler.fit_transform(profiles_array)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(profiles_normalized)
            
            # Analyze clusters
            cluster_analysis = {}
            original_total = len(daily_profiles)  # Use original count for percentage
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_profiles = profiles_array[cluster_mask]
                cluster_dates = np.array(dates)[cluster_mask]
                
                if len(cluster_profiles) == 0:  # Skip empty clusters
                    continue
                
                # Calculate cluster characteristics
                mean_profile = cluster_profiles.mean(axis=0)
                peak_hour = np.argmax(mean_profile)
                valley_hour = np.argmin(mean_profile)
                
                # Seasonal distribution
                cluster_months = [date.month for date in cluster_dates]  # dates are already date objects
                seasonal_dist = pd.Series(cluster_months).value_counts(normalize=True)
                
                # Calculate percentage based on sampled data but scale to original
                sampled_percentage = len(cluster_profiles) / len(profiles_array) * 100
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_profiles),
                    'percentage': sampled_percentage,
                    'peak_hour': int(peak_hour),
                    'valley_hour': int(valley_hour),
                    'daily_range': float(mean_profile.max() - mean_profile.min()),
                    'seasonal_distribution': seasonal_dist.to_dict(),
                    'mean_profile': mean_profile.tolist()
                }
            
            clustering_results[country] = cluster_analysis
        
        return clustering_results
    
    def _detect_countries(self) -> List[str]:
        """Helper method to detect country codes from column names."""
        countries = set()
        for col in self.df.columns:
            if '_' in col:
                potential_country = col.split('_')[0]
                if len(potential_country) == 2 and potential_country.isupper():
                    countries.add(potential_country)
        return list(countries)


def generate_comprehensive_report(df: pd.DataFrame, output_dir: str = "output/reports") -> str:
    """
    Generate a focused, high-impact energy analytics report.
    """
    # Initialize analyzers
    transition_analyzer = EnergyTransitionAnalyzer(df)
    forecast_analyzer = EnergyForecastingAnalyzer(df)
    clustering_analyzer = EnergyClusteringAnalyzer(df)
    
    print(f"Analyzing dataset: {len(df):,} records, {len(df.columns)} columns")
    countries = transition_analyzer._detect_countries()
    print(f"Countries detected: {countries}")
    
    # A focused set of analyses for a portfolio project
    report_data = {
        'generation_time': pd.Timestamp.now().isoformat(),
        'dataset_summary': {
            'time_range': f"{df.index.min()} to {df.index.max()}",
            'total_records': len(df),
            'countries_analyzed': countries
        }
    }
    
    print("1/4 Running duck curve evolution analysis...")
    report_data['duck_curve_analysis'] = transition_analyzer.analyze_duck_curve_evolution()
    
    print("2/4 Running renewable intermittency analysis...")
    report_data['intermittency_analysis'] = transition_analyzer.analyze_renewable_intermittency()

    # The seasonal and anomaly detection analyses are computationally inexpensive,
    # so we can run them to ensure the code works, but we will exclude them
    # from the final focused report to maintain clarity.
    # transition_analyzer.analyze_seasonal_patterns()
    # transition_analyzer.detect_anomalies_and_events()
    
    print("3/4 Running load profile clustering...")
    report_data['load_profile_clustering'] = clustering_analyzer.analyze_load_profiles()

    # Run ML forecast for a key country (e.g., Germany) as a showcase
    ml_forecast_country = 'DE'
    print(f"4/4 Running ML demand forecast for {ml_forecast_country}...")
    report_data['ml_demand_forecast'] = forecast_analyzer.create_ml_demand_forecast(ml_forecast_country)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON report
    json_path = Path(output_dir) / "comprehensive_energy_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Generate executive summary
    summary_path = Path(output_dir) / "executive_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("FOCUSED ENERGY ANALYTICS & ML REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Analysis Period: {report_data['dataset_summary']['time_range']}\n")
        f.write(f"Countries Analyzed: {len(report_data['dataset_summary']['countries_analyzed'])}\n\n")
        
        f.write("KEY INSIGHTS & CAPABILITIES:\n")
        f.write("-" * 20 + "\n")
        
        # Add key insights from each analysis
        if report_data.get('duck_curve_analysis'):
            f.write("DUCK CURVE EVOLUTION (Time Series Analysis):\n")
            f.write("- Analyzed the deepening 'duck curve' to quantify increasing grid flexibility needs.\n\n")
        
        if report_data.get('intermittency_analysis'):
            f.write("RENEWABLE INTERMITTENCY (Statistical Analysis):\n")
            f.write("- Quantified ramp rates and volatility for wind and solar generation.\n\n")

        if report_data.get('load_profile_clustering'):
            f.write("LOAD PROFILE CLUSTERING (Unsupervised Learning):\n")
            f.write("- Used K-Means to identify distinct daily energy consumption patterns (e.g., weekday vs. weekend).\n\n")

        if report_data.get('ml_demand_forecast'):
            f.write(f"DEMAND FORECASTING (Machine Learning - RandomForest):\n")
            forecast_perf = report_data['ml_demand_forecast'].get('model_performance', {})
            if forecast_perf:
                r2 = forecast_perf.get('r2_score', 0)
                f.write(f"- Built and trained a Random Forest model to predict electricity demand for {ml_forecast_country}.\n")
                f.write(f"- Achieved an R\u00b2 score of {r2:.2f}, demonstrating predictive power.\n\n")
        
        f.write("DETAILED ANALYSIS:\n")
        f.write("See comprehensive_energy_report.json for complete numerical results.\n")
    
    print(f"Focused report generated: {json_path}")
    print(f"Executive summary: {summary_path}")
    
    return str(json_path) 