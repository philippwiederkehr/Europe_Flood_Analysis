#!/usr/bin/env python3
"""
Make tables comparing exposure across different SSP scenarios.

Usage (example):
    python make_scenario_comparison_tables.py --year 2050 --rp RP100

Creates tables showing:
1. Absolute Number of SSP2 of Total People Exposed per country, % difference if SSP1 was used, % difference if SSP3 was used
2. Ratio of Total People Exposed at SSP2 to whole population per country, % difference if SSP1 was used, % difference if SSP3 was used
3. Absolute Number of SSP2 of Poorest 20% People Exposed per country, % difference if SSP1 was used, % difference if SSP3 was used
4. Ratio of Poorest 20% People Exposed at SSP2 to whole poorest 20% population per country, % difference if SSP1 was used, % difference if SSP3 was used
"""
import argparse
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hard-wired roots (same style as other make_fig files)
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

# Set up plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = False

def build_paths(ssp: str, year: int, rp: str):
    """Build paths to the analysis results files"""
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def calculate_exposure_differences(year: int, rp: str):
    """
    Calculate exposure differences between scenarios
    
    Parameters:
    -----------
    year : int
        Year to analyze
    rp : str
        Return period (e.g., 'RP100')
    """
    print(f"\nProcessing data for year {year} and return period {rp}")
    
    # Define scenarios
    base_scenario = 'SSP2'
    comparison_scenarios = ['SSP1', 'SSP3']
    
    # Dictionary to store results for each scenario
    scenario_results = {}
    
    # Process each scenario
    for scenario in [base_scenario] + comparison_scenarios:
        print(f"\nLoading data for {scenario}...")
        geojson_path = build_paths(scenario, year, rp)
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON not found for {scenario}, {year}, {rp}: {geojson_path}")
            continue
            
        # Load data
        gdf = gpd.read_file(geojson_path)
        print(f"Successfully loaded data for {scenario}")
        
        # Aggregate at country level (NUTS-0)
        gdf['nuts0_id'] = gdf['nuts_id'].str.slice(0, 2)
        
        country_agg = gdf.groupby('nuts0_id').agg({
            'total_population': 'sum',
            'total_affected': 'sum',
            'poorest_20_population': 'sum',
            'poorest_20_affected': 'sum'
        }).reset_index()
        
        # Calculate ratios
        country_agg['total_exposure_ratio'] = (country_agg['total_affected'] / country_agg['total_population']) * 100
        country_agg['poorest20_exposure_ratio'] = (country_agg['poorest_20_affected'] / country_agg['poorest_20_population']) * 100
        
        scenario_results[scenario] = country_agg
        print(f"Processed {len(country_agg)} countries for {scenario}")
    
    # Create tables
    create_exposure_tables(scenario_results, base_scenario, comparison_scenarios, year, rp)

def create_exposure_tables(scenario_results, base_scenario, comparison_scenarios, year, rp):
    """Create tables comparing exposure across scenarios"""
    
    # Initialize DataFrames for different metrics
    total_exposure = pd.DataFrame()
    total_exposure_ratio = pd.DataFrame()
    poorest20_exposure = pd.DataFrame()
    poorest20_exposure_ratio = pd.DataFrame()
    
    # Get base scenario data
    base_data = scenario_results[base_scenario]
    
    print("\nCalculating differences between scenarios...")
    
    # Process each country
    for _, country_row in base_data.iterrows():
        country_code = country_row['nuts0_id']
        
        # Add base scenario data
        total_exposure.loc[country_code, base_scenario] = country_row['total_affected']
        total_exposure_ratio.loc[country_code, base_scenario] = country_row['total_exposure_ratio']
        poorest20_exposure.loc[country_code, base_scenario] = country_row['poorest_20_affected']
        poorest20_exposure_ratio.loc[country_code, base_scenario] = country_row['poorest20_exposure_ratio']
        
        # Calculate differences for comparison scenarios
        for scenario in comparison_scenarios:
            if scenario in scenario_results:
                comp_data = scenario_results[scenario]
                comp_country = comp_data[comp_data['nuts0_id'] == country_code]
                
                if not comp_country.empty:
                    # For absolute numbers tables, add both absolute values and percentage differences
                    if 'total_affected' in comp_country.columns:
                        # Add absolute values
                        total_exposure.loc[country_code, scenario] = comp_country['total_affected'].iloc[0]
                        poorest20_exposure.loc[country_code, scenario] = comp_country['poorest_20_affected'].iloc[0]
                        
                        # Calculate and add percentage differences
                        total_diff = ((comp_country['total_affected'].iloc[0] - country_row['total_affected']) / 
                                    country_row['total_affected'] * 100 if country_row['total_affected'] > 0 else 0)
                        poorest20_diff = ((comp_country['poorest_20_affected'].iloc[0] - country_row['poorest_20_affected']) / 
                                        country_row['poorest_20_affected'] * 100 if country_row['poorest_20_affected'] > 0 else 0)
                        
                        total_exposure.loc[country_code, f'{scenario}_diff'] = total_diff
                        poorest20_exposure.loc[country_code, f'{scenario}_diff'] = poorest20_diff
                    
                    # For ratio tables, only add percentage differences
                    total_ratio_diff = ((comp_country['total_exposure_ratio'].iloc[0] - country_row['total_exposure_ratio']) / 
                                      country_row['total_exposure_ratio'] * 100 if country_row['total_exposure_ratio'] > 0 else 0)
                    poorest20_ratio_diff = ((comp_country['poorest20_exposure_ratio'].iloc[0] - country_row['poorest20_exposure_ratio']) / 
                                          country_row['poorest20_exposure_ratio'] * 100 if country_row['poorest20_exposure_ratio'] > 0 else 0)
                    
                    total_exposure_ratio.loc[country_code, f'{scenario}_diff'] = total_ratio_diff
                    poorest20_exposure_ratio.loc[country_code, f'{scenario}_diff'] = poorest20_ratio_diff
    
    # Save tables
    output_dir = os.path.join(FIG_DIR, f'scenario_comparison_{year}_{rp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Format and save tables
    tables = {
        'Total Population Exposure (Absolute Numbers)': total_exposure,
        'Total Population Exposure (Ratios)': total_exposure_ratio,
        'Poorest 20% Exposure (Absolute Numbers)': poorest20_exposure,
        'Poorest 20% Exposure (Ratios)': poorest20_exposure_ratio
    }
    
    for title, df in tables.items():
        print(f"\nProcessing {title}...")
        
        # Round numbers
        df = df.round(2)
        
        # Rename columns to remove '_diff' suffix
        df.columns = [col.replace('_diff', '') for col in df.columns]
        
        # Save to CSV
        csv_filename = f'{title.lower().replace(" ", "_")}_{year}_{rp}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path)
        print(f"Saved CSV file: {csv_path}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
        
        # Create table plot
        table = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_text_props(weight='bold')
            elif col == 0:  # Index column
                cell.set_text_props(weight='bold')
        
        # Remove axes
        ax.axis('off')
        
        # Add title
        plt.title(f'{title} - {year}, {rp}', pad=20)
        
        # Save figure
        png_filename = f'{title.lower().replace(" ", "_")}_{year}_{rp}.png'
        png_path = os.path.join(output_dir, png_filename)
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG file: {png_path}")

def main():
    parser = argparse.ArgumentParser(description='Create tables comparing exposure across different SSP scenarios')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze (e.g., 2050)')
    parser.add_argument('--rp', type=str, required=True, help='Return period (e.g., RP100)')
    
    args = parser.parse_args()
    
    print(f"\nStarting scenario comparison analysis...")
    calculate_exposure_differences(args.year, args.rp)
    print("\nAnalysis completed successfully!")

if __name__ == '__main__':
    main() 