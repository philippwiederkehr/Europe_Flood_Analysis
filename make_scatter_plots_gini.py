#!/usr/bin/env python3
"""
Make country-level scatter plots using GINI coefficient:
1. GINI coefficient vs. % total population exposed
2. GINI coefficient vs. % poorest 20% of total population exposed

Usage (example):
    python make_scatter_plots_gini.py --ssp SSP2 --year 2020 --rp RP100
"""
import argparse
import os
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / SLURM safe
import matplotlib.pyplot as plt

# Set sans-serif font globally for the script
plt.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------------
# Hard-wired roots
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
# Construct the path to the GINI CSV file relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GINI_CSV_FILE = os.path.join(SCRIPT_DIR, "estat_ilc_di12c_filtered_en.csv")
os.makedirs(FIG_DIR, exist_ok=True)

def build_paths(ssp: str, year: str, rp: str):
    """Builds file paths for input GeoJSON."""
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def load_gini_data(gini_csv_path: str) -> pd.DataFrame:
    """Loads GINI coefficient data from the provided CSV file."""
    try:
        gini_df = pd.read_csv(gini_csv_path)
    except FileNotFoundError:
        print(f"Error: GINI CSV file not found at {gini_csv_path}")
        return pd.DataFrame() # Return empty DataFrame

    # Select relevant columns and rename
    # Assuming 'geo' is country code and 'OBS_VALUE' is GINI.
    # 'TIME_PERIOD' is also available.
    if 'geo' not in gini_df.columns or 'OBS_VALUE' not in gini_df.columns or 'TIME_PERIOD' not in gini_df.columns:
        print("Error: Required columns ('geo', 'OBS_VALUE', 'TIME_PERIOD') not in GINI CSV.")
        return pd.DataFrame()

    gini_df = gini_df[['geo', 'OBS_VALUE', 'TIME_PERIOD']].copy()
    gini_df.rename(columns={'geo': 'country_code', 'OBS_VALUE': 'gini_coefficient'}, inplace=True)
    
    # Convert GINI to numeric, coercing errors
    gini_df['gini_coefficient'] = pd.to_numeric(gini_df['gini_coefficient'], errors='coerce')
    gini_df.dropna(subset=['gini_coefficient'], inplace=True) # Remove rows where GINI could not be converted

    # If multiple years per country, take the latest GINI value.
    gini_df.sort_values(by=['country_code', 'TIME_PERIOD'], ascending=[True, False], inplace=True)
    gini_df.drop_duplicates(subset=['country_code'], keep='first', inplace=True)
    
    return gini_df[['country_code', 'gini_coefficient']]

def aggregate_data_to_country_level(gdf: gpd.GeoDataFrame, gini_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates NUTS-level data to country level and merges GINI data."""
    if 'nuts_id' not in gdf.columns:
        raise ValueError("'nuts_id' column not found in GeoDataFrame.")

    gdf['country_code'] = gdf['nuts_id'].str.slice(0, 2)

    required_cols = ['total_population', 'total_affected', 
                     'poorest_20_population', 'poorest_20_affected', 
                     'country_code'] # country_median_income removed as it's not the focus
    
    # Check if country_median_income exists, if so, include it for potential other uses, but don't require it.
    agg_dict = {
        'total_population_country': ('total_population', 'sum'),
        'total_affected_country': ('total_affected', 'sum'),
        'poorest_20_population_country': ('poorest_20_population', 'sum'),
        'poorest_20_affected_country': ('poorest_20_affected', 'sum')
    }
    if 'country_median_income' in gdf.columns:
        required_cols.append('country_median_income') # Add to check if present
        agg_dict['country_median_income'] = ('country_median_income', 'first')


    for col in required_cols:
        if col not in gdf.columns: # This check might be too strict if country_median_income is optional
            # If country_median_income is truly optional, this check needs adjustment or it should be removed from required_cols if not used.
            # For now, let's assume it might be there from the GeoJSON.
            if col == 'country_median_income' and 'country_median_income' not in agg_dict: # If it was optional and not added
                continue
            raise ValueError(f"Required column '{col}' not found in GeoDataFrame.")

    numeric_cols_to_sum = ['total_population', 'total_affected', 'poorest_20_population', 'poorest_20_affected']
    for col in numeric_cols_to_sum:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    country_data = gdf.groupby('country_code').agg(**agg_dict).reset_index()

    # Merge with GINI data
    if not gini_df.empty:
        country_data = pd.merge(country_data, gini_df, on='country_code', how='left')
    else:
        country_data['gini_coefficient'] = pd.NA # Add column if GINI data is empty to avoid errors later

    country_data['perc_total_pop_exposed'] = country_data.apply(
        lambda row: (row['total_affected_country'] / row['total_population_country']) * 100
        if row['total_population_country'] > 0 else 0, axis=1
    )
    country_data['perc_poorest20_of_total_pop_exposed'] = country_data.apply(
        lambda row: (row['poorest_20_affected_country'] / row['poorest_20_population_country']) * 100
        if row['poorest_20_population_country'] > 0 else 0, axis=1
    )
    
    return country_data.dropna(subset=['gini_coefficient', 'perc_total_pop_exposed', 'perc_poorest20_of_total_pop_exposed'])


def plot_country_scatter_graphs_gini(geojson_path: str, gini_csv_path: str, ssp: str, year: str, rp: str):
    """Generates and saves the scatter plots using GINI coefficient."""
    gdf = gpd.read_file(geojson_path)
    gini_df = load_gini_data(gini_csv_path)

    if gini_df.empty:
        print(f"GINI data could not be loaded or is empty. Skipping plots that require GINI.")
        return

    country_df = aggregate_data_to_country_level(gdf, gini_df)

    if country_df.empty:
        print(f"No country data to plot for {ssp}, {year}, {rp} after aggregation, GINI merge, and cleaning. Skipping plots.")
        return

    # Plot 1: GINI Coefficient vs. % Total Population Exposed
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(country_df['gini_coefficient'], country_df['perc_total_pop_exposed'])
    
    for i, txt in enumerate(country_df['country_code']):
        ax1.annotate(txt, (country_df['gini_coefficient'].iloc[i], country_df['perc_total_pop_exposed'].iloc[i]),
                     textcoords="offset points", xytext=(0,5), ha='center')

    ax1.set_xlabel("GINI Coefficient")
    ax1.set_ylabel("% Total Population Exposed to Flooding")
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.subplots_adjust(bottom=0.2) # Adjust bottom margin for title
    title1 = f"GINI Coefficient vs. Total Population Exposed\n({ssp}, {year}, {rp})"
    ax1.set_title(title1, y=-0.25) # Position title at the bottom
    
    plot1_out_name = f"country_scatter_gini_vs_total_exposed_{ssp}_{year}_{rp}.png"
    plot1_fig_path = os.path.join(FIG_DIR, plot1_out_name)
    plt.savefig(plot1_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Saved Plot 1: {plot1_fig_path}")

    # Plot 2: GINI Coefficient vs. % Poorest 20% of Total Population Exposed
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.scatter(country_df['gini_coefficient'], country_df['perc_poorest20_of_total_pop_exposed'], color='coral')

    for i, txt in enumerate(country_df['country_code']):
        ax2.annotate(txt, (country_df['gini_coefficient'].iloc[i], country_df['perc_poorest20_of_total_pop_exposed'].iloc[i]),
                     textcoords="offset points", xytext=(0,5), ha='center')

    ax2.set_xlabel("GINI Coefficient")
    ax2.set_ylabel("% Poorest 20% Exposed to Flooding")
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.subplots_adjust(bottom=0.2) # Adjust bottom margin for title
    title2 = f"GINI Coefficient vs. Poorest 20% Exposed\n({ssp}, {year}, {rp})"
    ax2.set_title(title2, y=-0.25) # Position title at the bottom

    plot2_out_name = f"country_scatter_gini_vs_poorest20_exposed_{ssp}_{year}_{rp}.png"
    plot2_fig_path = os.path.join(FIG_DIR, plot2_out_name)
    plt.savefig(plot2_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved Plot 2: {plot2_fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate country-level scatter plots of flood exposure vs. GINI coefficient.")
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--year", required=True, help="Year (e.g. 2020 for flood data)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    geojson_fp = build_paths(args.ssp, args.year, args.rp)
    
    if not os.path.exists(geojson_fp):
        raise FileNotFoundError(f"GeoJSON not found: {geojson_fp}. Cannot generate plots.")
    
    if not os.path.exists(GINI_CSV_FILE):
        raise FileNotFoundError(f"GINI CSV file not found: {GINI_CSV_FILE}. Cannot generate plots.")

    print(f"Processing GeoJSON file: {geojson_fp}")
    print(f"Using GINI data from: {GINI_CSV_FILE}")
    plot_country_scatter_graphs_gini(geojson_fp, GINI_CSV_FILE, args.ssp, args.year, args.rp)
    print("Script finished.")
