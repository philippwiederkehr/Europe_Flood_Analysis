#!/usr/bin/env python3
"""
Make country-level scatter plots:
1. Median income vs. % total population exposed
2. Median income vs. % poorest 20% of total population exposed

Usage (example):
    python make_scatter_plots.py --ssp SSP2 --year 2020 --rp RP100
"""
import argparse
import os
import geopandas as gpd
import pandas as pd # Added for data manipulation
import matplotlib
matplotlib.use("Agg")  # headless / SLURM safe
import matplotlib.pyplot as plt

# Set sans-serif font globally for the script
plt.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------------
# Hard-wired roots (same style as make_fig_3_1.py)
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

def build_paths(ssp: str, year: str, rp: str):
    """Builds file paths for input GeoJSON and CSV files."""
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    # csv_file  = os.path.join(results_dir, f"detailed_analysis_results_{ssp}_{year}_{rp}.csv") # CSV not used in this script
    return geojson #, csv_file

def aggregate_data_to_country_level(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregates NUTS-level data to country level."""
    if 'nuts_id' not in gdf.columns:
        raise ValueError("'nuts_id' column not found in GeoDataFrame.")

    # Create a 'country_code' column from the first two letters of 'nuts_id'
    gdf['country_code'] = gdf['nuts_id'].str.slice(0, 2)

    # Define aggregation functions
    # Ensure required columns exist before aggregation
    required_cols = ['total_population', 'total_affected', 
                     'poorest_20_population', 'poorest_20_affected', 
                     'country_median_income', 'country_code']
    for col in required_cols:
        if col not in gdf.columns:
            raise ValueError(f"Required column '{col}' not found in GeoDataFrame.")

    # Handle potential non-numeric types before sum if necessary, though they should be numeric
    numeric_cols_to_sum = ['total_population', 'total_affected', 'poorest_20_population', 'poorest_20_affected']
    for col in numeric_cols_to_sum:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')


    country_data = gdf.groupby('country_code').agg(
        country_median_income=('country_median_income', 'first'),  # Assuming this is constant per country
        total_population_country=('total_population', 'sum'),
        total_affected_country=('total_affected', 'sum'),
        poorest_20_population_country=('poorest_20_population', 'sum'), # Not directly needed for final plots as per request
        poorest_20_affected_country=('poorest_20_affected', 'sum')
    ).reset_index()

    # Calculate exposure percentages
    # Avoid division by zero if total_population_country is 0 or NaN
    country_data['perc_total_pop_exposed'] = country_data.apply(
        lambda row: (row['total_affected_country'] / row['total_population_country']) * 100
        if row['total_population_country'] > 0 else 0, axis=1
    )
    country_data['perc_poorest20_of_total_pop_exposed'] = country_data.apply(
        lambda row: (row['poorest_20_affected_country'] / row['poorest_20_population_country']) * 100
        if row['poorest_20_population_country'] > 0 else 0, axis=1
    )

    return country_data.dropna(subset=['country_median_income', 'perc_total_pop_exposed', 'perc_poorest20_of_total_pop_exposed'])


def plot_country_scatter_graphs(geojson_path: str, ssp: str, year: str, rp: str):
    """Generates and saves the scatter plots."""
    gdf = gpd.read_file(geojson_path)
    country_df = aggregate_data_to_country_level(gdf)

    if country_df.empty:
        print(f"No country data to plot for {ssp}, {year}, {rp} after aggregation and cleaning. Skipping plots.")
        return

    # Plot 1: Median Income vs. % Total Population Exposed
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(country_df['country_median_income'], country_df['perc_total_pop_exposed'])
    
    for i, txt in enumerate(country_df['country_code']):
        ax1.annotate(txt, (country_df['country_median_income'].iloc[i], country_df['perc_total_pop_exposed'].iloc[i]),
                     textcoords="offset points", xytext=(0,5), ha='center')

    ax1.set_xlabel("Country Median Income ")
    ax1.set_ylabel("% Total Population Exposed to Flooding")
    ax1.set_title(f"Country Median Income vs. Total Population Exposed\n({ssp}, {year}, {rp})")
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot1_out_name = f"country_scatter_income_vs_total_exposed_{ssp}_{year}_{rp}.png"
    plot1_fig_path = os.path.join(FIG_DIR, plot1_out_name)
    plt.savefig(plot1_fig_path, dpi=300)
    plt.close(fig1)
    print(f"✓ Saved Plot 1: {plot1_fig_path}")

    # Plot 2: Median Income vs. % Poorest 20% of Total Population Exposed
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.scatter(country_df['country_median_income'], country_df['perc_poorest20_of_total_pop_exposed'], color='coral')

    for i, txt in enumerate(country_df['country_code']):
        ax2.annotate(txt, (country_df['country_median_income'].iloc[i], country_df['perc_poorest20_of_total_pop_exposed'].iloc[i]),
                     textcoords="offset points", xytext=(0,5), ha='center')

    ax2.set_xlabel("Country Median Income")
    ax2.set_ylabel("% Poorest 20% Exposed to Flooding")
    ax2.set_title(f"Country Median Income vs. Poorest 20% Exposed\n({ssp}, {year}, {rp})")
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot2_out_name = f"country_scatter_income_vs_poorest20_exposed_{ssp}_{year}_{rp}.png"
    plot2_fig_path = os.path.join(FIG_DIR, plot2_out_name)
    plt.savefig(plot2_fig_path, dpi=300)
    plt.close(fig2)
    print(f"✓ Saved Plot 2: {plot2_fig_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate country-level scatter plots of flood exposure vs. median income.")
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--year", required=True, help="Year (e.g. 2020)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    geojson_fp = build_paths(args.ssp, args.year, args.rp)
    
    if not os.path.exists(geojson_fp):
        raise FileNotFoundError(f"GeoJSON not found: {geojson_fp}. Cannot generate plots.")
    
    print(f"Processing file: {geojson_fp}")
    plot_country_scatter_graphs(geojson_fp, args.ssp, args.year, args.rp)
    print("Script finished.")