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
    ax1.set_title(f"Country Median Income vs. Total Population Exposed\\n({ssp}, {year}, {rp})", y=-0.1) # Position title at the bottom
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
    ax2.set_title(f"Country Median Income vs. Poorest 20% Exposed\\n({ssp}, {year}, {rp})", y=-0.1) # Position title at the bottom
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot2_out_name = f"country_scatter_income_vs_poorest20_exposed_{ssp}_{year}_{rp}.png"
    plot2_fig_path = os.path.join(FIG_DIR, plot2_out_name)
    plt.savefig(plot2_fig_path, dpi=300)
    plt.close(fig2)
    print(f"✓ Saved Plot 2: {plot2_fig_path}")

    # --- Generate TXT description for both scatter plots ---
    txt_out_name = f"country_scatter_plots_description_{ssp}_{year}_{rp}.txt"
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)

    def get_correlation_interpretation(corr_value):
        if pd.isna(corr_value):
            return "not calculable (likely due to insufficient data or variance)."
        abs_corr = abs(corr_value)
        strength = ""
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.1:
            strength = "weak"
        else:
            strength = "very weak or no"

        direction = ""
        if corr_value > 0.05:
            direction = "positive"
        elif corr_value < -0.05:
            direction = "negative"
        
        if strength == "very weak or no":
            return f"{strength} linear relationship"
        return f"a {strength} {direction} linear relationship"

    def format_top_bottom_list(df, column_name, metric_name, top_n=5, ascending=False, is_percentage=False):
        if df.empty or column_name not in df.columns:
            return f"     - Data for {metric_name} not available.\\n"
        
        df_sorted = df.sort_values(by=column_name, ascending=ascending).dropna(subset=[column_name])
        lines = []
        if df_sorted.empty:
            return f"     - No valid data for {metric_name}.\\n"

        num_entries = min(top_n, len(df_sorted))
        for i in range(num_entries):
            country_code = df_sorted['country_code'].iloc[i]
            value = df_sorted[column_name].iloc[i]
            unit = "%" if is_percentage else ""
            lines.append(f"     {i+1}. {country_code}: {value:.2f}{unit}")
        if not lines:
             return f"     - Not enough data to list top/bottom {metric_name}.\\n"
        return "\\n".join(lines)

    # Statistics for Plot 1
    corr1 = country_df['country_median_income'].corr(country_df['perc_total_pop_exposed'])
    interp1 = get_correlation_interpretation(corr1)
    
    top_exposed_total_str = format_top_bottom_list(country_df, 'perc_total_pop_exposed', "% Total Exposed", is_percentage=True, ascending=False)
    bottom_exposed_total_str = format_top_bottom_list(country_df, 'perc_total_pop_exposed', "% Total Exposed", is_percentage=True, ascending=True)

    # Statistics for Plot 2
    corr2 = country_df['country_median_income'].corr(country_df['perc_poorest20_of_total_pop_exposed'])
    interp2 = get_correlation_interpretation(corr2)

    top_exposed_poorest20_str = format_top_bottom_list(country_df, 'perc_poorest20_of_total_pop_exposed', "% Poorest 20% Exposed", is_percentage=True, ascending=False)
    bottom_exposed_poorest20_str = format_top_bottom_list(country_df, 'perc_poorest20_of_total_pop_exposed', "% Poorest 20% Exposed", is_percentage=True, ascending=True)
    
    # Common income stats
    top_income_str = format_top_bottom_list(country_df, 'country_median_income', "Median Income", ascending=False)
    bottom_income_str = format_top_bottom_list(country_df, 'country_median_income', "Median Income", ascending=True)

    description = f"""FigureS Description: Country-Level Scatter Plots of Flood Exposure vs. Median Income
================================================================================
Analysis Parameters: SSP: {ssp}, Year: {year}, Return Period: {rp}
Source GeoJSON file: {os.path.basename(geojson_path)}
================================================================================

This document describes two scatter plots:
1. Country Median Income vs. Percentage of Total Population Exposed to Flooding
2. Country Median Income vs. Percentage of Poorest 20% of the Total Population Exposed to Flooding

Data based on {len(country_df)} countries after aggregation and cleaning.

--------------------------------------------------------------------------------
Plot 1: Country Median Income vs. % Total Population Exposed
--------------------------------------------------------------------------------
Title: "Country Median Income vs. Total Population Exposed ({ssp}, {year}, {rp})"

1. General Information:
   - This scatter plot visualizes the relationship between the median income of countries and the percentage of their total population exposed to flooding.
   - Each point represents a country, annotated with its country code.
   - X-axis: Country Median Income (in monetary units specific to the dataset).
   - Y-axis: Percentage of Total Population Exposed to Flooding (%).

2. Data Summary for Plot 1:
   - Country Median Income:
     - Min: {country_df['country_median_income'].min():.2f}
     - Max: {country_df['country_median_income'].max():.2f}
     - Mean: {country_df['country_median_income'].mean():.2f}
     - Median: {country_df['country_median_income'].median():.2f}
     - Std Dev: {country_df['country_median_income'].std():.2f}
   - % Total Population Exposed:
     - Min: {country_df['perc_total_pop_exposed'].min():.2f}%
     - Max: {country_df['perc_total_pop_exposed'].max():.2f}%
     - Mean: {country_df['perc_total_pop_exposed'].mean():.2f}%
     - Median: {country_df['perc_total_pop_exposed'].median():.2f}%
     - Std Dev: {country_df['perc_total_pop_exposed'].std():.2f}%
   - Pearson Correlation: {corr1:.3f}
     - Interpretation: Suggests {interp1}.

3. Notable Observations:
   - Countries with highest % Total Population Exposed:
{top_exposed_total_str}
   - Countries with lowest % Total Population Exposed:
{bottom_exposed_total_str}
   - Countries with highest Median Income:
{top_income_str}
   - Countries with lowest Median Income:
{bottom_income_str}

4. Key Data Columns Used:
   - 'country_median_income': Median income for the country (derived from 'first' value in NUTS data).
   - 'perc_total_pop_exposed': Calculated as (total_affected_country / total_population_country) * 100.
   - 'country_code': Two-letter country identifier from 'nuts_id'.

5. Purpose of the Figure:
   - To investigate the relationship between a country's median income and the overall flood exposure of its population.
   - This can help identify if wealthier or poorer nations, on average, face higher percentages of population exposure to floods under the specified scenario.

--------------------------------------------------------------------------------
Plot 2: Country Median Income vs. % Poorest 20% of Population Exposed
--------------------------------------------------------------------------------
Title: "Country Median Income vs. Poorest 20% Exposed ({ssp}, {year}, {rp})"

1. General Information:
   - This scatter plot visualizes the relationship between the median income of countries and the percentage of their poorest 20% of the population exposed to flooding.
   - Each point represents a country, annotated with its country code.
   - X-axis: Country Median Income (in monetary units specific to the dataset).
   - Y-axis: Percentage of Poorest 20% of Population Exposed to Flooding (%).

2. Data Summary for Plot 2:
   - Country Median Income: (Same as Plot 1)
     - Min: {country_df['country_median_income'].min():.2f}
     - Max: {country_df['country_median_income'].max():.2f}
     - Mean: {country_df['country_median_income'].mean():.2f}
     - Median: {country_df['country_median_income'].median():.2f}
     - Std Dev: {country_df['country_median_income'].std():.2f}
   - % Poorest 20% Exposed:
     - Min: {country_df['perc_poorest20_of_total_pop_exposed'].min():.2f}%
     - Max: {country_df['perc_poorest20_of_total_pop_exposed'].max():.2f}%
     - Mean: {country_df['perc_poorest20_of_total_pop_exposed'].mean():.2f}%
     - Median: {country_df['perc_poorest20_of_total_pop_exposed'].median():.2f}
     - Std Dev: {country_df['perc_poorest20_of_total_pop_exposed'].std():.2f}%
   - Pearson Correlation: {corr2:.3f}
     - Interpretation: Suggests {interp2}.

3. Notable Observations:
   - Countries with highest % Poorest 20% Exposed:
{top_exposed_poorest20_str}
   - Countries with lowest % Poorest 20% Exposed:
{bottom_exposed_poorest20_str}
   - Countries with highest Median Income (same as for Plot 1):
{top_income_str}
   - Countries with lowest Median Income (same as for Plot 1):
{bottom_income_str}

4. Key Data Columns Used:
   - 'country_median_income': Median income for the country.
   - 'perc_poorest20_of_total_pop_exposed': Calculated as (poorest_20_affected_country / poorest_20_population_country) * 100.
   - 'country_code': Two-letter country identifier.

5. Purpose of the Figure:
   - To investigate the relationship between a country's median income and the flood exposure specifically of its poorest 20% segment.
   - This helps in understanding potential disproportionate impacts on vulnerable populations across countries with different income levels.
   - Comparing with Plot 1 can reveal if the poorest segment's exposure trend differs from the total population's exposure trend in relation to national income.

================================================================================
This description is auto-generated based on the data used for the plots.
Please note that "monetary units" for income are as per the source dataset and
may require further context for specific interpretation (e.g., currency, PPP adjustment).
The term "exposed" refers to being affected by flooding under the model assumptions.
Correlation does not imply causation.
"""
    with open(txt_fig_path, "w", encoding='utf-8') as f:
        f.write(description)
    print(f"✓ Saved description: {txt_fig_path}")

    # --- End of TXT description generation ---

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