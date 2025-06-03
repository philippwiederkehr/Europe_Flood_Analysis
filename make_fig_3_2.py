#!/usr/bin/env python3
"""
Make Figure 3-2 – Small-multiple line plots of % poorest-20 exposed over time.

Shows evolution for key macro-regions under a specific SSP and RP.
Macro-regions are defined as the top 4 NUTS-0 countries by poorest 20% population in 2020.

Usage (example):
    python make_fig_3_2.py --ssp SSP2 --rp RP100
"""
import argparse
import os
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # headless / SLURM safe
plt.rcParams['font.family'] = 'sans-serif' # Added for sans-serif font
import pandas as pd
import numpy as np
import seaborn as sns

# Hard-wired roots
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

YEARS = [2020, 2030, 2050, 2100]

def build_paths(ssp: str, year: int, rp: str):
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def get_share_poorest20_exposed(gdf):
    """Calculates share_poorest20_exposed, handling potential division by zero."""
    if "poorest_20_affected" in gdf.columns and "poorest_20_population" in gdf.columns:
        share = (gdf["poorest_20_affected"] / gdf["poorest_20_population"]) * 100
        return share.replace([np.inf, -np.inf], np.nan) # Replace inf with NaN
    return pd.Series(index=gdf.index, dtype='float64')


def plot_evolution_ssp(ssp: str, rp: str):
    """
    Generates small-multiple line plots for the evolution of poorest 20% exposure.
    Also generates a detailed text description file.
    """
    all_data = []
    nuts0_population_2020 = {}

    for year in YEARS:
        geojson_path = build_paths(ssp, year, rp)
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON not found for {ssp}, {year}, {rp}: {geojson_path}")
            continue
        
        gdf = gpd.read_file(geojson_path)
        gdf["year"] = year
        gdf["nuts0_id"] = gdf["nuts_id"].str.slice(0, 2) # Extract NUTS-0 country code

        # Calculate share for each NUTS-3 region
        gdf["share_poorest20_exposed_nuts3"] = get_share_poorest20_exposed(gdf)

        # Aggregate at NUTS-0 level
        # Sum affected and population at NUTS-0, then calculate share
        nuts0_agg = gdf.groupby("nuts0_id").agg(
            total_poorest_20_affected=("poorest_20_affected", "sum"),
            total_poorest_20_population=("poorest_20_population", "sum")
        ).reset_index()
        
        nuts0_agg["share_poorest20_exposed"] = (
            nuts0_agg["total_poorest_20_affected"] / nuts0_agg["total_poorest_20_population"]
        ) * 100
        nuts0_agg["share_poorest20_exposed"] = nuts0_agg["share_poorest20_exposed"].replace([np.inf, -np.inf], np.nan)
        
        nuts0_agg["year"] = year
        all_data.append(nuts0_agg)

        if year == 2020:
            pop_2020_series = gdf.groupby("nuts0_id")["poorest_20_population"].sum()
            nuts0_population_2020 = pop_2020_series.to_dict()

    if not all_data:
        print("No data loaded. Exiting.")
        return

    df_time = pd.concat(all_data)
    
    fallback_triggered = False
    # Determine top 4 macro-regions (NUTS-0 countries) based on 2020 poorest_20_population
    if not nuts0_population_2020:
        print("Could not determine top macro-regions due to missing 2020 data.")
        fallback_triggered = True
        # Fallback: use all available NUTS-0 regions or a predefined list if any data exists
        if not df_time.empty:
            top_nuts0_codes = sorted(df_time["nuts0_id"].unique())[:4]
            print(f"Using fallback macro-regions: {top_nuts0_codes}")
        else:
            print("No data available to select macro-regions for plot or description.")
            return
    else:
        sorted_nuts0_by_pop = sorted(nuts0_population_2020.items(), key=lambda item: item[1], reverse=True)
        top_nuts0_codes = [item[0] for item in sorted_nuts0_by_pop[:4]]
        print(f"Selected top 4 macro-regions (NUTS-0): {top_nuts0_codes}")

    # --- Generate TXT description ---
    description_lines = []
    title_str = f"% Poorest 20% Exposed Over Time ({ssp}, {rp})"

    description_lines.append("Figure Description: Evolution of Percentage of Poorest 20% Exposed to Flooding for Top Macro-Regions")
    description_lines.append("====================================================================================================")
    description_lines.append(f"Analysis Parameters: SSP: {ssp}, Return Period: {rp}")
    description_lines.append(f"Figure Title: \"{title_str}\" (Note: Title is at the bottom of the plot)")
    description_lines.append(f"Source Data: Aggregated from annual GeoJSON files (analysis_results_{ssp}_{{year}}_{rp}.geojson for years {YEARS})")
    description_lines.append("====================================================================================================")
    description_lines.append("")
    description_lines.append("1. General Information:")
    description_lines.append("   - This document describes a set of small-multiple line plots.")
    description_lines.append("   - The figure visualizes the temporal evolution of the percentage of the poorest 20% of the population exposed to flooding.")
    description_lines.append(f"   - Data is shown for the years: {', '.join(map(str, YEARS))}.")
    description_lines.append("   - The analysis focuses on key macro-regions, defined as the top 4 NUTS-0 (country-level) regions by their 'poorest_20_population' in the baseline year 2020.")
    if fallback_triggered:
        description_lines.append("     (Note: Fallback logic was used to select regions due to missing 2020 population data for ranking.)")
    description_lines.append("   - The 'share_poorest20_exposed' for each NUTS-0 region is calculated as:")
    description_lines.append("     (SUM of 'poorest_20_affected' in NUTS-3 regions within NUTS-0 / SUM of 'poorest_20_population' in NUTS-3 regions within NUTS-0) * 100.")
    description_lines.append("")
    description_lines.append("2. Plotted Macro-Regions:")
    description_lines.append("   - The following NUTS-0 regions are displayed:")
    if not top_nuts0_codes:
        description_lines.append("     - No regions were plotted due to lack of data.")
    for code in top_nuts0_codes:
        pop_2020 = nuts0_population_2020.get(code, "N/A")
        pop_2020_str = f"{pop_2020:,.0f}" if isinstance(pop_2020, (int, float)) else pop_2020
        description_lines.append(f"     - {code} (2020 Poorest 20% Pop: {pop_2020_str})")
    description_lines.append("")
    description_lines.append("3. Data Trends and Observations for Each Plotted Macro-Region:")
    
    if df_time.empty:
        description_lines.append("   - No time-series data available to describe trends.")
    else:
        for nuts0_code in top_nuts0_codes:
            description_lines.append(f"   For Region: {nuts0_code}")
            region_data_ts = df_time[df_time["nuts0_id"] == nuts0_code].set_index("year")
            if region_data_ts.empty:
                description_lines.append("     - No data available for this region in the time series.")
                continue

            exposure_values_strs = []
            valid_exposures = []
            for year_val in YEARS:
                if year_val in region_data_ts.index and pd.notna(region_data_ts.loc[year_val, "share_poorest20_exposed"]):
                    val = region_data_ts.loc[year_val, "share_poorest20_exposed"]
                    exposure_values_strs.append(f"Year {year_val}: {val:.2f}%")
                    valid_exposures.append({"year": year_val, "value": val})
                else:
                    exposure_values_strs.append(f"Year {year_val}: No data")
            description_lines.append(f"     - Data points (% Poorest 20% Exposed): {'; '.join(exposure_values_strs)}")

            if len(valid_exposures) >= 2:
                first_valid = valid_exposures[0]
                last_valid = valid_exposures[-1]
                change = last_valid['value'] - first_valid['value']
                description_lines.append(f"     - Overall trend ({first_valid['year']} to {last_valid['year']}): Exposure changed from {first_valid['value']:.2f}% to {last_valid['value']:.2f}% (Change: {change:+.2f}% points).")
                
                min_exp = min(item['value'] for item in valid_exposures)
                max_exp = max(item['value'] for item in valid_exposures)
                min_year = [item['year'] for item in valid_exposures if item['value'] == min_exp][0]
                max_year = [item['year'] for item in valid_exposures if item['value'] == max_exp][0]
                description_lines.append(f"     - Minimum exposure: {min_exp:.2f}% in {min_year}.")
                description_lines.append(f"     - Maximum exposure: {max_exp:.2f}% in {max_year}.")
            elif len(valid_exposures) == 1:
                description_lines.append(f"     - Only one data point available: {valid_exposures[0]['value']:.2f}% in {valid_exposures[0]['year']}.")
            else:
                description_lines.append("     - Insufficient data to determine trends, min/max exposure for this region.")
            description_lines.append("") # Add a blank line for readability between regions

    description_lines.append("4. Comparative Observations:")
    description_lines.append("   - Detailed inter-regional comparison requires manual interpretation of the plot.")
    description_lines.append("   - The plot allows for visual assessment of relative exposure levels and trends across the selected macro-regions.")
    description_lines.append("")
    description_lines.append("5. Plot Layout and Visual Details:")
    description_lines.append(f"   - Arrangement: Up to 2x2 grid of subplots (actual number depends on available data for top regions), one for each macro-region.")
    description_lines.append(f"   - X-Axis (for each subplot): Year, with points for {', '.join(map(str, YEARS))}.")
    description_lines.append("   - Y-Axis (for each subplot, shared scale): \"% Poorest 20% Exposed\".")
    description_lines.append("   - Lines: Each region's data is plotted as a line with 'o' markers.")
    description_lines.append("   - Title: Main title located at the bottom of the figure. Subplot titles indicate the NUTS-0 region code.")
    description_lines.append("   - Font: Sans-serif font used globally.")
    description_lines.append("   - Grid: Light grid lines present for readability in each subplot.")
    description_lines.append("   - Legend: Each subplot includes a legend indicating the NUTS-0 code.")
    description_lines.append("")
    description_lines.append("6. Key Data Columns Used in Aggregation and Calculation:")
    description_lines.append("   - 'poorest_20_affected': Number of people in the poorest 20% income group affected by flooding (at NUTS-3 level, then aggregated).")
    description_lines.append("   - 'poorest_20_population': Total number of people in the poorest 20% income group (at NUTS-3 level, then aggregated for NUTS-0 share and for 2020 ranking).")
    description_lines.append("   - 'nuts_id': NUTS identifier, used to extract 'nuts0_id'.")
    description_lines.append("   - 'year': The specific year of the data.")
    description_lines.append("")
    description_lines.append("7. Purpose of the Figure:")
    description_lines.append(f"   - To illustrate how flood exposure for the poorest 20% of the population is projected to evolve over time in key European macro-regions under the {ssp} scenario and {rp} return period.")
    description_lines.append("   - To identify regions where this vulnerable group may face increasing or decreasing risks in the future.")
    description_lines.append("   - This information can support the development of long-term, equitable climate adaptation and disaster risk reduction strategies, highlighting areas needing proactive intervention.")
    description_lines.append("")
    description_lines.append("This description is auto-generated based on the data and parameters used for the plot.")
    
    plot_filename = f"fig3_2_poorest20_evolution_{ssp}_{rp}.png"
    description_lines.append(f"Filename of the plot: {plot_filename}")

    txt_out_name = f"fig3_2_poorest20_evolution_{ssp}_{rp}_description.txt"
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)
    with open(txt_fig_path, "w", encoding='utf-8') as f:
        f.write("\\n".join(description_lines))
    print(f"✓ Saved description {txt_fig_path}")
    # --- End of TXT description generation ---

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i, nuts0_code in enumerate(top_nuts0_codes):
        if i >= len(axes): # Should not happen if top_nuts0_codes has <= 4 elements
            break 
        ax = axes[i]
        region_data = df_time[df_time["nuts0_id"] == nuts0_code]
        
        if not region_data.empty:
            sns.lineplot(data=region_data, x="year", y="share_poorest20_exposed", ax=ax, marker='o', label=nuts0_code)
            ax.set_title(f"Region: {nuts0_code}")
            ax.set_xlabel("Year")
            ax.set_ylabel("% Poorest 20% Exposed")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No data for {nuts0_code}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"Region: {nuts0_code}")

    # Handle cases where fewer than 4 regions have data
    for j in range(len(top_nuts0_codes), len(axes)): # Corrected loop to hide unused axes based on actual top_nuts0_codes
        axes[j].axis('off') # Hide unused subplots

    fig.suptitle(title_str, fontsize=16, y=0.03) # Adjusted y for bottom title
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle (rect might need tuning)

    out_name = plot_filename # Use the variable defined earlier
    fig_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Figure 3-2: Small-multiple line plots of poorest 20% exposure evolution.")
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    plot_evolution_ssp(args.ssp, args.rp)
