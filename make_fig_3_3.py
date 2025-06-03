#!/usr/bin/env python3
"""
Make Figure 3-3 – Grouped bar plot of exposure by return period.

Shows total affected and poorest 20% affected for different return periods 
for a specific SSP and year (e.g., 2050).

Usage (example):
    python make_fig_3_3.py --ssp SSP2 --year 2050
"""
import argparse
import os
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless / SLURM safe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns # For easier grouped bar plots

# Hard-wired roots
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

RETURN_PERIODS = ["RP30", "RP100", "RP500"]

def build_paths(ssp: str, year: int, rp: str):
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def plot_hazard_severity_bars(ssp: str, year: int):
    """
    Generates a grouped bar plot for exposure by hazard severity (return period)
    and a detailed text file describing the data.
    """
    # Set font to sans-serif for the plot
    matplotlib.rcParams['font.family'] = 'sans-serif'

    all_rp_data = []

    for rp in RETURN_PERIODS:
        geojson_path = build_paths(ssp, year, rp)
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON not found for {ssp}, {year}, {rp}: {geojson_path}")
            continue
        
        gdf = gpd.read_file(geojson_path)
        
        # Calculate total affected and poorest 20% affected for the entire dataset (Europe)
        total_affected_sum = gdf["total_affected"].sum()
        poorest_20_affected_sum = gdf["poorest_20_affected"].sum()
        
        # Data for plotting
        all_rp_data.append({
            "rp": rp,
            "group": "Total Population Affected",
            "affected_population": total_affected_sum
        })
        all_rp_data.append({
            "rp": rp,
            "group": "Poorest 20% Affected",
            "affected_population": poorest_20_affected_sum
        })

    if not all_rp_data:
        print("No data loaded for any return period. Exiting.")
        return

    df_bar = pd.DataFrame(all_rp_data)

    # --- Generate detailed text file ---
    txt_content = []
    txt_content.append(f"Detailed Analysis for Figure 3-3: Flood Exposure by Return Period")
    txt_content.append(f"SSP Scenario: {ssp}")
    txt_content.append(f"Year: {year}")
    txt_content.append("\\n")
    txt_content.append("This document provides a detailed textual description of the data presented in the bar graph,")
    txt_content.append("which illustrates estimated flood-affected populations in Europe under the specified SSP scenario and year.")
    txt_content.append(f"The graph displays data for three distinct flood return periods: {', '.join(RETURN_PERIODS)}.")
    txt_content.append("For each return period, two categories of affected populations are shown:")
    txt_content.append("1. Total Population Affected: Represents the overall number of individuals projected to be impacted by flooding.")
    txt_content.append("2. Poorest 20% Affected: Represents the subset of the total affected population that falls within the poorest 20% income group.")
    txt_content.append("\\n")
    txt_content.append("Data Summary by Return Period:")
    txt_content.append("------------------------------")

    for rp_value in RETURN_PERIODS:
        txt_content.append(f"\\nReturn Period: {rp_value}")
        
        total_affected_row = df_bar[(df_bar['rp'] == rp_value) & (df_bar['group'] == "Total Population Affected")]
        poorest_affected_row = df_bar[(df_bar['rp'] == rp_value) & (df_bar['group'] == "Poorest 20% Affected")]
        
        if not total_affected_row.empty:
            total_pop_affected = total_affected_row['affected_population'].iloc[0]
            txt_content.append(f"  - Total Population Affected: {total_pop_affected/1e6:.2f} million people")
        else:
            txt_content.append(f"  - Total Population Affected: Data not available for {rp_value}")
            
        if not poorest_affected_row.empty:
            poorest_pop_affected = poorest_affected_row['affected_population'].iloc[0]
            txt_content.append(f"  - Poorest 20% Affected: {poorest_pop_affected/1e6:.2f} million people")
        else:
            txt_content.append(f"  - Poorest 20% Affected: Data not available for {rp_value}")

    txt_content.append("\\n")
    txt_content.append("Interpretation Notes:")
    txt_content.append("--------------------")
    txt_content.append("- All population figures are estimates and presented in millions.")
    txt_content.append("- The 'Return Period' (e.g., RP10) signifies the average recurrence interval of a flood event of a certain magnitude (e.g., a 1-in-10-year flood).")
    txt_content.append("- Data is aggregated at the European level based on underlying GeoJSON analysis results for each scenario.")
    txt_content.append(f"- Source GeoJSON files are expected to be found in directories like: {BASE_OUT_DIR}/results_{ssp}_{year}_[RP]/analysis_results_{ssp}_{year}_[RP].geojson")
    txt_content.append("- The 'poorest 20%' classification is based on the income data within the analyzed datasets.")
    txt_content.append("\\n")
    txt_content.append("This textual summary is designed to assist Large Language Models (LLMs) and researchers in accurately understanding and describing the visual information conveyed by the corresponding graph.")

    txt_out_name = f"fig3_3_hazard_severity_{ssp}_{year}_data.txt"
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)
    try:
        with open(txt_fig_path, "w") as f:
            f.write("\\n".join(txt_content))
        print(f"✓ Saved text summary {txt_fig_path}")
    except IOError as e:
        print(f"Error writing text summary file {txt_fig_path}: {e}")
    # --- End of text file generation ---

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_bar, x="rp", y="affected_population", hue="group")
    
    plt.xlabel("Return Period")
    plt.ylabel("Affected Population (millions)")
    plt.legend(title="Population Group")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Format y-axis to millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    # Add title at the bottom of the figure
    fig_title = f"Flood Exposure by Return Period ({ssp}, {year})"
    plt.figtext(0.5, 0.02, fig_title, ha="center", va="bottom", fontsize=12, wrap=True)

    out_name = f"fig3_3_hazard_severity_{ssp}_{year}.png"
    fig_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Figure 3-3: Grouped bar plot of exposure by return period.")
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--year", required=True, type=int, help="Year (e.g. 2050)")
    args = p.parse_args()

    plot_hazard_severity_bars(args.ssp, args.year)
