#!/usr/bin/env python3
"""
Make Figure 3-4 – Lorenz-style curve for socio-economic differential.

Shows cumulative population vs. cumulative exposed for a specific SSP, year, and RP.

Usage (example):
    python make_fig_3_4.py --ssp SSP2 --year 2050 --rp RP100
"""
import argparse
import os
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless / SLURM safe
matplotlib.rcParams['font.family'] = 'sans-serif'  # Set default font to sans-serif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime # Added for timestamp in text file

# Hard-wired roots
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

def build_paths(ssp: str, year: int, rp: str):
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def plot_lorenz_curve(geojson_path: str, ssp: str, year: int, rp: str):
    """
    Generates a Lorenz-style curve for flood exposure inequality.
    """
    if not os.path.exists(geojson_path):
        print(f"Error: GeoJSON not found: {geojson_path}")
        return

    gdf = gpd.read_file(geojson_path)

    # Ensure necessary columns are present and numeric
    required_cols = ['poorest_20_population', 'poorest_20_affected', 'rest_population', 'rest_affected']
    for col in required_cols:
        if col not in gdf.columns:
            print(f"Error: Column '{col}' not found in GeoDataFrame.")
            return
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

    # Create total population and total affected columns for sorting if they don't exist
    # For this plot, we are interested in the distribution of the poorest 20% affected
    # relative to their population, and the rest of the population affected relative to theirs.

    # We need to sort regions by their poverty exposure ratio or a similar metric to make the Lorenz curve meaningful
    # For simplicity, let's sort by the share of poorest 20% exposed in each NUTS-3 region
    gdf["share_poorest20_exposed"] = (gdf["poorest_20_affected"] / gdf["poorest_20_population"]).replace([np.inf, -np.inf], 0).fillna(0) * 100
    gdf_sorted = gdf.sort_values(by="share_poorest20_exposed", ascending=True)

    # Calculate cumulative sums for the poorest 20%
    cum_poorest_pop = np.cumsum(gdf_sorted["poorest_20_population"]) / gdf_sorted["poorest_20_population"].sum()
    cum_poorest_affected = np.cumsum(gdf_sorted["poorest_20_affected"]) / gdf_sorted["poorest_20_affected"].sum()

    # Calculate cumulative sums for the total population (poorest 20% + rest)
    gdf_sorted["total_pop_group"] = gdf_sorted["poorest_20_population"] + gdf_sorted["rest_population"]
    gdf_sorted["total_aff_group"] = gdf_sorted["poorest_20_affected"] + gdf_sorted["rest_affected"]
    
    cum_total_pop = np.cumsum(gdf_sorted["total_pop_group"]) / gdf_sorted["total_pop_group"].sum()
    cum_total_affected = np.cumsum(gdf_sorted["total_aff_group"]) / gdf_sorted["total_aff_group"].sum()


    plt.figure(figsize=(8, 8))
    plt.plot(cum_total_pop, cum_total_affected, label="Total Population Exposure", color='blue', linestyle='--')
    plt.plot(cum_poorest_pop, cum_poorest_affected, label="Poorest 20% Exposure", color='red')
    plt.plot([0, 1], [0, 1], linestyle=':', color='black', label="Line of Perfect Equality")

    # plt.title(f"Lorenz Curve of Flood Exposure ({ssp}, {year}, {rp})", pad=15) # Removed title from top
    plt.xlabel("Cumulative Share of Population")
    plt.ylabel("Cumulative Share of Affected Population")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('square') # Ensure a square plot for better Lorenz curve interpretation
    # Add title to the bottom
    plt.figtext(0.5, 0.01, f"Lorenz Curve of Flood Exposure ({ssp}, {year}, {rp})", ha="center", fontsize=12)


    out_name = f"fig3_4_lorenz_curve_{ssp}_{year}_{rp}.png"
    fig_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {fig_path}")

    # Generate detailed text file
    txt_out_name = f"fig3_4_lorenz_curve_details_{ssp}_{year}_{rp}.txt"
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)
    with open(txt_fig_path, "w") as f:
        f.write(f"Detailed Analysis of Lorenz Curve for Flood Exposure\\n")
        f.write(f"Scenario: {ssp}, Year: {year}, Return Period: {rp}\\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write(f"This document provides a detailed breakdown of the data used to generate the Lorenz curve, \\n")
        f.write(f"showing the distribution of flood-affected population relative to the total population, \\n")
        f.write(f"with a special focus on the poorest 20% of the population.\\n\\n")

        f.write(f"Input GeoJSON file: {geojson_path}\\n\\n")

        f.write(f"Summary Statistics:\\n")
        total_population_overall = gdf["poorest_20_population"].sum() + gdf["rest_population"].sum()
        total_affected_overall = gdf["poorest_20_affected"].sum() + gdf["rest_affected"].sum()
        total_poorest_20_population = gdf["poorest_20_population"].sum()
        total_poorest_20_affected = gdf["poorest_20_affected"].sum()
        total_rest_population = gdf["rest_population"].sum()
        total_rest_affected = gdf["rest_affected"].sum()

        f.write(f"  Total Population (Overall): {total_population_overall:,.0f}\\n")
        f.write(f"  Total Affected Population (Overall): {total_affected_overall:,.0f}\\n")
        f.write(f"  Percentage Affected (Overall): { (total_affected_overall / total_population_overall) * 100 if total_population_overall else 0:.2f}%\\n\\n")

        f.write(f"  Poorest 20% Population: {total_poorest_20_population:,.0f}\\n")
        f.write(f"  Poorest 20% Affected: {total_poorest_20_affected:,.0f}\\n")
        f.write(f"  Percentage of Poorest 20% Affected: { (total_poorest_20_affected / total_poorest_20_population) * 100 if total_poorest_20_population else 0:.2f}%\\n\\n")

        f.write(f"  Rest of Population: {total_rest_population:,.0f}\\n")
        f.write(f"  Rest of Population Affected: {total_rest_affected:,.0f}\\n")
        f.write(f"  Percentage of Rest of Population Affected: { (total_rest_affected / total_rest_population) * 100 if total_rest_population else 0:.2f}%\\n\\n")

        f.write(f"Interpretation Notes for the Lorenz Curve:\\n")
        f.write(f"1. Line of Perfect Equality (Black Dashed Line): Represents a scenario where flood exposure is perfectly proportional to population distribution. \\n")
        f.write(f"   If the curve follows this line, it means that X% of the population accounts for X% of the flood-affected individuals.\\n")
        f.write(f"2. Total Population Exposure Curve (Blue Dashed Line): Shows the cumulative share of the total affected population against the cumulative share of the total population, \\n")
        f.write(f"   when regions are sorted by their 'share_poorest20_exposed' ratio (ascending). Deviation from the line of equality indicates inequality in exposure across all population groups.\\n")
        f.write(f"3. Poorest 20% Exposure Curve (Red Line): Shows the cumulative share of the affected poorest 20% population against the cumulative share of the poorest 20% population, \\n")
        f.write(f"   when regions are sorted by their 'share_poorest20_exposed' ratio (ascending). This curve specifically highlights inequality in flood exposure within the poorest segment of the population.\\n")
        f.write(f"   If this curve is further away from the line of equality than the 'Total Population Exposure' curve, it suggests that the poorest 20% are disproportionately affected or that exposure is more concentrated within this group.\\n")
        f.write(f"4. Gini Coefficient (Not explicitly calculated here but can be inferred): The area between the line of perfect equality and the Lorenz curve, as a ratio of the total area under the line of equality, \\n")
        f.write(f"   quantifies the inequality. A larger area (and thus a higher Gini coefficient) indicates greater inequality.\\n\\n")

        f.write(f"Data Points for Curves (Sorted by 'share_poorest20_exposed'):\\n")
        f.write(f"The table below shows the cumulative shares that form the basis of the Lorenz curves. \\n")
        f.write(f"Each row represents a NUTS-3 region, sorted by the percentage of its poorest 20% population that is affected by floods.\\n\\n")

        f.write(f"{'NUTS_ID'.ljust(10)} | {'Poorest Pop Share'.ljust(20)} | {'Poorest Affected Share'.ljust(25)} | {'Total Pop Share'.ljust(20)} | {'Total Affected Share'.ljust(25)} | {'Share Poorest Exposed (%)'.ljust(25)}\\n")
        f.write(f"{'-'*10} | {'-'*20} | {'-'*25} | {'-'*20} | {'-'*25} | {'-'*25}\\n")

        # Recalculate cumulative sums for detailed output, ensuring no division by zero if sums are zero
        sum_poorest_pop = gdf_sorted["poorest_20_population"].sum()
        sum_poorest_affected = gdf_sorted["poorest_20_affected"].sum()
        sum_total_pop_group = gdf_sorted["total_pop_group"].sum()
        sum_total_aff_group = gdf_sorted["total_aff_group"].sum()

        # Use a temporary DataFrame for cumulative calculations to avoid SettingWithCopyWarning
        temp_df = gdf_sorted[['nuts_id', 'poorest_20_population', 'poorest_20_affected', 'total_pop_group', 'total_aff_group', 'share_poorest20_exposed']].copy()

        temp_df["cum_poorest_pop_abs"] = temp_df["poorest_20_population"].cumsum()
        temp_df["cum_poorest_affected_abs"] = temp_df["poorest_20_affected"].cumsum()
        temp_df["cum_total_pop_abs"] = temp_df["total_pop_group"].cumsum()
        temp_df["cum_total_affected_abs"] = temp_df["total_aff_group"].cumsum()

        for i in range(len(temp_df)):
            nuts_id = temp_df["nuts_id"].iloc[i]
            cpp = temp_df["cum_poorest_pop_abs"].iloc[i] / sum_poorest_pop if sum_poorest_pop else 0
            cpa = temp_df["cum_poorest_affected_abs"].iloc[i] / sum_poorest_affected if sum_poorest_affected else 0
            ctp = temp_df["cum_total_pop_abs"].iloc[i] / sum_total_pop_group if sum_total_pop_group else 0
            cta = temp_df["cum_total_affected_abs"].iloc[i] / sum_total_aff_group if sum_total_aff_group else 0
            spe = temp_df["share_poorest20_exposed"].iloc[i]

            f.write(f"{str(nuts_id).ljust(10)} | {f'{cpp:.4f}'.ljust(20)} | {f'{cpa:.4f}'.ljust(25)} | {f'{ctp:.4f}'.ljust(20)} | {f'{cta:.4f}'.ljust(25)} | {f'{spe:.2f}%'.ljust(25)}\\n")

        f.write(f"\\nEnd of Report.\\n")
    print(f"✓ Saved detailed text file {txt_fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Figure 3-4: Lorenz-style curve for socio-economic differential.")
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--year", required=True, type=int, help="Year (e.g. 2050)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    geojson_fp = build_paths(args.ssp, args.year, args.rp)
    plot_lorenz_curve(geojson_fp, args.ssp, args.year, args.rp)
