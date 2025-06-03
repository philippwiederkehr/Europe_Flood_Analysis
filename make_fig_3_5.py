#!/usr/bin/env python3
"""
Make Figure 3-5 – Side-by-side boxplots of NUTS-3 poverty-exposure ratio.

Compares SSP1, SSP2, and SSP3 for a fixed year (2100) and return period (RP100).

Usage (example):
    python make_fig_3_5.py --year 2100 --rp RP100
"""
import argparse
import os
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # headless / SLURM safe
matplotlib.rcParams['font.family'] = 'sans-serif' # Set default font to sans-serif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime # Added for timestamp

# Hard-wired roots
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations"
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"
os.makedirs(FIG_DIR, exist_ok=True)

SSPS = ["SSP1", "SSP2", "SSP3"]

def build_paths(ssp: str, year: int, rp: str):
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    return geojson

def calculate_poverty_exposure_ratio(gdf):
    """Calculates the poverty exposure ratio, handling division by zero."""
    # Calculate share of poorest 20% exposed
    share_poorest20_exposed = (gdf["poorest_20_affected"] / gdf["poorest_20_population"]).replace([np.inf, -np.inf], np.nan) * 100
    
    # Calculate share of total population exposed
    share_total_exposed = (gdf["total_affected"] / gdf["total_population"]).replace([np.inf, -np.inf], np.nan) * 100
    
    # Calculate poverty exposure ratio
    poverty_exposure_ratio = (share_poorest20_exposed / share_total_exposed).replace([np.inf, -np.inf], np.nan)
    return poverty_exposure_ratio

def plot_ssp_comparison_boxplots(year: int, rp: str):
    """
    Generates side-by-side boxplots of poverty-exposure ratio for SSP1, SSP2, SSP3.
    The figure will contain two subplots: one without outliers, one with outliers.
    """
    all_ratios_data = []

    for ssp in SSPS:
        geojson_path = build_paths(ssp, year, rp)
        if not os.path.exists(geojson_path):
            print(f"Warning: GeoJSON not found for {ssp}, {year}, {rp}: {geojson_path}")
            continue
        
        gdf = gpd.read_file(geojson_path)
        
        required_cols = ['poorest_20_affected', 'poorest_20_population', 'total_affected', 'total_population']
        skip_ssp = False
        for col in required_cols:
            if col not in gdf.columns:
                print(f"Warning: Column '{col}' not found in GeoDataFrame for {ssp}, {year}, {rp}. Skipping this SSP.")
                skip_ssp = True
                break
        if skip_ssp:
            continue
            
        gdf["poverty_exposure_ratio"] = calculate_poverty_exposure_ratio(gdf)
        valid_ratios = gdf[["poverty_exposure_ratio"]].dropna()
        valid_ratios["ssp"] = ssp
        all_ratios_data.append(valid_ratios)

    if not all_ratios_data:
        print("No data loaded for any SSP. Cannot generate plot.")
        return

    df_ratios = pd.concat(all_ratios_data)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), sharex=True)
    
    # Subplot 1: Boxplots without outliers
    sns.boxplot(ax=axes[0], data=df_ratios, x="ssp", y="poverty_exposure_ratio", order=SSPS, showfliers=False)
    axes[0].set_title("Poverty Exposure Ratio Distribution (Outliers Hidden for Clarity)")
    axes[0].set_ylabel("Poverty Exposure Ratio")
    axes[0].grid(True, linestyle='--', alpha=0.7, axis='y')

    # Subplot 2: Boxplots with outliers
    sns.boxplot(ax=axes[1], data=df_ratios, x="ssp", y="poverty_exposure_ratio", order=SSPS, showfliers=True)
    axes[1].set_title("Poverty Exposure Ratio Distribution (Including Outliers)")
    axes[1].set_xlabel("SSP Scenario")
    axes[1].set_ylabel("Poverty Exposure Ratio")
    axes[1].grid(True, linestyle='--', alpha=0.7, axis='y')

    # fig.suptitle(f"Poverty Exposure Ratio by SSP Scenario ({year}, {rp})", fontsize=14, y=0.99) # Moved to bottom
    plt.figtext(0.5, 0.01, f"Poverty Exposure Ratio by SSP Scenario ({year}, {rp})", ha="center", fontsize=14) # Title at the bottom
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for bottom title and ensure top title of subplot is visible

    out_name = f"fig3_5_poverty_exposure_ratio_ssp_comparison_{year}_{rp}.png"
    fig_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig) # Close the figure object
    print(f"✓ Saved {fig_path}")

    # --- Generate detailed TXT description ---
    txt_out_name = f"fig3_5_poverty_exposure_ratio_ssp_comparison_details_{year}_{rp}.txt"
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)

    with open(txt_fig_path, "w", encoding='utf-8') as f:
        f.write(f"Detailed Analysis of Poverty Exposure Ratio Comparison by SSP Scenario\\n")
        f.write(f"========================================================================\\n")
        f.write(f"Year: {year}, Return Period: {rp}\\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"SSPs Compared: {f', '.join(SSPS)}\\n")
        f.write(f"Figure Title: Poverty Exposure Ratio by SSP Scenario ({year}, {rp})\\n")
        f.write(f"========================================================================\\n\\n")

        f.write(f"1. General Information:\\n")
        f.write(f"   - This document describes a two-panel figure with side-by-side boxplots comparing the 'Poverty Exposure Ratio' across different SSP scenarios (SSP1, SSP2, SSP3) for NUTS-3 regions.\\n")
        f.write(f"   - Panel 1 (Top): Shows the distribution with outliers hidden to provide a clearer view of the main body of the data (median, quartiles, typical range).\\n")
        f.write(f"   - Panel 2 (Bottom): Shows the full distribution including all outliers, illustrating the complete range and extent of extreme values.\\n")
        f.write(f"   - The Poverty Exposure Ratio is calculated as: \\n")
        f.write(f"     (Share of Poorest 20% Exposed) / (Share of Total Population Exposed)\\n")
        f.write(f"     where Share Exposed = (Affected Population / Total Population) * 100.\\n")
        f.write(f"   - A ratio > 1 suggests that the poorest 20% are disproportionately exposed compared to the general population.\\n")
        f.write(f"   - A ratio < 1 suggests they are less exposed than the general population.\\n")
        f.write(f"   - A ratio = 1 suggests proportional exposure.\\n")
        f.write(f"   - NaN or infinite values in the ratio calculation (e.g., due to zero denominators) are excluded from the boxplot statistics.\\n\\n")

        f.write(f"2. Data Sources and Preprocessing per SSP:\\n")
        for ssp_val in SSPS:
            geojson_path_val = build_paths(ssp_val, year, rp)
            f.write(f"   - SSP: {ssp_val}\\n")
            f.write(f"     - Input GeoJSON: {os.path.basename(geojson_path_val)}\\n")
            if not os.path.exists(geojson_path_val):
                f.write(f"     - Status: File NOT FOUND. This SSP was not included in the plot.\\n")
                continue

            gdf_temp = gpd.read_file(geojson_path_val)
            req_cols_present = True
            for col_check in required_cols:
                if col_check not in gdf_temp.columns:
                    f.write(f"     - Status: Missing required column '{col_check}'. This SSP was likely skipped or had issues.\\n")
                    req_cols_present = False
                    break
            if not req_cols_present:
                continue

            # Recalculate for detailed stats, ensuring consistency
            share_poorest20_exposed_temp = (gdf_temp["poorest_20_affected"] / gdf_temp["poorest_20_population"]).replace([np.inf, -np.inf], np.nan) * 100
            share_total_exposed_temp = (gdf_temp["total_affected"] / gdf_temp["total_population"]).replace([np.inf, -np.inf], np.nan) * 100
            poverty_exposure_ratio_temp = (share_poorest20_exposed_temp / share_total_exposed_temp).replace([np.inf, -np.inf], np.nan)
            valid_ratios_ssp = poverty_exposure_ratio_temp.dropna()

            f.write(f"     - Number of NUTS-3 regions processed: {len(gdf_temp)}\\n")
            f.write(f"     - Number of valid 'Poverty Exposure Ratio' values calculated: {len(valid_ratios_ssp)}\\n")
            num_nan_poorest_share = share_poorest20_exposed_temp.isnull().sum()
            num_nan_total_share = share_total_exposed_temp.isnull().sum()
            num_zero_total_share = (share_total_exposed_temp == 0).sum()
            f.write(f"       - Regions with NaN share_poorest20_exposed (e.g., poorest_20_population is 0): {num_nan_poorest_share}\\n")
            f.write(f"       - Regions with NaN share_total_exposed (e.g., total_population is 0): {num_nan_total_share}\\n")
            f.write(f"       - Regions with share_total_exposed = 0 (leading to NaN/inf ratio if share_poorest20_exposed > 0): {num_zero_total_share}\\n")
            f.write(f"     - Total regions excluded from boxplot due to NaN/inf ratios: {len(gdf_temp) - len(valid_ratios_ssp)}\\n")
        f.write(f"\\n")

        f.write(f"3. Boxplot Interpretation Guide:\\n")
        f.write(f"   The figure contains two panels of boxplots. For each SSP scenario boxplot in both panels:\\n")
        f.write(f"   - The central line in the box represents the Median (50th percentile) of the Poverty Exposure Ratio for NUTS-3 regions.\\n")
        f.write(f"   - The bottom and top edges of the box represent the 1st Quartile (Q1 - 25th percentile) and 3rd Quartile (Q3 - 75th percentile), respectively.\\n")
        f.write(f"   - The Interquartile Range (IQR) is the height of the box (Q3 - Q1).\\n")
        f.write(f"   - The whiskers extend from the box to show the range of the data, typically 1.5 * IQR from the Q1 and Q3 (unless outliers are hidden).\\n")
        f.write(f"   - Top Panel (Outliers Hidden): Points beyond the whiskers are NOT shown, to focus on the central distribution. The y-axis is scaled to the whiskers.\\n")
        f.write(f"   - Bottom Panel (Outliers Shown): Points beyond the whiskers (outliers) ARE plotted individually. The y-axis is scaled to include these outliers, which may make the boxes appear smaller if outliers are extreme.\\n")
        f.write(f"   - A taller box indicates greater variability in the ratio across regions for that SSP.\\n")
        f.write(f"   - A higher median line suggests a general tendency for higher disproportionate exposure of the poorest 20% in that SSP.\\n\\n")

        f.write(f"4. Detailed Statistics for Poverty Exposure Ratio per SSP (based on valid, non-NaN, non-infinite ratios):\\n")
        if df_ratios.empty:
            f.write("   No valid ratio data available to display statistics.\\n")
        else:
            for ssp_val in SSPS:
                ssp_data = df_ratios[df_ratios["ssp"] == ssp_val]["poverty_exposure_ratio"]
                if ssp_data.empty:
                    f.write(f"   SSP {ssp_val}: No valid data for statistics (this SSP might have been skipped or had no valid ratios).\\n")
                    continue
                f.write(f"   SSP {ssp_val}:\\n")
                f.write(f"     - Count of NUTS-3 regions (with valid ratio): {ssp_data.count()}\\n")
                f.write(f"     - Mean Ratio: {ssp_data.mean():.3f}\\n")
                f.write(f"     - Median Ratio (Q2): {ssp_data.median():.3f}\\n")
                f.write(f"     - Standard Deviation: {ssp_data.std():.3f}\\n")
                f.write(f"     - Minimum Ratio: {ssp_data.min():.3f}\\n")
                f.write(f"     - Maximum Ratio: {ssp_data.max():.3f}\\n")
                f.write(f"     - 1st Quartile (Q1 - 25th percentile): {ssp_data.quantile(0.25):.3f}\\n")
                f.write(f"     - 3rd Quartile (Q3 - 75th percentile): {ssp_data.quantile(0.75):.3f}\\n")
                # Identify potential outliers (example: > Q3 + 1.5*IQR or < Q1 - 1.5*IQR)
                q1 = ssp_data.quantile(0.25)
                q3 = ssp_data.quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                outliers = ssp_data[(ssp_data < lower_bound) | (ssp_data > upper_bound)]
                f.write(f"     - Number of potential outliers (beyond 1.5*IQR): {len(outliers)}\\n")
                if not outliers.empty:
                    f.write(f"       - Min/Max of these outliers: {outliers.min():.3f} / {outliers.max():.3f}\\n")

                # Further analysis: How many regions have ratio > 1 (disproportionate exposure)?
                disproportionate_count = ssp_data[ssp_data > 1].count()
                f.write(f"     - Number of regions with Ratio > 1 (disproportionate exposure): {disproportionate_count} ({(disproportionate_count/ssp_data.count())*100 if ssp_data.count() > 0 else 0:.1f}% of regions with valid ratio)\\n")
                # How many regions have ratio < 1 (less exposure)?
                less_exposed_count = ssp_data[ssp_data < 1].count()
                f.write(f"     - Number of regions with Ratio < 1 (less than proportional exposure): {less_exposed_count} ({(less_exposed_count/ssp_data.count())*100 if ssp_data.count() > 0 else 0:.1f}% of regions with valid ratio)\\n")
                # How many regions have ratio = 1 (proportional exposure)?
                equal_exposed_count = ssp_data[ssp_data == 1].count()
                f.write(f"     - Number of regions with Ratio = 1 (proportional exposure): {equal_exposed_count} ({(equal_exposed_count/ssp_data.count())*100 if ssp_data.count() > 0 else 0:.1f}% of regions with valid ratio)\\n\\n")

        f.write(f"5. Potential Observations & Comparisons (to be filled by LLM based on the data above):\\n")
        f.write(f"   - Comparison of Medians: Which SSP shows the highest/lowest median poverty exposure ratio? What does this imply?\\n")
        f.write(f"   - Comparison of IQR/Variability: Which SSP shows the widest/narrowest spread of ratios? What does this indicate about regional disparities within that SSP?\\n")
        f.write(f"   - Outliers: Are there significant outliers in any SSP? What could be the reasons for extreme ratios in certain regions?\\n")
        f.write(f"   - Proportion of Disproportionately Affected Regions: How does the percentage of regions with Ratio > 1 compare across SSPs?\\n")
        f.write(f"   - Overall Trend: Does any SSP consistently show a tendency towards greater or lesser equity in terms of flood exposure for the poorest 20%?\\n\\n")

        f.write(f"6. Data Columns Used from GeoJSONs (per SSP):\\n")
        f.write(f"   - 'poorest_20_affected': Number of people in the poorest 20% income group affected by flooding.\\n")
        f.write(f"   - 'poorest_20_population': Total number of people in the poorest 20% income group.\\n")
        f.write(f"   - 'total_affected': Total number of people (all income groups) affected by flooding.\\n")
        f.write(f"   - 'total_population': Total number of people (all income groups).\\n")
        f.write(f"   - 'nuts_id' (implicitly, as data is per NUTS-3 region).\\n\\n")

        f.write(f"7. Purpose of the Figure & Analysis:\\n")
        f.write(f"   - To visually and statistically compare how different socio-economic pathways (SSPs) influence the relative flood exposure of the poorest 20% of the population compared to the general population, showing both the central tendency and the full range of variation including extreme values.\\n")
        f.write(f"   - This helps in understanding potential future inequalities in flood risk under various development scenarios and can inform policy decisions aimed at equitable climate adaptation and disaster risk reduction.\\n\\n")

        f.write(f"End of Detailed Report.\\n")

    print(f"✓ Saved detailed text file {txt_fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Figure 3-5: Boxplots of poverty-exposure ratio by SSP.")
    # Year and RP are fixed as per the plan for this specific figure, but kept as args for potential flexibility
    p.add_argument("--year", required=True, type=int, help="Year (e.g. 2100)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    # Validate that the arguments match the plan for this figure
    if args.year != 2100 or args.rp != "RP100":
        print(f"Warning: This script is intended for year 2100 and RP100 as per the plan for Figure 3.5.")
        print(f"You provided year={args.year} and rp={args.rp}.")
        # Proceeding with provided arguments, but be aware of the plan.

    plot_ssp_comparison_boxplots(args.year, args.rp)
