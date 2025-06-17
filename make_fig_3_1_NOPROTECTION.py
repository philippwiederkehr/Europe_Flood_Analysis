#!/usr/bin/env python3
"""
Make Figure 3-1 – Choropleth of % poorest-20 exposed (NOPROTECTION scenario).

Usage (example):
    python make_fig_3_1_NOPROTECTION.py --ssp SSP2 --year 2020 --rp RP100

Available columns in the GeoDataFrame: Index(['region_name', 'nuts_id', 'nuts_level', 'total_population',
'total_affected', 'percentage_affected', 'poorest_10_population',
'poorest_20_population', 'rest_population', 'poorest_10_affected',
'poorest_20_affected', 'rest_affected', '0.0-0.5m', '0.5-1.0m',
'1.0-2.0m', '2.0-4.0m', '4.0-6.0m', '>6.0m', '0.0-0.5m_poorest_10',
'0.5-1.0m_poorest_10', '1.0-2.0m_poorest_10', '2.0-4.0m_poorest_10',
'4.0-6.0m_poorest_10', '>6.0m_poorest_10', '0.0-0.5m_poorest_20',
'0.5-1.0m_poorest_20', '1.0-2.0m_poorest_20', '2.0-4.0m_poorest_20',
'4.0-6.0m_poorest_20', '>6.0m_poorest_20', '0.0-0.5m_rest',
'0.5-1.0m_rest', '1.0-2.0m_rest', '2.0-4.0m_rest', '4.0-6.0m_rest',
'>6.0m_rest', 'country_median_income', 'country_10th_percentile',
'country_20th_percentile', 'vulnerable_pop_thresh',
'vulnerable_pct_thresh', 'income_threshold_vuln', 'geometry',
'share_poorest20_exposed'],
dtype='object')
"""
import argparse
import os
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")          # headless / SLURM safe
import matplotlib.pyplot as plt
import mapclassify # Added import
import pandas as pd

# Set sans-serif font globally for the script
plt.rcParams['font.family'] = 'sans-serif'

# ----------------------------------------------------------------------
# Hard-wired roots (same style as flood_population_analysis_new.py)
BASE_OUT_DIR = "/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations_NOPROTECTION" # Updated for NOPROTECTION
FIG_DIR      = "/hdrive/all_users/wiederkehr/analysis/bachelor/"        # <-- new folder
os.makedirs(FIG_DIR, exist_ok=True)

def build_paths(ssp: str, year: str, rp: str):
    results_dir = os.path.join(BASE_OUT_DIR, f"results_{ssp}_{year}_{rp}")
    geojson = os.path.join(results_dir, f"analysis_results_{ssp}_{year}_{rp}.geojson")
    csv     = os.path.join(results_dir, f"detailed_analysis_results_{ssp}_{year}_{rp}.csv")
    return geojson, csv

def plot_poorest20_share(geojson_path: str, ssp: str, year: str, rp: str):
    gdf = gpd.read_file(geojson_path)

    # Add the share if it is not already present
    if "share_poorest20_exposed" not in gdf.columns:
        gdf["share_poorest20_exposed"] = (
            gdf["poorest_20_affected"] / gdf["poorest_20_population"]
        ) * 100  # Multiply by 100 to convert proportion to percentage

    # --- Generate TXT description ---
    data_col_name = "share_poorest20_exposed"
    valid_data = gdf[data_col_name].dropna()

    # Descriptive statistics
    stats_lines = [
        f"   - Total number of regions in dataset: {len(gdf)}",
        f"   - Number of regions with valid '{data_col_name}' data: {valid_data.count()}",
        f"   - Number of regions with no data for '{data_col_name}': {gdf[data_col_name].isnull().sum()}"
    ]
    if not valid_data.empty:
        stats_lines.extend([
            f"   - Minimum exposure: {valid_data.min():.2f}% (in regions with data)",
            f"   - Maximum exposure: {valid_data.max():.2f}% (in regions with data)",
            f"   - Mean exposure: {valid_data.mean():.2f}%",
            f"   - Median exposure: {valid_data.median():.2f}%",
            f"   - Standard Deviation: {valid_data.std():.2f}%",
            f"   - 25th Percentile (Q1): {valid_data.quantile(0.25):.2f}%",
            f"   - 75th Percentile (Q3): {valid_data.quantile(0.75):.2f}%"
        ])
    else:
        stats_lines.append("   - No valid data available for statistical summary.")
    
    summary_stats_str = "\\n".join(stats_lines)

    # Classification for legend
    legend_classes_str = "   - Classification not applicable (e.g., no data or single unique value)."
    k_classes = 5 # Number of classes for NaturalBreaks
    if not valid_data.empty and valid_data.nunique() >= k_classes: # Need enough unique values for k classes
        try:
            classifier = mapclassify.NaturalBreaks(valid_data, k=k_classes)
            legend_classes = []
            # Iterate through the bins and format them
            for i in range(len(classifier.bins)):
                lower_bound = classifier.bins[i-1] if i > 0 else valid_data.min()
                upper_bound = classifier.bins[i]
                # For the first class, ensure the lower bound is the actual minimum
                if i == 0:
                    lower_bound = valid_data.min()
                # For the last class, ensure the upper bound is the actual maximum
                if i == len(classifier.bins) -1: # This is the last upper bound from classifier.bins
                     upper_bound = valid_data.max()

                # Count how many regions fall into this class
                if i == 0: # First class
                    count = valid_data[(valid_data >= lower_bound) & (valid_data <= classifier.bins[i])].count()
                else: # Subsequent classes
                    count = valid_data[(valid_data > classifier.bins[i-1]) & (valid_data <= classifier.bins[i])].count()

                # Format the class range string
                class_range_str = f"{lower_bound:.0f}% - {upper_bound:.0f}%"
                if i == 0 and lower_bound == upper_bound : # handles case where min value is a class itself
                     class_range_str = f"{lower_bound:.0f}%"
                elif lower_bound == upper_bound: # handles case where a bin edge is repeated (e.g. single value class)
                     class_range_str = f"{upper_bound:.0f}%"

                legend_classes.append(f"     - Class {i+1}: {class_range_str} (contains {count} regions)")
            legend_classes_str = "\\n".join(legend_classes)
            
        except Exception as e:
            legend_classes_str = f"   - Could not determine legend classes due to: {e}"
            
    elif not valid_data.empty and valid_data.nunique() > 0 :
         legend_classes_str = f"   - Data has {valid_data.nunique()} unique value(s), less than {k_classes} classes. Min: {valid_data.min():.0f}%, Max: {valid_data.max():.0f}%."


    # Top N regions
    top_n = 5
    top_regions_lines = [f"   - Top {top_n} regions with highest '{data_col_name}':"]
    if not valid_data.empty:
        gdf_sorted = gdf.sort_values(by=data_col_name, ascending=False).dropna(subset=[data_col_name])
        for i in range(min(top_n, len(gdf_sorted))):
            row = gdf_sorted.iloc[i]
            region_name = row.get('region_name', 'N/A')
            nuts_id = row.get('nuts_id', 'N/A')
            value = row[data_col_name]
            top_regions_lines.append(f"     {i+1}. {region_name} ({nuts_id}): {value:.2f}%")
    else:
        top_regions_lines.append("     - No data to determine top regions.")
    top_regions_str = "\n".join(top_regions_lines)

    # Regions with zero exposure
    zero_exposure_lines = [f"   - Regions with zero exposure ({data_col_name} = 0%):"]
    if not valid_data.empty:
        zero_exposure_gdf = gdf[gdf[data_col_name] == 0].dropna(subset=[data_col_name])
        zero_exposure_count = len(zero_exposure_gdf)
        zero_exposure_lines.append(f"     - Count: {zero_exposure_count}")
        if 0 < zero_exposure_count <= 5: # Show individual regions if 5 or fewer
            for i in range(zero_exposure_count):
                row = zero_exposure_gdf.iloc[i]
                region_name = row.get('region_name', 'N/A')
                nuts_id = row.get('nuts_id', 'N/A')
                zero_exposure_lines.append(f"       - {region_name} ({nuts_id})")
        elif zero_exposure_count > 5: # Show examples if more than 5
            zero_exposure_lines.append(f"       - (Examples: {zero_exposure_gdf.iloc[0].get('region_name', 'N/A')} ({zero_exposure_gdf.iloc[0].get('nuts_id', 'N/A')}), ...)")

    else:
        zero_exposure_lines.append("     - No data to determine zero exposure regions.")
    zero_exposure_str = "\n".join(zero_exposure_lines)

    # Regions with 100% exposure
    hundred_exposure_lines = [f"   - Regions with 100% exposure ({data_col_name} = 100%):"]
    if not valid_data.empty:
        hundred_exposure_gdf = gdf[gdf[data_col_name] == 100].dropna(subset=[data_col_name])
        hundred_exposure_count = len(hundred_exposure_gdf)
        hundred_exposure_lines.append(f"     - Count: {hundred_exposure_count}")
        if 0 < hundred_exposure_count <= 5: # Show individual regions if 5 or fewer
            for i in range(hundred_exposure_count):
                row = hundred_exposure_gdf.iloc[i]
                region_name = row.get('region_name', 'N/A')
                nuts_id = row.get('nuts_id', 'N/A')
                hundred_exposure_lines.append(f"       - {region_name} ({nuts_id})")
        elif hundred_exposure_count > 5: # Show examples if more than 5
            hundred_exposure_lines.append(f"       - (Examples: {hundred_exposure_gdf.iloc[0].get('region_name', 'N/A')} ({hundred_exposure_gdf.iloc[0].get('nuts_id', 'N/A')}), ...)")
    else:
        hundred_exposure_lines.append("     - No data to determine 100% exposure regions.")
    hundred_exposure_str = "\n".join(hundred_exposure_lines)

    description = f"""Figure Description: Percentage of Poorest 20% Exposed to Flooding (NOPROTECTION Scenario)
====================================================================
Analysis Parameters: SSP: {ssp}, Year: {year}, Return Period: {rp}, Data Source: NOPROTECTION
Figure Title: "% of Poorest 20 % Exposed – {year}, {rp}, {ssp} (NOPROTECTION)"
Source GeoJSON file: {os.path.basename(geojson_path)} (from NOPROTECTION dataset)
====================================================================

1. General Information:
   - This document describes a choropleth map visualizing the percentage of the poorest 20% of the population exposed to flooding within various NUTS regions, based on the NOPROTECTION scenario.
   - The 'share_poorest20_exposed' is calculated as: (poorest_20_affected / poorest_20_population) * 100.

2. Data Summary for '{data_col_name}':
{summary_stats_str}

3. Map Classification and Legend:
   - Visualization Type: Choropleth map.
   - Classification Scheme for Colors: Natural Breaks (Jenks).
   - Number of Classes: {k_classes} (intended).
   - Color Palette: OrRd (Orange-Red gradient, typically light for low values, dark for high values).
   - Legend Classes (Share Exposed %, rounded to nearest integer for legend display):
{legend_classes_str}
   - Regions with no data or where the share could not be calculated are typically shown in light grey.
   - Region Borders: Black, linewidth 0.2.

4. Notable Observations from the Data:
{top_regions_str}

{zero_exposure_str}

{hundred_exposure_str}

5. Key Data Columns Used:
   - 'poorest_20_affected': Number of people in the poorest 20% income group affected by flooding in each region.
   - 'poorest_20_population': Total number of people in the poorest 20% income group in each region.
   - 'nuts_id': NUTS (Nomenclature of Territorial Units for Statistics) identifier for the region.
   - 'region_name': Name of the NUTS region.
   - 'geometry': Defines the geographical boundaries of the regions for plotting.

6. Purpose of the Figure:
   - To provide a clear visual representation of the spatial distribution of flood exposure specifically among the poorest 20% of the population under the NOPROTECTION scenario.
   - This helps in identifying regions where this socio-economically vulnerable segment is disproportionately impacted by flood events under the specified scenario ({ssp}, {year}, {rp}) without existing flood protection measures.
   - The map can aid in pinpointing areas that may require targeted socio-economic support, tailored flood mitigation strategies, and policy interventions focused on enhancing equity in disaster risk reduction, particularly where protection is absent.

This description is auto-generated based on the data used for the plot.
"""

    txt_out_name = f"fig3_1_poorest20_share_{ssp}_{year}_{rp}_NOPROTECTION_description.txt" # Updated filename
    txt_fig_path = os.path.join(FIG_DIR, txt_out_name)
    with open(txt_fig_path, "w", encoding='utf-8') as f:
        f.write(description)
    print(f"✓ Saved description {txt_fig_path}")
    # --- End of TXT description generation ---

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    gdf.plot(
        column=data_col_name,        scheme="NaturalBreaks", 
        k=k_classes,        cmap="OrRd",
        edgecolor="black",
        linewidth=0.2,
        legend=True,
        legend_kwds={"fmt": "{:.0f} %", "title": "Share Exposed (%)"},
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "none",
            "label": "No data"},
    )
    ax.set_axis_off()
    # Figure title remains the same, NOPROTECTION is implied by data source and filename
    fig.suptitle(f"% of Poorest 20 % Exposed – {year}, {rp}, {ssp}", y=0.08, fontsize=14) 

    out_name = f"fig3_1_poorest20_share_{ssp}_{year}_{rp}_NOPROTECTION.png" # Updated filename
    fig_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {fig_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ssp",  required=True, help="SSP scenario (e.g. SSP2)")
    p.add_argument("--year", required=True, help="Year (e.g. 2020)")
    p.add_argument("--rp",   required=True, help="Return-period string (e.g. RP100)")
    args = p.parse_args()

    geojson_fp, csv_fp = build_paths(args.ssp, args.year, args.rp)
    if not os.path.exists(geojson_fp):
        print(f"Error: GeoJSON file not found at {geojson_fp}")
        exit() # Or handle error appropriately
    plot_poorest20_share(geojson_fp, args.ssp, args.year, args.rp)
