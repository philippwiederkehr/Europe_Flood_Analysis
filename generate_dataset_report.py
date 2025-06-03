\
import os
import glob
import pandas as pd
import geopandas as gpd
import re # For parsing directory names
import traceback # For more detailed error logging if needed

# --- Configuration ---
# Assuming this script is in the 'Analysis' directory
BASE_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_OUTPUT_DIR = os.path.join(BASE_SCRIPT_DIR, 'analysis_runs_output') # Old path
BASE_OUTPUT_DIR = '/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations' # New, absolute path
OUTPUT_TEXT_FILE = os.path.join(BASE_SCRIPT_DIR, 'dataset_summary_for_llm.txt')

# Expected depth ranges and corresponding column name parts
DEPTH_RANGES_INFO = [
    {"label": "0.0-0.5m", "col_suffix": "0.0_0.5"},
    {"label": "0.5-1.0m", "col_suffix": "0.5_1.0"},
    {"label": "1.0-2.0m", "col_suffix": "1.0_2.0"},
    {"label": "2.0-4.0m", "col_suffix": "2.0_4.0"},
    {"label": "4.0-6.0m", "col_suffix": "4.0_6.0"},
    {"label": ">6.0m", "col_suffix": "6.0_inf"},
]

# Expected income category column names (these should be exact)
INCOME_CATEGORIES_INFO = [
    {"label": "Poorest 10% Affected", "col_name": "pop_income_poorest_10_affected"},
    {"label": "10-20% Poorest Affected", "col_name": "pop_income_10_20_poorest_affected"},
    {"label": "Rest (>20th percentile) Affected", "col_name": "pop_income_rest_affected"},
]

# Expected column names for key metrics
# These are primary names; fallbacks can be added in the summarization logic if needed.
VULNERABLE_POP_COL = "vulnerable_pop_low_income_and_flooded"
TOTAL_POP_IN_REGION_COL = "total_population_in_region" # Or a common alternative like 'total_pop'
TOTAL_POP_AFFECTED_COL = "total_pop_affected" # Or 'total_affected_pop'

def parse_run_details_from_dir_name(dir_name):
    match = re.match(r"results_(.*?)_(.*?)_(.*)", dir_name)
    if match:
        ssp = match.group(1)
        year = match.group(2)
        event = match.group(3)
        return ssp, year, event
    print(f"Warning: Could not parse run details from directory name: {dir_name}")
    return "unknown_ssp", "unknown_year", "unknown_event"

def find_column_or_fallback(df_columns, primary_name, fallbacks=None):
    """Checks for a primary column name, then tries fallbacks."""
    if primary_name in df_columns:
        return primary_name
    if fallbacks:
        for fallback in fallbacks:
            if fallback in df_columns:
                return fallback
    return None

def summarize_run_data(csv_path, geojson_path, ssp, year, event):
    summary_parts = []
    summary_parts.append(f"### Run: SSP: {ssp}, Year: {year}, Event: {event}")
    summary_parts.append(f"**Source Directory:** {os.path.dirname(csv_path)}")
    summary_parts.append(f"**CSV File:** {os.path.basename(csv_path)}")
    summary_parts.append(f"**GeoJSON File:** {os.path.basename(geojson_path) if geojson_path else 'N/A (not found or not used for this summary)'}")
    summary_parts.append("")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            summary_parts.append("Result: CSV file is empty.")
            return "\\n".join(summary_parts)

        df_cols = df.columns

        # Infer NUTS level
        nuts_level_inferred = "Unknown"
        if geojson_path and os.path.exists(geojson_path):
            try:
                gdf = gpd.read_file(geojson_path)
                if not gdf.empty and 'LEVL_CODE' in gdf.columns:
                    levels = gdf['LEVL_CODE'].unique()
                    valid_levels = [lvl for lvl in levels if pd.notna(lvl)]
                    if len(valid_levels) == 1:
                        nuts_level_inferred = f"NUTS {int(valid_levels[0])}"
                    elif len(valid_levels) > 1:
                        nuts_level_inferred = f"Multiple NUTS levels ({[int(l) for l in valid_levels]})"
                elif not gdf.empty and 'NUTS_ID' in gdf.columns: # Fallback to NUTS_ID length
                    sample_id_len = gdf['NUTS_ID'].astype(str).str.len().median()
                    if pd.notna(sample_id_len):
                        lvl = int(sample_id_len) - 2 # Approximation: DE=0, DE1=1, DE11=2, DE111=3
                        nuts_level_inferred = f"NUTS {lvl} (inferred from NUTS_ID length in GeoJSON)"
            except Exception as e:
                summary_parts.append(f"- Note: Could not read NUTS level from GeoJSON: {str(e)[:100]}")
        
        if nuts_level_inferred == "Unknown" and 'LEVL_CODE' in df_cols: # Try from CSV
            levels = df['LEVL_CODE'].unique()
            valid_levels = [lvl for lvl in levels if pd.notna(lvl)]
            if len(valid_levels) == 1:
                nuts_level_inferred = f"NUTS {int(valid_levels[0])} (from CSV)"
            elif len(valid_levels) > 1:
                 nuts_level_inferred = f"Multiple NUTS levels ({[int(l) for l in valid_levels]}) (from CSV)"
        elif nuts_level_inferred == "Unknown" and 'NUTS_ID' in df_cols: # Fallback to NUTS_ID length from CSV
            sample_id_len = df['NUTS_ID'].astype(str).str.len().median()
            if pd.notna(sample_id_len):
                lvl = int(sample_id_len) - 2
                nuts_level_inferred = f"NUTS {lvl} (inferred from NUTS_ID length in CSV)"


        summary_parts.append(f"**NUTS Level for Aggregation (Inferred):** {nuts_level_inferred}")
        summary_parts.append(f"**Overall Statistics for this Run:**")
        summary_parts.append(f"- Number of NUTS regions analyzed: {len(df)}")

        actual_total_pop_col = find_column_or_fallback(df_cols, TOTAL_POP_IN_REGION_COL, ['total_pop', 'total_population'])
        total_pop_in_regions_val = df[actual_total_pop_col].sum() if actual_total_pop_col else None
        
        if actual_total_pop_col and total_pop_in_regions_val is not None:
            summary_parts.append(f"- Total population in these regions: {total_pop_in_regions_val:,.0f}")
        else:
            summary_parts.append(f"- Total population in these regions: N/A (column like '{TOTAL_POP_IN_REGION_COL}' not found)")

        actual_total_affected_col = find_column_or_fallback(df_cols, TOTAL_POP_AFFECTED_COL, ['total_affected', 'sum_total_pop_affected'])
        total_pop_affected_run_val = df[actual_total_affected_col].sum() if actual_total_affected_col else None

        if actual_total_affected_col and total_pop_affected_run_val is not None:
            summary_parts.append(f"- Total population affected by flooding: {total_pop_affected_run_val:,.0f}")
            if total_pop_in_regions_val is not None and total_pop_in_regions_val > 0:
                perc_affected = (total_pop_affected_run_val / total_pop_in_regions_val) * 100
                summary_parts.append(f"- Percentage of total population affected: {perc_affected:.2f}%")
            else:
                summary_parts.append(f"- Percentage of total population affected: N/A (total population in regions is zero or N/A)")
        else:
            summary_parts.append(f"- Total population affected by flooding: N/A (column like '{TOTAL_POP_AFFECTED_COL}' not found)")
            summary_parts.append(f"- Percentage of total population affected: N/A")
        
        summary_parts.append("")
        summary_parts.append("**Affected Population by Flood Depth (Total for Run):**")
        found_depth_cols = []
        for range_info in DEPTH_RANGES_INFO:
            # Construct expected column name, e.g., pop_depth_0.0_0.5
            col_name_candidate = f"pop_depth_{range_info['col_suffix']}"
            if col_name_candidate in df_cols:
                pop_in_range = df[col_name_candidate].sum()
                summary_parts.append(f"- {range_info['label']}: {pop_in_range:,.0f}")
                found_depth_cols.append(col_name_candidate)
            else:
                summary_parts.append(f"- {range_info['label']}: N/A (column '{col_name_candidate}' not found)")
        
        summary_parts.append("")
        summary_parts.append("**Affected Population by Income Category (Total for Run):**")
        found_income_cols = []
        for cat_info in INCOME_CATEGORIES_INFO:
            col_name = cat_info["col_name"]
            if col_name in df_cols:
                pop_in_cat = df[col_name].sum()
                summary_parts.append(f"- {cat_info['label']}: {pop_in_cat:,.0f}")
                found_income_cols.append(col_name)
            else:
                summary_parts.append(f"- {cat_info['label']}: N/A (column '{col_name}' not found)")

        summary_parts.append("")
        summary_parts.append("**Vulnerable Population (Low Income & Flooded - Total for Run):**")
        actual_vulnerable_col = find_column_or_fallback(df_cols, VULNERABLE_POP_COL, ['vulnerable_pop'])
        if actual_vulnerable_col:
            total_vulnerable_pop_val = df[actual_vulnerable_col].sum()
            summary_parts.append(f"- Total vulnerable: {total_vulnerable_pop_val:,.0f}")
            if total_pop_affected_run_val is not None and total_pop_affected_run_val > 0:
                perc_vulnerable_of_affected = (total_vulnerable_pop_val / total_pop_affected_run_val) * 100
                summary_parts.append(f"- Percentage of affected population that is vulnerable: {perc_vulnerable_of_affected:.2f}%")
            else:
                summary_parts.append(f"- Percentage of affected population that is vulnerable: N/A (total affected pop is zero or N/A)")
        else:
            summary_parts.append(f"- Total vulnerable: N/A (column like '{VULNERABLE_POP_COL}' not found)")
            summary_parts.append(f"- Percentage of affected population that is vulnerable: N/A")

        summary_parts.append("")
        summary_parts.append("**Top 5 Affected Countries (by total affected population):**")
        if 'CNTR_CODE' in df_cols and actual_total_affected_col and actual_vulnerable_col:
            # Ensure affected and vulnerable columns are numeric before aggregation
            df_country_summary = df.copy()
            df_country_summary[actual_total_affected_col] = pd.to_numeric(df_country_summary[actual_total_affected_col], errors='coerce').fillna(0)
            df_country_summary[actual_vulnerable_col] = pd.to_numeric(df_country_summary[actual_vulnerable_col], errors='coerce').fillna(0)

            country_summary = df_country_summary.groupby('CNTR_CODE').agg(
                total_affected=(actual_total_affected_col, 'sum'),
                total_vulnerable=(actual_vulnerable_col, 'sum')
            ).sort_values(by='total_affected', ascending=False)

            for i, (index, row) in enumerate(country_summary.head(5).iterrows()):
                summary_parts.append(f"{i+1}. {index}: Affected: {row['total_affected']:,.0f}, Vulnerable: {row['total_vulnerable']:,.0f}")
        else:
            missing_cols_for_country_summary = [_ for _ in ['CNTR_CODE', actual_total_affected_col, actual_vulnerable_col] if _ not in df_cols or _ is None]
            summary_parts.append(f"Country-level summary N/A (missing required columns: {', '.join(missing_cols_for_country_summary)}).")

        summary_parts.append("")
        summary_parts.append("**Key Columns found and used in this CSV summary:**")
        
        used_cols = [_f for _f in [
            'NUTS_ID', 'NAME_LATN', 'CNTR_CODE', 'LEVL_CODE', # Standard NUTS columns
            actual_total_pop_col, actual_total_affected_col, actual_vulnerable_col
        ] if _f and _f in df_cols]
        used_cols.extend([col for col in found_depth_cols if col in df_cols])
        used_cols.extend([col for col in found_income_cols if col in df_cols])
        summary_parts.append(f"`{', '.join(sorted(list(set(used_cols))))}`")

    except FileNotFoundError:
        summary_parts.append(f"Result: Error - CSV file not found at {csv_path}")
    except pd.errors.EmptyDataError:
        summary_parts.append(f"Result: Error - CSV file at {csv_path} is empty.")
    except Exception as e:
        summary_parts.append(f"Result: An error occurred while processing {os.path.basename(csv_path)}: {type(e).__name__} - {str(e)[:200]}")
        # summary_parts.append(f"Traceback: {traceback.format_exc()}") # Uncomment for debugging
    
    return "\\n".join(summary_parts)

def generate_dataset_description():
    if not os.path.exists(BASE_OUTPUT_DIR):
        print(f"Error: Base output directory not found: {BASE_OUTPUT_DIR}")
        # Create an empty report file stating the issue
        with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# Dataset Description Generation Error\n\n")
            f.write(f"The base output directory was not found: {BASE_OUTPUT_DIR}\n")
            f.write("No analysis could be performed.\n")
        return

    all_text_content = []

    all_text_content.append("# Comprehensive Dataset Description for LLM Analysis")
    all_text_content.append("## Dataset Overview")
    all_text_content.append(
        "This document summarizes a dataset of simulated flood exposure and socio-economic vulnerability "
        "analyses for Europe. The analyses cover various combinations of Shared Socioeconomic Pathways (SSPs), "
        "future years, and flood Return Periods (RPs)."
    )
    all_text_content.append(
        "Each run typically produces a detailed CSV and a GeoJSON file with NUTS region-level statistics."
    )
    all_text_content.append("\\n## Original Data Generation Context:")
    all_text_content.append("The results were generated using a Python script named 'flood_population_analysis_new.py'.")
    all_text_content.append("Key inputs to that script likely included:")
    all_text_content.append("- Population projection rasters (e.g., SSPx_yyyy.tif)")
    all_text_content.append("- Flood hazard map rasters (e.g., Europe_RPxx_filled_depth.tif)")
    all_text_content.append("- Income data rasters")
    all_text_content.append("- Flood protection level data (e.g., floodProtection_v2019_paper3.tif)")
    all_text_content.append("- NUTS regional boundaries (e.g., NUTS_RG_01M_2024_4326.geojson)")
    
    all_text_content.append("\\n## General Analysis Parameters (derived from generating script structure):")
    all_text_content.append("- NUTS Level for aggregation: Typically one NUTS level per run (inferred and reported in individual run summaries).")
    depth_labels = [d['label'] for d in DEPTH_RANGES_INFO]
    all_text_content.append(f"- Flood depth categories analyzed (meters): {', '.join(depth_labels)}")
    income_labels = [i['label'].replace(" Affected", "") for i in INCOME_CATEGORIES_INFO] # Cleaner labels
    all_text_content.append(f"- Income categories analyzed: {', '.join(income_labels)} (based on country-specific income percentiles).")
    all_text_content.append("- Vulnerability definition: Population with income below 60% of their national median income AND exposed to any flood depth (>0m).")
    
    all_text_content.append("\\n## Individual Run Summaries:")

    processed_runs_count = 0
    run_directories = sorted([d for d in os.listdir(BASE_OUTPUT_DIR) if d.startswith("results_") and os.path.isdir(os.path.join(BASE_OUTPUT_DIR, d))])

    if not run_directories:
        all_text_content.append("\\nNo result directories matching the pattern 'results_*_*_*' were found in "
                                f"{BASE_OUTPUT_DIR}.")
    
    for dir_name in run_directories:
        ssp, year, event = parse_run_details_from_dir_name(dir_name)
        run_dir_path = os.path.join(BASE_OUTPUT_DIR, dir_name)

        csv_filename_pattern = f"detailed_analysis_results_{ssp}_{year}_{event}.csv"
        csv_glob_pattern = os.path.join(run_dir_path, csv_filename_pattern)
        
        geojson_filename_pattern = f"analysis_results_{ssp}_{year}_{event}.geojson" # Optional for NUTS level inference
        geojson_glob_pattern = os.path.join(run_dir_path, geojson_filename_pattern)

        csv_files_found = glob.glob(csv_glob_pattern)
        geojson_files_found = glob.glob(geojson_glob_pattern)

        if csv_files_found:
            csv_file_path = csv_files_found[0]
            geojson_file_path = geojson_files_found[0] if geojson_files_found else None
            
            all_text_content.append("\\n---\\n") # Separator for runs
            run_summary = summarize_run_data(csv_file_path, geojson_file_path, ssp, year, event)
            all_text_content.append(run_summary)
            processed_runs_count += 1
        else:
            all_text_content.append(f"\\n---\\n### Run: SSP: {ssp}, Year: {year}, Event: {event} (No CSV found)")
            all_text_content.append(f"**Source Directory:** {run_dir_path}")
            all_text_content.append(f"Result: No CSV file found matching pattern '{csv_filename_pattern}'.")
            print(f"Warning: No CSV file found for pattern '{csv_filename_pattern}' in {run_dir_path}")

    if processed_runs_count > 0:
         all_text_content.append(f"\\n\\nProcessed a total of {processed_runs_count} runs.")
    elif run_directories: # Directories existed but no CSVs matched
        all_text_content.append(f"\\n\\nFound {len(run_directories)} potential run directories, but no matching CSV files were processed.")


    all_text_content.append("\\n\\n## General Notes for LLM Interpretation:")
    all_text_content.append("- This summary provides aggregated data per run. The raw CSV/GeoJSON files contain finer per-NUTS region details.")
    all_text_content.append("- Population figures represent counts of individuals.")
    all_text_content.append("- Income categories are relative to national income distributions within each country.")
    all_text_content.append("- Flood depths are reported in meters.")
    all_text_content.append("- This dataset is suitable for analyzing spatial patterns of flood risk, socio-economic disparities in flood exposure, and the potential impacts of different climate (via RPs) and socio-economic development (via SSPs/years) scenarios.")
    all_text_content.append("- Emphasize the multi-scenario nature of the dataset when describing its scope and potential uses.")
    all_text_content.append(f"- This report was generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
            f.write("\\n".join(all_text_content))
        print(f"Successfully generated dataset description: {OUTPUT_TEXT_FILE}")
    except Exception as e:
        print(f"Error writing output file {OUTPUT_TEXT_FILE}: {e}")
        # print(f"Traceback: {traceback.format_exc()}") # Uncomment for debugging

if __name__ == "__main__":
    print(f"Starting dataset description generation...")
    print(f"Base output directory being scanned: {BASE_OUTPUT_DIR}")
    print(f"Report will be saved to: {OUTPUT_TEXT_FILE}")
    generate_dataset_description()
    print(f"Script finished.")
