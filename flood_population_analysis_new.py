print("Imported libraries...")
# Standard library imports
import os
import platform
import re
import signal
import sys
import time
import traceback
import argparse # Add this import
import gc

# Third-party imports
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling # Added import
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
import rioxarray
import seaborn as sns
from shapely.geometry import box
import xarray as xr


def print_memory_usage(label="Current"):
    """Print current memory usage with optional label"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"MEMORY [{label}]: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")

def print_timestamp(label="Current operation", start_time=None):
    """Print current time or elapsed time since start_time with a descriptive label"""
    from datetime import datetime
    current_time = datetime.now()
    
    if start_time is None:
        # Just print current timestamp
        print(f"TIME [{current_time.strftime('%H:%M:%S')}] {label}")
        return current_time
    else:
        # Print elapsed time
        elapsed = current_time - start_time
        elapsed_seconds = elapsed.total_seconds()
        if elapsed_seconds < 60:
            time_str = f"{elapsed_seconds:.2f} seconds"
        elif elapsed_seconds < 3600:
            time_str = f"{elapsed_seconds/60:.2f} minutes"
        else:
            time_str = f"{elapsed_seconds/3600:.2f} hours"
        
        print(f"TIME [{current_time.strftime('%H:%M:%S')}] {label} - took {time_str}")
        return current_time

# Configure seaborn for headless environments
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': False})
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# Set up signal handling for graceful termination
def signal_handler(sig, frame):
    signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
    print(f'Script received {signal_name}, cleaning up...')
    
    # Ensure log file is flushed
    sys.stdout.flush()
    if 'log_file_handle' in globals() and log_file_handle and not log_file_handle.closed:
        log_file_handle.flush()
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Starting script...")

def load_population_data(file_path, bounds=None):
    print(f"\n==== LOADING POPULATION DATA ====")
    print(f"Population file: {file_path}")
    print(f"Requested bounds: {bounds}")
    try:
        with rasterio.open(file_path) as src:
            print(f"\nPopulation dataset metadata:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height} pixels")
            print(f"  Resolution: {src.res} degrees per pixel")
            print(f"  Transform: {src.transform}")
            print(f"  Reported bounds: {src.bounds}")
            print(f"  Nodata value: {src.nodata}")
            print(f"  Data type: {src.dtypes[0]}")
            
            
            # The global population dataset has these specific bounds
            global_bounds = (-180.000000, -72.000417, 180.000000, 83.999583)
            print(f"\nGlobal population bounds: {global_bounds}")
            
            # Create corrected transform
            corrected_transform = transform_from_bounds(
                global_bounds[0], global_bounds[1], 
                global_bounds[2], global_bounds[3],
                src.width, src.height
            )
            print(f"Corrected transform: {corrected_transform}")
            
            # Process window if bounds are provided
            if bounds:
                print(f"Reading population data in bounds: {bounds}")
                min_lon, min_lat, max_lon, max_lat = bounds
                
                # Make sure bounds are within the global extent
                min_lon = max(min_lon, global_bounds[0])
                min_lat = max(min_lat, global_bounds[1])
                max_lon = min(max_lon, global_bounds[2])
                max_lat = min(max_lat, global_bounds[3])
                
                print(f"Adjusted bounds: ({min_lon}, {min_lat}, {max_lon}, {max_lat})")
                
                try:
                    # Use the corrected transform to create the window
                    window = from_bounds(min_lon, min_lat, max_lon, max_lat, corrected_transform)
                    print(f"Window: {window}")
                    
                    # Check if window has a valid size
                    if window.width <= 0 or window.height <= 0:
                        raise ValueError(f"Invalid window size: {window.width} x {window.height}")
                    
                    # Read at full resolution without downsampling
                    print(f"Reading window at full resolution")
                    data = src.read(1, window=window)
                    
                    print(f"Original data sum: {np.nansum(data)}")
                    
                    # Mask nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Create an xarray DataArray with full resolution
                    height, width = data.shape
                    x_coords = np.linspace(min_lon, max_lon, width)
                    y_coords = np.linspace(max_lat, min_lat, height)
                    
                    pop_data = xr.DataArray(
                        data[np.newaxis, :, :],
                        dims=['band', 'y', 'x'],
                        coords={
                            'band': [1],
                            'y': y_coords,
                            'x': x_coords
                        }
                    )
                    
                    # Add CRS information
                    pop_data.rio.write_crs(src.crs, inplace=True)
                    
                    print(f"Population data loaded with shape: {pop_data.shape}")
                    print(f"Full resolution total population: {float(pop_data.sum().values)}")
                    
                    print_memory_usage("After loading population data")
                    return pop_data
                    
                except Exception as e:
                    print(f"Error reading window: {str(e)}")
                    print("Falling back to reading a subsample of the entire dataset")
            
            print("ERROR: Bounds not provided or invalid.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading population data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_flood_data(file_path, bounds=None):
    print(f"\n==== LOADING FLOOD DATA ====")
    print(f"Flood file: {file_path}")
    print(f"Requested bounds: {bounds}")
    
    try:
        # Use rioxarray's direct loading instead of manual coordinate creation which was needed with the population dataset because of the "weird" global bounds
        flood_data = rioxarray.open_rasterio(file_path, masked=True)
        
        # Print basic metadata
        print(f"\nFlood dataset metadata:")
        print(f"  CRS: {flood_data.rio.crs}")
        print(f"  Shape: {flood_data.shape}")
        print(f"  Resolution: {flood_data.rio.resolution()}")
        print(f"  Bounds: {flood_data.rio.bounds()}")
        
        # Clip to bounds
        if bounds:
            print(f"Clipping flood data to bounds: {bounds}")
            flood_data = flood_data.rio.clip_box(*bounds)
        
        # Print loaded data info
        print(f"Flood data loaded with shape: {flood_data.shape}")
        print(f"Flood data CRS: {flood_data.rio.crs}")
        print(f"Flood data bounds: {flood_data.rio.bounds()}")
        
        # Additional statistics to get an overview
        valid_data = flood_data.values[~np.isnan(flood_data.values)]
        if len(valid_data) > 0:
            print(f"\nFlood data statistics:")
            print(f"  Value range: {np.nanmin(valid_data)} to {np.nanmax(valid_data)}")
            print(f"  Median depth: {np.nanmedian(valid_data)}")
            print(f"  Mean depth: {np.nanmean(valid_data)}")
            print(f"  Cells with depth > 0.5m: {np.sum(valid_data > 0.5)}")
            print(f"  Cells with depth > 1.0m: {np.sum(valid_data > 1.0)}")
        
        return flood_data
            
    except Exception as e:
        print(f"Error loading flood data: {str(e)}")
        traceback.print_exc()
        return None

def load_protection_data(file_path, bounds=None):
    print(f"\n==== LOADING PROTECTION DATA ====")
    print(f"Protection file: {file_path}")
    print(f"Requested bounds: {bounds}")
    
    try:
        protection_data = rioxarray.open_rasterio(file_path, masked=True)
        print(f"Protection data loaded successfully. Original CRS: {protection_data.rio.crs}")
        print(f"Protection data original bounds (in original CRS): {protection_data.rio.bounds()}")

        target_crs = "EPSG:4326" # Target CRS for alignment with other datasets

        # Reproject if CRS is different from target_crs
        if str(protection_data.rio.crs).upper() != target_crs and protection_data.rio.crs is not None:
            print(f"Reprojecting protection data from {protection_data.rio.crs} to {target_crs}...")
            protection_data = protection_data.rio.reproject(target_crs)
            print(f"Protection data reprojected. New CRS: {protection_data.rio.crs}")
            print(f"Protection data bounds after reprojection (in {target_crs}): {protection_data.rio.bounds()}")
        elif protection_data.rio.crs is None:
            print(f"ERROR: Protection data CRS is None.")
            sys.exit(1)


        if bounds:
            print(f"Attempting to clip protection data (now in {protection_data.rio.crs}) with bounds: {bounds}")
            protection_data = protection_data.rio.clip_box(*bounds)
            print(f"Protection data clipped. New bounds: {protection_data.rio.bounds()}")
            
        print(f"Protection data loaded with shape: {protection_data.shape}")
        return protection_data
            
    except Exception as e:
        print(f"Error loading protection data: {str(e)}")
        traceback.print_exc()
        return None

def load_income_data(file_path, bounds=None):
    print(f"\n==== LOADING INCOME DATA ====")
    print(f"Income file: {file_path}")
    print(f"Requested bounds: {bounds}")
    
    try:
        # Use rioxarray's direct loading instead of manual coordinate creation
        income_data = rioxarray.open_rasterio(file_path, masked=True)
        
        # Clip to bounds if provided
        if bounds:
            income_data = income_data.rio.clip_box(*bounds)
            
        # Print loaded data info
        print(f"Income data loaded with shape: {income_data.shape}")
        print(f"Income data CRS: {income_data.rio.crs}")
        print(f"Income data bounds: {income_data.rio.bounds()}")
        print(f"Income Y range: {float(income_data.y.min().values)} to {float(income_data.y.max().values)}")
        
        return income_data
            
    except Exception as e:
        print(f"Error loading income data: {str(e)}")
        traceback.print_exc()
        return None

def align_datasets(pop_data, flood_data, income_data, protection_data):
    print(f"\n==== ALIGNING DATASETS ====")
    print(f"Population data: {pop_data.shape} with CRS {pop_data.rio.crs}")
    print(f"Flood data: {flood_data.shape} with CRS {flood_data.rio.crs}")
    print(f"Income data: {income_data.shape} with CRS {income_data.rio.crs}")
    print(f"Protection data: {protection_data.shape} with CRS {protection_data.rio.crs}")
    
    # Get resolutions of each dataset
    pop_res = pop_data.rio.resolution()
    flood_res = flood_data.rio.resolution()
    income_res = income_data.rio.resolution()
    protection_res = protection_data.rio.resolution()
    
    print(f"Population data resolution: {pop_res}")
    print(f"Flood data resolution: {flood_res}")
    print(f"Income data resolution: {income_res}")
    print(f"Protection data resolution: {protection_res}")
    
    # Calculate resolution as the average of x and y resolution
    pop_avg_res = (abs(pop_res[0]) + abs(pop_res[1])) / 2
    flood_avg_res = (abs(flood_res[0]) + abs(flood_res[1])) / 2
    income_avg_res = (abs(income_res[0]) + abs(income_res[1])) / 2
    protection_avg_res = (abs(protection_res[0]) + abs(protection_res[1])) / 2
    
    # Determine which dataset has the finest resolution
    resolutions = {
        'population': pop_avg_res,
        'flood': flood_avg_res,
        'income': income_avg_res,
        'protection': protection_avg_res
    }
    
    finest_dataset = min(resolutions, key=resolutions.get)
    print(f"Dataset with finest resolution: {finest_dataset} ({resolutions[finest_dataset]} degrees/pixel)")
    
    # Set the reference dataset for reprojection
    if finest_dataset == 'population':
        reference_data = pop_data
    elif finest_dataset == 'flood':
        reference_data = flood_data
    elif finest_dataset == 'protection':
        reference_data = protection_data
    else:
        reference_data = income_data
    
    print(f"Using {finest_dataset} data as reference for all reprojections")
    
    # Reproject all datasets to match the reference
    try:
        # Reproject datasets that are not the reference
        reprojected_datasets = []
        
        if finest_dataset != 'population':
            print(f"Reprojecting population data to match {finest_dataset} resolution...")
            
            pop_sum_before = float(pop_data.sum().values)
            pop_data = pop_data.rio.reproject_match(reference_data, resampling=rasterio.enums.Resampling.bilinear)
            pop_sum_after = float(pop_data.sum().values)

            # Attempt to conserve population counts if reprojection changes them significantly due to resolution differences
            if pop_sum_before > 0 and pop_sum_after > 0:
                scaling_factor = pop_sum_before / pop_sum_after
                if abs(scaling_factor - 1.0) > 0.01: # Only adjust if difference is more than 1%
                    print(f"Scaling reprojected population data by factor: {scaling_factor:.4f}")
                    pop_data = pop_data * scaling_factor
            
            print(f"Population data shape after reprojection: {pop_data.shape}")
            print(f"Population sum before: {pop_sum_before}, after: {float(pop_data.sum().values)}")
            reprojected_datasets.append('population')

        if finest_dataset != 'flood':
            print(f"Reprojecting flood data to match {finest_dataset} resolution...")
            flood_data = flood_data.rio.reproject_match(reference_data, resampling=rasterio.enums.Resampling.bilinear)
            print(f"Flood data shape after reprojection: {flood_data.shape}")
            reprojected_datasets.append('flood')

        if finest_dataset != 'income':
            print(f"Reprojecting income data to match {finest_dataset} resolution...")
            income_data = income_data.rio.reproject_match(reference_data, resampling=rasterio.enums.Resampling.bilinear)
            print(f"Income data shape after reprojection: {income_data.shape}")
            reprojected_datasets.append('income')

        if finest_dataset != 'protection':
            print(f"Reprojecting protection data to match {finest_dataset} resolution...")
            # Use nearest neighbor for protection data as it represents classes (RP years)
            protection_data = protection_data.rio.reproject_match(reference_data, resampling=rasterio.enums.Resampling.nearest)
            print(f"Protection data shape after reprojection: {protection_data.shape}")
            reprojected_datasets.append('protection')
        
        if reprojected_datasets:
            print(f"Successfully reprojected datasets: {', '.join(reprojected_datasets)}")
        else:
            print(f"No reprojection needed, all datasets already match the finest resolution")
        
        return pop_data, flood_data, income_data, protection_data
            
    except Exception as e:
        print(f"Error aligning datasets: {str(e)}")
        traceback.print_exc()
        
        return pop_data, flood_data, income_data, protection_data

def calculate_country_income_stats(income_data_aligned, pop_data_aligned, all_nuts_regions_gdf):
    """
    Calculates population-weighted median, 10th percentile, and 20th percentile
    income for each country based on the provided income and population data, and NUTS regions.
    These statistics are calculated for the portion of the country covered by the aligned data.

    Args:
        income_data_aligned (xr.DataArray): Aligned income data.
        pop_data_aligned (xr.DataArray): Aligned population data.
        all_nuts_regions_gdf (gpd.GeoDataFrame): GeoDataFrame containing all NUTS regions
                                                 (must have 'CNTR_CODE' and 'geometry').

    Returns:
        dict: A dictionary mapping country codes to a sub-dictionary with
              'median_income', 'percentile_10th', and 'percentile_20th' values.
    """
    print("\\n==== CALCULATING COUNTRY-LEVEL WEIGHTED INCOME STATISTICS ====")
    country_stats = {}

    if income_data_aligned is None or pop_data_aligned is None or all_nuts_regions_gdf is None or all_nuts_regions_gdf.empty:
        print("Error: Income data, Population data, or NUTS GDF is missing for country income stats calculation.")
        sys.exit(1)

    if 'CNTR_CODE' not in all_nuts_regions_gdf.columns:
        print("Error: 'CNTR_CODE' not found in NUTS GeoDataFrame.")
        sys.exit(1)

    if income_data_aligned.rio.crs is None:
        print("Error: Aligned income data is missing CRS information.")
        sys.exit(1)
    if pop_data_aligned.rio.crs is None:
        print("Error: Aligned population data is missing CRS information.")
        sys.exit(1)

    unique_country_codes = all_nuts_regions_gdf['CNTR_CODE'].unique()
    print(f"Found {len(unique_country_codes)} unique country codes in NUTS data for income stats calculation.")

    for country_code in unique_country_codes:
        if pd.isna(country_code):
            print(f"Skipping country {country_code} due to NaN value.")
            continue
        
        print(f"  Processing country for weighted income statistics: {country_code}")
        print_memory_usage(f"Before processing country {country_code}")
        country_specific_nuts = all_nuts_regions_gdf[all_nuts_regions_gdf['CNTR_CODE'] == country_code]
        
        if country_specific_nuts.empty:
            print(f"    No NUTS regions found for country {country_code}. Skipping income stats calculation.")
            continue

        country_stats[country_code] = {
            'median_income': np.nan,
            'percentile_10th': np.nan,
            'percentile_20th': np.nan
        }

        try:
            country_geom_dissolved = country_specific_nuts.dissolve(by='CNTR_CODE')
            if country_geom_dissolved.empty or country_geom_dissolved.geometry.iloc[0].is_empty:
                print(f"    Dissolved geometry for {country_code} is empty or invalid. Skipping.")
                continue
            
            country_geometry_for_clip = country_geom_dissolved.geometry

            if country_geometry_for_clip.crs != income_data_aligned.rio.crs:
                print(f"ERROR: Country {country_code} geometry CRS {country_geometry_for_clip.crs} does not match income data CRS {income_data_aligned.rio.crs}.")
                sys.exit(1)

            try:
                income_country_clipped = income_data_aligned.rio.clip(country_geometry_for_clip, all_touched=True)
                pop_country_clipped = pop_data_aligned.rio.clip(country_geometry_for_clip, all_touched=True)
            except rioxarray.exceptions.NoDataInBounds:
                print(f"WARNING: No data found in bounds when clipping for country {country_code}. Stats will be NaN.")
                continue
            
            country_income_values = income_country_clipped.values.flatten()
            country_pop_values = pop_country_clipped.values.flatten()
            
            valid_mask = ~np.isnan(country_income_values) & ~np.isnan(country_pop_values) & (country_pop_values > 0)
            final_income_values = country_income_values[valid_mask]
            final_pop_weights = country_pop_values[valid_mask]
            
            if len(final_income_values) > 0:
                sorted_indices = np.argsort(final_income_values)
                income_sorted = final_income_values[sorted_indices]
                weights_sorted = final_pop_weights[sorted_indices]
                
                cum_weights = np.cumsum(weights_sorted)
                total_weight = cum_weights[-1]
                
                if total_weight > 0:
                    # Median
                    median_idx = np.searchsorted(cum_weights, total_weight / 2.0)
                    if median_idx < len(income_sorted):
                        country_stats[country_code]['median_income'] = income_sorted[median_idx]
                    elif len(income_sorted) > 0:
                        country_stats[country_code]['median_income'] = income_sorted[-1]
                        print(f"    WARNING: Median index {median_idx} out of bounds for {country_code}. Using last income value.")
                    
                    # 10th Percentile
                    p10_idx = np.searchsorted(cum_weights, total_weight * 0.10)
                    if p10_idx < len(income_sorted):
                        country_stats[country_code]['percentile_10th'] = income_sorted[p10_idx]
                    elif len(income_sorted) > 0:
                         country_stats[country_code]['percentile_10th'] = income_sorted[-1] # Fallback
                         print(f"    WARNING: 10th percentile index {p10_idx} out of bounds for {country_code}. Using last income value.")

                    # 20th Percentile
                    p20_idx = np.searchsorted(cum_weights, total_weight * 0.20)
                    if p20_idx < len(income_sorted):
                        country_stats[country_code]['percentile_20th'] = income_sorted[p20_idx]
                    elif len(income_sorted) > 0:
                        country_stats[country_code]['percentile_20th'] = income_sorted[-1] # Fallback
                        print(f"    WARNING: 20th percentile index {p20_idx} out of bounds for {country_code}. Using last income value.")

                    print(f"    Weighted Income Stats for {country_code} (within study area bounds):")
                    print(f"      Median: {country_stats[country_code]['median_income']:,.2f}")
                    print(f"      10th Percentile: {country_stats[country_code]['percentile_10th']:,.2f}")
                    print(f"      20th Percentile: {country_stats[country_code]['percentile_20th']:,.2f}")
                else:
                    print(f"    WARNING: No positive population weight found for {country_code} after clipping. Stats not calculated.")
            else:
                print(f"    WARNING: No valid overlapping income/population data found for {country_code} after clipping. Stats not calculated.")
        
        except Exception as e:
            print(f"    Error processing country {country_code} for weighted income stats: {e}")
            traceback.print_exc()
        print_memory_usage(f"After processing country {country_code}")
        gc.collect() 

    print(f"Calculated weighted income statistics for {len(country_stats)} countries.")
    print_memory_usage("After calculating country weighted income stats")
    return country_stats

def main(population_file_path: str, income_file_path: str, flood_file_path: str, base_output_dir=None, 
         nuts_level=None):
    """
    Main function to analyze population affected by flooding by depth category and income level
    """
    start_time = print_timestamp("Starting main analysis")
    
    # Add specific SLURM termination handling
    if 'SLURM_JOB_ID' in os.environ:
        print(f"Running as SLURM job {os.environ['SLURM_JOB_ID']}")
        print("Registering additional safeguards for SLURM environment")
        
        # Define a SLURM-specific signal handler that logs the event
        def slurm_signal_handler(sig, frame):
            signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
            print(f"\\n[SLURM TERMINATION] Received {signal_name} in job {os.environ.get('SLURM_JOB_ID', 'unknown')}")
            print("[SLURM TERMINATION] Flushing and closing log file")
            sys.stdout.flush()
            
            # Safety: directly flush the global log file handle if available
            log_handle = globals().get('log_file_handle')
            if log_handle and not log_handle.closed:
                log_handle.flush()
                log_handle.write("\\n[SLURM TERMINATION] Job ended by SLURM\\n")
                log_handle.close()
                
            sys.exit(1)
            
        # Register the SLURM-specific handler
        signal.signal(signal.SIGTERM, slurm_signal_handler)
    
    # Automatically determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get basename of the flood file, used later
    current_flood_filename = os.path.basename(flood_file_path)
    
    # Protection file and NUTS regions file remain the same, relative to script_dir
    protection_file = os.path.join(script_dir, 'floodProtection_v2019_paper', 'floodProtection_v2019_paper3.tif')
    nuts_file = os.path.join(script_dir, 'NUTS_RG_01M_2024_4326.geojson')
    
    # --- Construct the dynamic output directory name ---
    ssp_name = "unknown_ssp"
    year = "unknown_year"
    flood_event_name = "unknown_event"

    try:
        pop_basename = os.path.basename(population_file_path)
        # Expects format like "SSPx_xxxx.tif"
        name_parts = pop_basename.split('.')[0].split('_') 
        if len(name_parts) >= 1:
            ssp_name = name_parts[0]
        if len(name_parts) >= 2:
            year = name_parts[1]
        print(f"Parsed from population file ('{pop_basename}'): SSP='{ssp_name}', Year='{year}'")
    except Exception as e:
        print(f"Warning: Could not parse SSP/year from population file '{population_file_path}': {e}")

    try:
        flood_basename = os.path.basename(flood_file_path)
        # E.g., "Europe_RP10_filled_depth.tif" -> "RP10"
        name_parts = flood_basename.split('.')[0].split('_')
        found_rp = False
        for part in name_parts:
            if part.startswith("RP") and len(part) > 2 and part[2:].isdigit():
                flood_event_name = part
                found_rp = True
                break
        if not found_rp: # Fallback if specific RPxxx pattern not found
            # Attempt to extract a meaningful part if "RP" is not present or not in expected format
            if "Europe" in name_parts and len(name_parts) > 1: # e.g. Europe_EVENT_something
                flood_event_name = name_parts[1]
            elif len(name_parts) > 0: # Take the first part if no "Europe"
                 flood_event_name = name_parts[0]
            else: # Use the whole filename without extension if no underscores
                 flood_event_name = flood_basename.split('.')[0]
        print(f"Parsed from flood file ('{flood_basename}'): Event='{flood_event_name}'")
    except Exception as e:
        print(f"Warning: Could not parse flood event from flood file '{flood_file_path}': {e}")

    if base_output_dir is None:
        # Default base directory for all results if not specified by the user
        base_output_dir = os.path.join(script_dir, 'analysis_runs_output') 
    
    # Construct specific output directory name for this combination
    specific_run_output_dir_name = f"results_{ssp_name}_{year}_{flood_event_name}"
    output_dir_for_this_run = os.path.join(base_output_dir, specific_run_output_dir_name)
    
    os.makedirs(output_dir_for_this_run, exist_ok=True)
    
    # Set up logging to capture console output to a file in the new specific directory
    log_file = setup_logging(output_dir_for_this_run) 
    
    # Print the paths being used
    print(f"Using population data: {population_file_path}")
    print(f"Using flood data: {flood_file_path}")
    print(f"Using income data: {income_file_path}")
    print(f"Using protection data: {protection_file}")
    print(f"Using NUTS regions: {nuts_file}")
    print(f"Output will be saved to: {output_dir_for_this_run}")
    

    region_bounds = (-10.001249957162784, 35.99958337736567, 30.007083227302346, 71.14124989996125)
    region_name = "Mainland Europe"

    print(f"Analysis region: {region_name}")
    print(f"Bounds: {region_bounds}")
    
    # Load all datasets
    loading_start = print_timestamp("Starting data loading")
    pop_data = load_population_data(population_file_path, bounds=region_bounds)
    print_timestamp("Loaded population data", loading_start)

    flood_data = load_flood_data(flood_file_path, bounds=region_bounds)
    print_timestamp("Loaded flood data", loading_start)

    income_data = load_income_data(income_file_path, bounds=region_bounds)
    print_timestamp("Loaded income data", loading_start)
    
    protection_data = load_protection_data(protection_file, bounds=region_bounds)
    print_timestamp("Loaded protection data", loading_start)

    # Load NUTS regions data once in main
    nuts_data_loaded = gpd.read_file(nuts_file)
    print(f"Loaded {len(nuts_data_loaded)} NUTS regions from main.")

    print_memory_usage("After loading all datasets")
    
    if pop_data is not None and flood_data is not None and income_data is not None and protection_data is not None and nuts_data_loaded is not None:
        # Align all datasets
        align_start = print_timestamp("Starting dataset alignment")
        pop_data, flood_data, income_data, protection_data = align_datasets(pop_data, flood_data, income_data, protection_data)
        print_memory_usage("After dataset alignment")
        print_timestamp("Completed dataset alignment", align_start)

        # Calculate country-level income stats using the aligned income_data and pop_data
        country_income_statistics = calculate_country_income_stats(income_data, pop_data, nuts_data_loaded)
        if not country_income_statistics:
            print("Warning: Country income statistics could not be calculated. Vulnerability and income analysis might use fallbacks.")
        
        # Apply flood protection filter
        filter_start = print_timestamp("Starting flood protection filter")
        flood_data_protected = apply_protection_filter(flood_data, protection_data, current_flood_filename)
        print_memory_usage("After applying protection filter")
        print_timestamp("Completed flood protection filter", filter_start)

        # Save the protected flood data
        if output_dir_for_this_run:
            protected_flood_output_path = os.path.join(output_dir_for_this_run, f"{os.path.splitext(current_flood_filename)[0]}_protected.tif")
            try:
                if hasattr(flood_data_protected, 'rio'):
                    flood_data_protected.rio.to_raster(protected_flood_output_path, compress='LZW', dtype='float32')
                    print(f"Protected flood data saved to: {protected_flood_output_path}")
                else:
                    print("Warning: flood_data_protected does not have rio accessor, cannot save as raster.")
            except Exception as e:
                print(f"Error saving protected flood data: {str(e)}")
                traceback.print_exc()

        print("All datasets successfully loaded, aligned, and flood data filtered by protection levels")
        
        # Chunk population data for performance
        print("Chunking population data for performance...")
        if hasattr(pop_data, 'chunk'):
             pop_data = pop_data.chunk({'x': 1000, 'y': 1000})
        
        # Visualize data alignment
        vis_start = print_timestamp("Starting visualizations")
        visualize_data_alignment_with_nuts(pop_data, flood_data_protected, income_data, nuts_data_loaded, output_dir_for_this_run) 
        print_timestamp("Completed alignment visualization", vis_start)

        # Run analysis
        analysis_start = print_timestamp("Starting population analysis by depth and income")

        nuts_pop_by_depth_income = analyze_population_by_flood_depth_and_income(
            pop_data, flood_data_protected, income_data=income_data, 
            nuts_data=nuts_data_loaded,
            nuts_level=nuts_level,
            country_income_stats_dict=country_income_statistics # Pass the full stats dict
        )
        print_timestamp("Completed population analysis", analysis_start)
        
        print_memory_usage("After main analysis")

        # Save income-stratified results to GeoJSON for GIS applications
        if output_dir_for_this_run and nuts_pop_by_depth_income is not None:
            print("Saving results to GeoJSON...")
            geojson_path = os.path.join(output_dir_for_this_run, f'analysis_results_{ssp_name}_{year}_{flood_event_name}.geojson')
            nuts_pop_by_depth_income.to_file(geojson_path, driver='GeoJSON')
            print(f"GeoJSON results with income analysis saved to {geojson_path}")
        
            # Save the detailed results to CSV as well
            print("Saving detailed results to CSV...")
            csv_path = os.path.join(output_dir_for_this_run, f'detailed_analysis_results_{ssp_name}_{year}_{flood_event_name}.csv')
            # Drop the geometry column before saving to CSV
            nuts_pop_by_depth_income.drop(columns='geometry', errors='ignore').to_csv(csv_path, index=False)
            print(f"Detailed CSV results saved to {csv_path}")


        if nuts_pop_by_depth_income is not None:
            visualize_population_by_income_and_depth(
                nuts_pop_by_depth_income, output_dir_for_this_run,
                ssp_name=ssp_name, year=year, flood_event_name=flood_event_name
            )
        print_memory_usage("After visualization")

        # Call the refactored visualization function for vulnerable areas
        if nuts_pop_by_depth_income is not None:
            
            visualize_vulnerable_areas(
                results_gdf=nuts_pop_by_depth_income,
                output_dir=output_dir_for_this_run, 
                ssp_name=ssp_name,
                year=year,
                flood_event_name=flood_event_name               
            )
        
        print("\\n====== ANALYSIS COMPLETE ======")
        print(f"Results saved to {output_dir_for_this_run}")
        if log_file: # Check if log_file was successfully created by setup_logging
             print(f"Log file saved to {log_file if isinstance(log_file, str) else getattr(log_file, 'name', 'unknown_log_file_path')}")

        # Final memory usage report
        print_memory_usage("End of script")
    else:
        print("Error: Failed to load required datasets.")
        if pop_data is None:
            print("  - Population data could not be loaded")
        if flood_data is None:
            print("  - Flood data could not be loaded")
        if income_data is None:
            print("  - Income data could not be loaded")
        if protection_data is None:
            print("  - Protection data could not be loaded")
    print_timestamp("Finished main analysis", start_time)

def analyze_population_by_flood_depth_and_income(population_data, flooding_data, income_data, 
                                               nuts_data=None, nuts_level=None,
                                               country_income_stats_dict=None):    
    """
    Analyze how many people are affected by different flood depth ranges with optional income stratification
    Also calculates vulnerability based on country-specific income thresholds (60% of country median) and flood > 0m.
    Income stratification (poorest 10%, 10-20%, rest) is based on country-specific 10th and 20th percentiles.
    """
    print(f"\n==== ANALYZING POPULATION BY FLOOD DEPTH AND INCOME (COUNTRY-SPECIFIC THRESHOLDS) ====")
    
    # Define the flood depth ranges (in meters)
    global depth_ranges
    depth_ranges = [
         (0.0, 0.5),
         (0.5, 1.0),
         (1.0, 2.0),
         (2.0, 4.0),
         (4.0, 6.0),
         (6.0, float('inf'))  # Everything above 6 meters
     ]
    

    if nuts_data is None or nuts_data.empty:
        print("Error: NUTS data not provided to analyze_population_by_flood_depth_and_income.")
        sys.exit(1)
    print(f"Using {len(nuts_data)} NUTS regions provided as GeoDataFrame for analysis.")
    
    # Get NUTS regions at the specified level
    print(f"Selecting all NUTS level {nuts_level} regions in the study area")
    nuts_level_regions = nuts_data[nuts_data['LEVL_CODE'] == nuts_level].copy() # Use nuts_data
    print(f"Found {len(nuts_level_regions)} NUTS level {nuts_level} regions in the dataset")
    
    # Filter regions that intersect with our study area
    x_min, y_min, x_max, y_max = (
        float(population_data.x.min().values),
        float(population_data.y.min().values),
        float(population_data.x.max().values),
        float(population_data.y.max().values)
    )
    study_area_poly = box(x_min, y_min, x_max, y_max)
    nuts_level_regions['intersects_study_area'] = nuts_level_regions.geometry.intersects(study_area_poly)
    study_regions = nuts_level_regions[nuts_level_regions['intersects_study_area']].copy()
    print(f"Found {len(study_regions)} NUTS level {nuts_level} regions that intersect with the study area")

    if len(study_regions) == 0:
        print(f"Warning: No NUTS level {nuts_level} regions found in study area.")
        sys.exit(1)
    
    study_regions_crs = study_regions.crs # Get CRS to pass to worker

    # region_processing_tuples = [] # This list is no longer needed

    # for idx, region_obj_iter in study_regions.iterrows():
    #     region_country_code = region_obj_iter.get('CNTR_CODE')
    #     region_specific_poverty_threshold = np.nan # Initialize with NaN
    #     country_specific_income_10th_percentile = np.nan
    #     country_specific_income_20th_percentile = np.nan

    #     if country_income_stats_dict and region_country_code and region_country_code in country_income_stats_dict:
    #         country_stats = country_income_stats_dict[region_country_code]
    #         median_income = country_stats.get('median_income')
    #         country_specific_income_10th_percentile = country_stats.get('percentile_10th')
    #         country_specific_income_20th_percentile = country_stats.get('percentile_20th')
            
    #         if not pd.isna(median_income) and median_income > 0:
    #             region_specific_poverty_threshold = 0.6 * median_income
    #         else:
    #             print(f"Warning: Median income for country {region_country_code} is invalid ({median_income}). Poverty threshold for region {region_obj_iter.get('NUTS_ID', idx)} will be NaN.")
    #     else:
    #         if not country_income_stats_dict:
    #             print(f"Warning: country_income_stats_dict is missing. Thresholds for region {region_obj_iter.get('NUTS_ID', idx)} will be NaN.")
    #         elif not region_country_code:
    #              print(f"Warning: CNTR_CODE missing for region {region_obj_iter.get('NUTS_ID', idx)}. Thresholds will be NaN.")
    #         else: # country_code not in dict
    #              print(f"Warning: Country code {region_country_code} for region {region_obj_iter.get('NUTS_ID', idx)} not found in income stats. Thresholds will be NaN.")
        
    #     region_processing_tuples.append( 
    #         (idx, region_obj_iter, region_specific_poverty_threshold, 
    #          country_specific_income_10th_percentile, country_specific_income_20th_percentile, study_regions_crs)
    #     ) 
    
    total_regions = len(study_regions) # Get total regions directly

    result_data = []

    start_time = time.time()

    import gc
    gc.collect()  # Force garbage collection
    print_memory_usage("Before starting processing")

    mp_start = print_timestamp("Starting region processing")

    # Using a sequential for loop instead of ThreadPoolExecutor
    print(f"👉 Starting sequential processing for {total_regions} regions")
    
    # Process tasks sequentially
    print("Processing tasks sequentially...")

    for original_idx, region_object in study_regions.iterrows():
        
        # Calculate region_specific_poverty_threshold inside the loop
        region_country_code = region_object.get('CNTR_CODE')
        income_vuln_thresh = np.nan # Initialize with NaN
        country_median_for_region = np.nan
        country_10th_percentile_for_region = np.nan
        country_20th_percentile_for_region = np.nan

        if country_income_stats_dict and region_country_code and region_country_code in country_income_stats_dict:
            country_stats = country_income_stats_dict[region_country_code]
            country_median_for_region = country_stats.get('median_income')
            country_10th_percentile_for_region = country_stats.get('percentile_10th')
            country_20th_percentile_for_region = country_stats.get('percentile_20th')

            if not pd.isna(country_median_for_region) and country_median_for_region > 0:
                income_vuln_thresh = 0.6 * country_median_for_region
            else:
                print(f"Warning: Median income for country {region_country_code} is invalid ({country_median_for_region}). Poverty threshold for region {region_object.get('NUTS_ID', original_idx)} will be NaN.")
        else:
            if not country_income_stats_dict:
                print(f"Warning: country_income_stats_dict is missing. Thresholds for region {region_object.get('NUTS_ID', original_idx)} will be NaN.")
            elif not region_country_code:
                 print(f"Warning: CNTR_CODE missing for region {region_object.get('NUTS_ID', original_idx)}. Thresholds will be NaN.")
            else: # country_code not in dict
                 print(f"Warning: Country code {region_country_code} for region {region_object.get('NUTS_ID', original_idx)} not found in income stats. Thresholds will be NaN.")

        # region_crs is study_regions_crs, defined before the loop
        region_crs = study_regions_crs

        current_region_geometry = region_object.geometry # Assuming region_object is a GeoPandas Series or similar
        region_bounds = current_region_geometry.bounds

        # Clip and load data specifically for the current region
        pop_clipped_loaded = population_data.rio.clip_box(*region_bounds).load()
        flood_clipped_loaded = flooding_data.rio.clip_box(*region_bounds).load()
        income_clipped_loaded = income_data.rio.clip_box(*region_bounds).load()

        # Call process_region_with_income with the pre-clipped and loaded data
        region_result = process_region_with_income(
            region_object, # This is the NUTS region's metadata and geometry
            income_vuln_thresh,
            region_crs,
            pop_clipped_loaded,
            flood_clipped_loaded,
            income_clipped_loaded,
            country_median_for_region, # Pass country median
            country_10th_percentile_for_region, # Pass country 10th percentile
            country_20th_percentile_for_region  # Pass country 20th percentile
        )
        
        if region_result:
            result_data.append(region_result)
        else:
            print(f"Warning: No valid result for region {region_object.get('NUTS_ID', original_idx)}. Skipping.")
            
    print(f"🏁 All regions processed sequentially.")
    print_timestamp("Completed region processing", mp_start)
    print("\\n==== PROCESSING COMPLETED ====")
    
    # Memory cleanup
    gc.collect()
    print_memory_usage("FINAL STATE")

    # Print final processing time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Finished processing {len(result_data)}/{total_regions} regions ({(len(result_data)/total_regions*100):.1f}%) in {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Add debug information before creating the GeoDataFrame
    print(f"Results count: {len(result_data)}")
    print(f"Geometries count: {len(study_regions.geometry.values)}")
    
    # Count duplicate IDs in results
    nuts_ids = [r.get('nuts_id') for r in result_data]
    unique_nuts_ids = set(nuts_ids)
    print(f"Unique region IDs in results: {len(unique_nuts_ids)} out of {len(nuts_ids)}")
    
    # Add after counting unique IDs
    print(f"First few result nuts_ids: {nuts_ids[:5] if nuts_ids else 'No IDs'}")
    print(f"First few region NUTS_IDs: {[r.get('NUTS_ID', 'None') for _, r in list(study_regions.iterrows())[:5]]}")

    # Create case-insensitive region ID mapping
    region_id_to_geom = {}
    for idx, region in study_regions.iterrows():
        nuts_id = region.get('NUTS_ID', f'unknown_{idx}')
        if nuts_id:
            region_id_to_geom[nuts_id] = region.geometry
            # Also add lowercase version
            region_id_to_geom[nuts_id.lower()] = region.geometry

    print(f"Created mapping with {len(region_id_to_geom)} regions")
    
    # Extract only results for regions we have geometries for
    valid_results = [result for result in result_data 
                    if result.get('nuts_id') in region_id_to_geom]
    
    # Create a properly aligned geometry list
    geometries = [region_id_to_geom[result['nuts_id']] for result in valid_results]
    
    # Create result GeoDataFrame with matching data and geometries
    result_gdf = gpd.GeoDataFrame(
        valid_results,
        geometry=geometries,
        crs=study_regions.crs
    )
    # -- REGION PROCESSING SUMMARY REPORT --
    print("\n==== REGION PROCESSING SUMMARY REPORT ====\n")
    
    # Collect all region names from original data
    all_regions = {region.get('NAME_LATN', region.get('NUTS_NAME', f'Unknown-{idx}')): region.get('NUTS_ID', 'Unknown') 
                  for idx, region in study_regions.iterrows()}
    
    # Identify processed regions
    processed_regions = {result['region_name']: result['nuts_id'] for result in valid_results}
    
    # Calculate missing regions (difference between all and processed)
    processed_region_ids = set(r['nuts_id'] for r in valid_results)
    missing_regions = {name: nuts_id for name, nuts_id in all_regions.items() 
                      if nuts_id not in processed_region_ids}
    
    # Generate summary statistics
    print(f"Total regions in study area: {len(all_regions)}")
    print(f"Successfully processed: {len(processed_regions)} ({len(processed_regions)/len(all_regions)*100:.1f}%)")
    print(f"Not processed: {len(missing_regions)} ({len(missing_regions)/len(all_regions)*100:.1f}%)")
    
    # Print list of regions with issues (if any)
    if missing_regions:
        print("\nThe following regions were NOT successfully processed:")
        for i, (name, nuts_id) in enumerate(missing_regions.items(), 1):
            print(f"  {i}. {name} (ID: {nuts_id})")
            
    print("\nSuccessfully processed regions:")
    for i, (name, nuts_id) in enumerate(processed_regions.items(), 1):
        print(f"  {i}. {name} (ID: {nuts_id})")
    
    print("\n==== END OF REGION PROCESSING REPORT ====\n")
    
    return result_gdf

def visualize_population_by_income_and_depth(nuts_pop_by_depth_income, output_dir=None, ssp_name=None, year=None, flood_event_name=None):
    """
    Create focused visualizations of population affected.
    
    Parameters:
    - nuts_pop_by_depth_income: GeoDataFrame with analysis results by region
    - output_dir: Directory to save visualizations
    - region_filter: Optional filter for regions to include in visualization
    - ssp_name: SSP scenario name for the title
    - year: Year of the analysis for the title
    - flood_event_name: Name of the flood event for the title
    """
    print("\n==== VISUALIZING POPULATION BY INCOME (DECILES) AND FLOOD DEPTH ====")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Construct the title based on provided SSP, year, and event name, or fall back to flood_filename
    if ssp_name and year and flood_event_name:
        plot_title = f"Flood Impact Analysis ({ssp_name}, {year}, {flood_event_name})"
    else:
        plot_title = "Flood Impact Analysis"

    filtered_regions = nuts_pop_by_depth_income.copy()
    
    if filtered_regions.empty:
        print(f"Warning: No data available for visualization after filtering. Skipping.")
        return
    else:
        print(f"Visualizing data for: ({len(filtered_regions)} regions)")

    # Define  income categories and corresponding labels
    income_labels = ['Poorest 10%', '10th-20th Percentile', 'Rest (>20%)']
    
    # EXAMPLE VISUALIZATION: Income-stratified flood impact summary chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # First subplot: Affected population by income level
    # Data columns from process_region_with_income:
    # 'poorest_10_affected', 'poorest_20_affected' (0-20th), 'rest_affected'
    affected_poorest_10 = filtered_regions['poorest_10_affected'].sum()
    affected_poorest_20_total = filtered_regions['poorest_20_affected'].sum() # This is 0-20th percentile
    affected_next_10 = affected_poorest_20_total - affected_poorest_10       # This is 10th-20th percentile
    affected_rest = filtered_regions['rest_affected'].sum()
    
    total_affected_by_income = [affected_poorest_10, affected_next_10, affected_rest]
    total_people_affected = sum(total_affected_by_income)
    
    ax1.pie(
        total_affected_by_income, 
        labels=income_labels,
        autopct='%1.1f%%',
        colors=['#ff6666', '#ffcc66', '#99ff99'],
        shadow=True,
        startangle=90
    )
    ax1.set_title(f'Affected Population by Income Level\nTotal: {total_people_affected:,.0f} people')
    
    # Second subplot: Overall income distribution
    # Data columns: 'poorest_10_population', 'poorest_20_population' (0-20th), 'rest_population'
    pop_poorest_10 = filtered_regions['poorest_10_population'].sum()
    pop_poorest_20_total = filtered_regions['poorest_20_population'].sum() # 0-20th percentile
    pop_next_10 = pop_poorest_20_total - pop_poorest_10                   # 10th-20th percentile
    pop_rest = filtered_regions['rest_population'].sum()
    
    total_pop_by_income = [pop_poorest_10, pop_next_10, pop_rest]
    total_population_sum = sum(total_pop_by_income)

    ax2.pie(
        total_pop_by_income, 
        labels=income_labels,
        autopct='%1.1f%%',
        colors=['#ff6666', '#ffcc66', '#99ff99'],
        shadow=True,
        startangle=90
    )
    ax2.set_title(f'Overall Income Distribution\nTotal: {total_population_sum:,.0f} people')
    
    plt.suptitle(f"{plot_title}\n Income Distribution Analysis", fontsize=16)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{plot_title}_income_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close(fig) 

    gc.collect()

def visualize_data_alignment_with_nuts(pop_data, flood_data, income_data, nuts_data, output_dir=None):
    """
    Create visualizations of datasets with NUTS regions overlaid to verify alignment
    Focused only on Vienna region to save memory
    
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent
    - income_data: xarray DataArray with income data
    - nuts_data: GeoDataFrame with NUTS regions
    - output_dir: directory to save visualizations
    """
    print("\\n==== VISUALIZING DATA ALIGNMENT FOR VIENNA REGION ONLY ====")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    print_memory_usage("Before Vienna alignment check")
    
    # Load NUTS regions
    if nuts_data is not None and not nuts_data.empty:
        nuts_gdf_for_viz = nuts_data # Use passed data
        print(f"Using {len(nuts_gdf_for_viz)} NUTS regions for Vienna alignment check from provided GeoDataFrame")
    else:
        print("NUTS data not provided or empty, skipping alignment check")
        return
    
    print_memory_usage("After loading NUTS regions")
    
    # Create Vienna-specific visualization only
    print("Creating Vienna-specific alignment check...")
    
    # Find Vienna in the NUTS regions
    vienna_regions = nuts_gdf_for_viz[nuts_gdf_for_viz['NAME_LATN'].str.contains('Wien', case=False, na=False) | 
                             nuts_gdf_for_viz['NUTS_NAME'].str.contains('Vienna', case=False, na=False) |
                             nuts_gdf_for_viz['NUTS_ID'].str.startswith('AT13', na=False)]
    
    if len(vienna_regions) > 0:
        print(f"Found {len(vienna_regions)} Vienna region(s) in NUTS data")
        
        # Get Vienna's bounding box with a small buffer
        vienna_bbox = vienna_regions.total_bounds
        buffer_size = 0.05  # Add a buffer around Vienna (in degrees)
        vienna_bounds = (
            vienna_bbox[0] - buffer_size,  # min_x
            vienna_bbox[1] - buffer_size,  # min_y
            vienna_bbox[2] + buffer_size,  # max_x
            vienna_bbox[3] + buffer_size   # max_y
        )
        
        # Create a new figure for Vienna
        fig_vienna, axs_vienna = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Population density in Vienna
        pop_vienna = pop_data.rio.clip_box(*vienna_bounds)
        pop_vienna.plot(ax=axs_vienna[0], cmap='viridis', vmin=0, add_colorbar=True)
        axs_vienna[0].set_title('Vienna: Population Data')
        vienna_regions.boundary.plot(ax=axs_vienna[0], color='red', linewidth=1.0)
        nuts_gdf_for_viz.boundary.plot(ax=axs_vienna[0], color='gray', linewidth=0.3) # Use nuts_gdf_for_viz
        
        # Plot 2: Flood data in Vienna
        flood_vienna = flood_data.rio.clip_box(*vienna_bounds)
        flood_vienna.plot(ax=axs_vienna[1], cmap='Blues', add_colorbar=True)
        axs_vienna[1].set_title('Vienna: Flood Data')
        vienna_regions.boundary.plot(ax=axs_vienna[1], color='red', linewidth=1.0)
        nuts_gdf_for_viz.boundary.plot(ax=axs_vienna[1], color='gray', linewidth=0.3) # Use nuts_gdf_for_viz
        
        # Plot 3: Income data in Vienna
        income_vienna = income_data.rio.clip_box(*vienna_bounds)
        income_vienna.plot(ax=axs_vienna[2], cmap='plasma', add_colorbar=True)
        axs_vienna[2].set_title('Vienna: Income Data')
        vienna_regions.boundary.plot(ax=axs_vienna[2], color='red', linewidth=1.0)
        nuts_gdf_for_viz.boundary.plot(ax=axs_vienna[2], color='gray', linewidth=0.3) # Use nuts_gdf_for_viz
        
        # Add scale information to each plot
        for i, ax in enumerate(axs_vienna):
            ax.text(0.01, 0.01, f"Vienna region", 
                    transform=ax.transAxes, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
            
        plt.tight_layout()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'vienna_data_alignment.png'), dpi=300, bbox_inches='tight')
            print(f"Vienna alignment visualization saved to {os.path.join(output_dir, 'vienna_data_alignment.png')}")
            
        # Close figure to free memory
        plt.close(fig_vienna)
        
    else:
        print("Vienna region not found in NUTS data - skipping Vienna-specific alignment check")
        
    # Free memory
    import gc
    gc.collect()
    print_memory_usage("After Vienna visualization")

def visualize_vulnerable_areas(results_gdf, output_dir=None, ssp_name=None, year=None, flood_event_name=None):
    """
    Visualize areas where flooding exceeds a threshold depth AND income is below a threshold,
    using pre-calculated data from results_gdf.
    """

    print("\\n==== VISUALIZING VULNERABLE AREAS ====")
    print(f"Visualizing based on thresholds: Flood > 0m & Income < 60% of respective country's median income")

    if results_gdf is None or results_gdf.empty:
        print("No data provided for vulnerability visualization. Skipping.")
        return 0
    
    if ssp_name and year and flood_event_name:
        plot_title = f"Vulnerability ({ssp_name}, {year}, {flood_event_name})"
    else:
        plot_title = "Vulnerability Analysis"

    # Ensure required columns exist for vulnerability plotting
    # Expected columns from process_region_with_income: 'vulnerable_pop_thresh', 'total_population', 'vulnerable_pct_thresh'
    required_vuln_cols = ['vulnerable_pop_thresh', 'total_population', 'vulnerable_pct_thresh', 'geometry', 'region_name', 'nuts_id']
    missing_cols = [col for col in required_vuln_cols if col not in results_gdf.columns]
    if missing_cols:
        print(f"ERROR: results_gdf is missing required columns for vulnerability visualization: {missing_cols}")
        print(f"Available columns: {results_gdf.columns.tolist()}")
        return 0 # Or handle error appropriately
    
    # Filter out regions with no population for plotting to avoid division by zero in percentages or an empty plot
    vuln_gdf_plot = results_gdf[results_gdf['total_population'] > 0].copy()

    if vuln_gdf_plot.empty:
        print("No regions with population > 0 to plot for vulnerability. Skipping plot.")
        # Still return the sum from the original GDF if needed
        sys.exit(0) # Exit gracefully if no data to plot


    # Create a visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot the map
    vuln_gdf_plot.plot(
        column='vulnerable_pct_thresh',
        ax=ax,
        legend=True,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        legend_kwds={'label': 'Percentage of Vulnerable Population (%)', 
                     'orientation': 'horizontal',
                     'shrink': 0.8,
                     'pad': 0.05},
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "No Data / Pop Zero",
        }
    )

    flood_condition_text = "Any Flooding (Depth > 0m)"
    income_condition_text = "Income < 60% of Country Median"
    ax.set_title(f"Vulnerable Population by Region\\n({flood_condition_text} & {income_condition_text})", 
                fontsize=14, pad=20)
    
    # Add explanatory text box
    ax.text(0.02, 0.02, 
            f"Percentage of each region's population vulnerable to:\\n"
            f"• {flood_condition_text}\\n"
            f"• {income_condition_text}",
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            verticalalignment='bottom')
    
    # Add analysis info for plotted regions
    total_pop_all_regions_plot = vuln_gdf_plot['total_population'].sum()
    total_vulnerable_pop_all_regions_plot = vuln_gdf_plot['vulnerable_pop_thresh'].sum()
    
    overall_vulnerable_pct_plot = (total_vulnerable_pop_all_regions_plot / total_pop_all_regions_plot * 100) if total_pop_all_regions_plot > 0 else 0
    
    ax.text(0.98, 0.98, 
            f"Overall (plotted regions): {overall_vulnerable_pct_plot:.2f}%\n"
            f"({total_vulnerable_pop_all_regions_plot:,.0f} people) vulnerable",
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
            horizontalalignment='right', verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f'vulnerability_by_region_{plot_title}.png' if plot_title else 'vulnerability_by_region.png'
        plot_path = os.path.join(output_dir, plot_filename)
    
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Vulnerability map saved to {plot_path}")
    
    # Calculate total vulnerable population from the original GDF (before filtering for plot)
    # This gives the sum over ALL regions, even those not plotted (e.g. pop=0)
    total_vulnerable_pop_all_regions_gdf = results_gdf['vulnerable_pop_thresh'].sum()
    print(f"Total vulnerable population across all regions (from GDF): {total_vulnerable_pop_all_regions_gdf:,.0f}")
          
    return total_vulnerable_pop_all_regions_gdf # Return sum from all regions in GDF


def setup_logging(output_dir):
    """
    Set up logging to capture both console output and log to a file
    with immediate flushing and signal handling for proper closure
    """
    import sys
    import signal
    import atexit
    from datetime import datetime
    
    log_file_path = os.path.join(output_dir, f"flood_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Global file handle to ensure it can be accessed by signal handlers
    global log_file_handle
    log_file_handle = open(log_file_path, "w", encoding="utf-8", buffering=1)  # Line buffering
    
    # Create a custom logger that outputs to both console and file with auto-flushing
    class AutoFlushLogger:
        def __init__(self, file_handle):
            self.terminal = sys.stdout
            self.log = file_handle
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            # Flush after every write
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Set up the logger
    sys.stdout = AutoFlushLogger(log_file_handle)
    
    # Define cleanup function
    def cleanup_logging():
        print("\nCleaning up logging resources...", file=sys.__stdout__)  # Use the original stdout
        
        # Safely access the log file
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'log'):
            if not sys.stdout.log.closed:  # Check if file is already closed
                sys.stdout.log.flush()
                sys.stdout.log.close()

                print("Log file closed properly.", file=sys.__stdout__)

            else:
                print("Log file was already closed.", file=sys.__stdout__)
    
    # Register cleanup for normal exits
    atexit.register(cleanup_logging)
    
    # Define signal handlers for various termination signals
    def signal_handler(sig, frame):
        signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
        print(f"\nReceived {signal_name}. Flushing logs before exit...")
        cleanup_logging()
        sys.exit(1)
    
    # Register handlers for common termination signals
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, AttributeError):
            # Some signals might not be available on all platforms
            pass
    
    print(f"Logging to file: {log_file_path}")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Signal handlers registered for proper log flushing on termination")
    print(f"=" * 80)
    
    return log_file_path

def process_region_with_income(
    region, income_threshold_vuln_arg, region_crs_from_main,
    pop_data_for_region,
    flood_data_for_region,
    income_data_for_region,
    country_median_income_arg, # Added: country-level median income
    country_10th_percentile_arg, # Added: country-level 10th percentile income
    country_20th_percentile_arg  # Added: country-level 20th percentile income
):
    """
    Process a single region with income analysis and vulnerability assessment.
    Vulnerability is defined as: income < income_threshold_vuln_arg AND flood_depth > 0.
    Income stratification (poorest 10%, 10-20%, rest) uses country-specific percentiles.
    """
    
    global depth_ranges

    region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
    region_start = print_timestamp(f"Starting processing for region: {region_name}")
    
    output_buffer = [f"\nAnalyzing region: {region_name}"]
    
    region_geom = region.geometry
    
    # Initialize result to return in case of error
    result_row = {
        'region_name': region_name,
        'nuts_id': region.get('NUTS_ID', 'Unknown'),
        'nuts_level': region.get('LEVL_CODE', 'Unknown')
    }
    
    try:
        # Convert region geometry to GeoDataFrame for clipping
        region_gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=region_crs_from_main)
        
        # Clip the data to the region
        pop_region = pop_data_for_region.rio.clip(region_gdf.geometry, region_gdf.crs)
        flood_region = flood_data_for_region.rio.clip(region_gdf.geometry, region_gdf.crs)
        income_region = income_data_for_region.rio.clip(region_gdf.geometry, region_gdf.crs)
        
        # Calculate total population in region
        total_pop = float(pop_region.sum().values)
        output_buffer.append(f"  Total population in region: {total_pop:,.2f}")
        
        # Use country-specific income thresholds for stratification
        income_10th_country_thresh = country_10th_percentile_arg
        income_20th_country_thresh = country_20th_percentile_arg

        output_buffer.append(f"  Using country-specific income thresholds for stratification:")
        output_buffer.append(f"    Country 10th Percentile: {income_10th_country_thresh:.2f}")
        output_buffer.append(f"    Country 20th Percentile: {income_20th_country_thresh:.2f}")

        if pd.isna(income_10th_country_thresh) or pd.isna(income_20th_country_thresh):
            output_buffer.append("  Warning: Country-specific income percentiles are NaN. Cannot stratify by income.")
            # Create dummy masks if percentiles are NaN to avoid errors, all pop will be in 'rest'
            poorest_10_mask = income_region < -np.inf # Effectively false
            poorest_20_mask = income_region < -np.inf # Effectively false
            rest_mask = income_region >= -np.inf    # Effectively true
        else:
            # Create income masks based on country-level percentiles
            poorest_10_mask = income_region <= income_10th_country_thresh
            # poorest_20_mask defines the 0-20th percentile group based on country threshold
            # This means income <= country_20th_percentile_arg
            poorest_20_mask = income_region <= income_20th_country_thresh 
            # Rest is > country_20th_percentile_arg
            rest_mask = income_region > income_20th_country_thresh
        
        # Calculate population by income level (using country-specific thresholds)
        pop_poorest_10 = float(pop_region.where(poorest_10_mask, 0).sum().values)
        pop_poorest_20 = float(pop_region.where(poorest_20_mask, 0).sum().values)
        pop_rest = float(pop_region.where(rest_mask, 0).sum().values)
        
        output_buffer.append(f"  Population by income level:")
        output_buffer.append(f"    Poorest 10% (0-10th percentile): {pop_poorest_10:,.2f} ({(pop_poorest_10/total_pop*100) if total_pop > 0 else 0:.2f}%)")
        output_buffer.append(f"    Poorest 20% (0-20th percentile): {pop_poorest_20:,.2f} ({(pop_poorest_20/total_pop*100) if total_pop > 0 else 0:.2f}%)")
        output_buffer.append(f"    Rest (>20th percentile): {pop_rest:,.2f} ({(pop_rest/total_pop*100) if total_pop > 0 else 0:.2f}%)")
        
        # Initialize depth dictionaries for each income level
        depth_all_income = {}
        depth_poorest_10 = {}
        depth_poorest_20 = {}
        depth_rest = {}
        
        # Calculate population in each flood depth range for each income level
        for min_depth, max_depth in depth_ranges:
            # Create mask for this depth range
            if max_depth == float('inf'):
                depth_mask = flood_region > min_depth
                range_name = f">{min_depth}m"
            else:
                depth_mask = (flood_region > min_depth) & (flood_region <= max_depth)
                range_name = f"{min_depth}-{max_depth}m"
            
            # Calculate affected population (total and by income level)
            affected_pop = float(pop_region.where(depth_mask, 0).sum().values)
            depth_all_income[range_name] = affected_pop
            
            # Calculate for each income level
            affected_poorest_10 = float(pop_region.where(depth_mask & poorest_10_mask, 0).sum().values)
            affected_poorest_20 = float(pop_region.where(depth_mask & poorest_20_mask, 0).sum().values)
            affected_rest = float(pop_region.where(depth_mask & rest_mask, 0).sum().values)
            
            depth_poorest_10[f"{range_name}_poorest_10"] = affected_poorest_10
            depth_poorest_20[f"{range_name}_poorest_20"] = affected_poorest_20
            depth_rest[f"{range_name}_rest"] = affected_rest
            
            poorest_10_pct_of_depth_affected = (affected_poorest_10/affected_pop*100) if affected_pop > 0 else 0.0
            poorest_20_pct_of_depth_affected = (affected_poorest_20/affected_pop*100) if affected_pop > 0 else 0.0
            rest_pct_of_depth_affected = (affected_rest/affected_pop*100) if affected_pop > 0 else 0.0
        
            output_buffer.append(f"  Population affected by {range_name} flooding:")
            output_buffer.append(f"    Total: {affected_pop:,.2f}")
            output_buffer.append(f"    Poorest 10% (0-10th percentile): {affected_poorest_10:,.2f} ({poorest_10_pct_of_depth_affected:.2f}% of affected in this depth range)")
            output_buffer.append(f"    Poorest 20% (0-20th percentile): {affected_poorest_20:,.2f} ({poorest_20_pct_of_depth_affected:.2f}% of affected in this depth range)")
            output_buffer.append(f"    Rest (>20th percentile): {affected_rest:,.2f} ({rest_pct_of_depth_affected:.2f}% of affected in this depth range)")
        
        # Calculate total affected population
        total_affected = float(pop_region.where(flood_region > 0, 0).sum().values)
        percentage_affected = (total_affected / total_pop * 100) if total_pop > 0 else 0
        output_buffer.append(f"  Total affected population: {total_affected:,.2f} ({percentage_affected:.2f}%)")
        
        # Calculate total affected by income level
        total_affected_poorest_10 = float(pop_region.where((flood_region > 0) & poorest_10_mask, 0).sum().values)
        # total_affected_poorest_20 is the total affected population in the 0-20th percentile
        total_affected_poorest_20 = float(pop_region.where((flood_region > 0) & poorest_20_mask, 0).sum().values)
        total_affected_rest = float(pop_region.where((flood_region > 0) & rest_mask, 0).sum().values)
        
        if country_median_income_arg is None:
            output_buffer.append(f"  Warning: No country median income provided for {region_name}.")
            country_median_income_arg = np.nan

        # Store results
        result_row = {
            'region_name': region_name,
            'nuts_id': region.get('NUTS_ID', 'Unknown'),
            'nuts_level': region.get('LEVL_CODE', 'Unknown'),
            'total_population': total_pop, 
            'total_affected': total_affected,
            'percentage_affected': percentage_affected,
            'poorest_10_population': pop_poorest_10,
            'poorest_20_population': pop_poorest_20, # Population in 0-20th percentile (country-based)
            'rest_population': pop_rest,
            'poorest_10_affected': total_affected_poorest_10,
            'poorest_20_affected': total_affected_poorest_20, # Affected in 0-20th percentile (country-based)
            'rest_affected': total_affected_rest,
            **depth_all_income,
            **depth_poorest_10,
            **depth_poorest_20,
            **depth_rest,
            'country_median_income': country_median_income_arg, # Use the passed country median
            'country_10th_percentile': country_10th_percentile_arg,
            'country_20th_percentile': country_20th_percentile_arg
        }

        # Vulnerability: Income < income_threshold_vuln_arg AND Flood Depth > 0m
        vulnerable_pop_specific = np.nan
        vulnerable_pct_specific = np.nan

        if pd.isna(income_threshold_vuln_arg):
            output_buffer.append(f"  Skipping vulnerability calculation: No valid country-specific income threshold provided (is NaN).")
        else:
            output_buffer.append(f"  Calculating vulnerability (flood > 0m & income < {income_threshold_vuln_arg:.2f} PPP [country-specific threshold])")
            
            vuln_flood_mask = flood_region > 0
            vuln_income_mask = income_region < income_threshold_vuln_arg
            vuln_combined_mask = vuln_flood_mask & vuln_income_mask
            
            vulnerable_pop_specific = float(pop_region.where(vuln_combined_mask, 0).sum().values)
            
            if total_pop > 0:
                vulnerable_pct_specific = (vulnerable_pop_specific / total_pop * 100)
            elif vulnerable_pop_specific > 0: # Should not happen if total_pop is 0
                print(f"WARNING: Total population is 0 but vulnerable population is {vulnerable_pop_specific:.2f}. Setting vulnerable percentage to NaN.")
                sys.exit(1)  # Exit if this condition occurs, as it indicates a data issue
            else: # total_pop is 0 and vulnerable_pop_specific is 0
                vulnerable_pct_specific = 0.0


            output_buffer.append(f"    Vulnerable population (flood > 0m, income < {income_threshold_vuln_arg:,.0f} PPP): {vulnerable_pop_specific:,.2f}")
            output_buffer.append(f"    Vulnerable percentage: {vulnerable_pct_specific:.2f}%")
        
        result_row['vulnerable_pop_thresh'] = vulnerable_pop_specific
        result_row['vulnerable_pct_thresh'] = vulnerable_pct_specific
        result_row['income_threshold_vuln'] = income_threshold_vuln_arg
        
    except Exception as e:
        output_buffer.append(f"  Error processing region: {str(e)}")
        import traceback
        output_buffer.append(traceback.format_exc())
        # Ensure vulnerability fields are NaN in case of an error during processing
        result_row['vulnerable_pop_thresh'] = np.nan
        result_row['vulnerable_pct_thresh'] = np.nan
        result_row['income_threshold_vuln'] = np.nan

    
    print("\n".join(output_buffer))
    
    print_memory_usage(f"End processing region {region_name}")
    print_timestamp(f"Completed processing for region: {region_name}", region_start)
    # Force garbage collection before returning
    import gc # Import the gc module
    gc.collect()
    
    return result_row
def apply_protection_filter(hazard_data, protection_data, hazard_filename):
    """
    Apply the binary \"hold vs breach\" filter to the hazard data based on protection levels.

    Args:
        hazard_data (xr.DataArray): The original flood hazard GeoTIFF (water depths).
        protection_data (xr.DataArray): The flood protection GeoTIFF (design return period P in years).
        hazard_filename (str): Filename of the hazard raster, used to extract its RP.

    Returns:
        xr.DataArray: A new hazard map with protection levels applied.
    """
    print(f"\n==== APPLYING FLOOD PROTECTION FILTER ====")
    print(f"Hazard data shape: {hazard_data.shape}")
    print(f"Protection data shape: {protection_data.shape}")
    print(f"Hazard filename: {hazard_filename}")

    # 1. Identify the return period (RP) of the hazard data
    rp_match = re.search(r'RP(\d+)', hazard_filename, re.IGNORECASE)
    if not rp_match:
        print(f"WARNING: Could not parse RP from hazard filename: {hazard_filename}. Assuming RP=0 (no protection effective).")
        sys.exit(1)  # Exit if RP cannot be determined, as this is critical for the filter logic
    else:
        hazard_rp = int(rp_match.group(1))
        print(f"Parsed Hazard Return Period (RP): {hazard_rp} years")

    # Ensure protection_data is in the same shape as hazard_data (it should be after alignment)
    if hazard_data.shape != protection_data.shape:
        print(f"ERROR: Hazard data shape {hazard_data.shape} and Protection data shape {protection_data.shape} do not match!")
        sys.exit(1)  # Exit if shapes do not match, as this is critical for the filter logic

    # Create a copy of the hazard data to modify
    protected_hazard_data = hazard_data.copy()
    
    # Ensure protection_data has no NaNs where hazard_data is valid, fill with 0 (no protection)
    protection_values = protection_data.fillna(0).values 
    hazard_values = protected_hazard_data.values

    # Create the mask
    # Defences hold if protection_level >= hazard_return_period
    defences_hold_mask = protection_values >= hazard_rp
    
    # Apply the filter: set hazard to 0 where defences hold
    # Ensure we are operating on the correct band if multi-band
    print(f"hazard_values shape: {hazard_values.shape}, ndim: {hazard_values.ndim}")
    print(f"defences_hold_mask shape: {defences_hold_mask.shape}, ndim: {defences_hold_mask.ndim}")

    if hazard_values.ndim == 3 and hazard_values.shape[0] == 1: # Single band raster
        hazard_values[0][defences_hold_mask[0]] = np.nan  # Set to NaN where defences hold
        print(f"Applied protection filter to single band raster, shape: {hazard_values[0].shape}")
    elif hazard_values.ndim == 2: # Already 2D array
         hazard_values[defences_hold_mask[0]] = np.nan  # Set to NaN where defences hold
         print(f"Applied protection filter to 2D raster, shape: {hazard_values.shape}")
    else:
        print(f"WARNING: Hazard data has unexpected dimensions {hazard_values.ndim}. Filter may not apply correctly.")
        # Attempt to apply to the first band if it's a multi-band array that wasn't caught
        sys.exit(1)  # Exit if dimensions are unexpected


    protected_hazard_data.values = hazard_values
    
    # Log how many cells were affected
    original_flooded_cells = float((hazard_data > 0).sum())
    protected_flooded_cells = float((protected_hazard_data > 0).sum())
    cells_protected = original_flooded_cells - protected_flooded_cells
    
    print(f"Original number of flooded cells (>0m): {original_flooded_cells}")
    print(f"Number of flooded cells after applying protection: {protected_flooded_cells}")
    if original_flooded_cells > 0:
        print(f"Cells where flood was mitigated by protection: {cells_protected} ({cells_protected/original_flooded_cells*100:.2f}% of originally flooded)")
    else:
        print(f"Cells where flood was mitigated by protection: {cells_protected}")

    print(f"Protected hazard data created with shape: {protected_hazard_data.shape}")
    return protected_hazard_data

if __name__ == "__main__":

    # --- Add argument parsing ---
    parser = argparse.ArgumentParser(description="Run flood population analysis for specified SSP, year, and flood return period.")
    parser.add_argument("--ssps", nargs='+', required=True, help="List of SSP scenarios (e.g., SSP1 SSP2)")
    parser.add_argument("--years", nargs='+', required=True, help="List of years (e.g., 2030 2050)")
    parser.add_argument("--flood_rps", nargs='+', type=int, required=True, help="List of flood return periods (e.g., 30 100)")
    args = parser.parse_args()
    
    # Start with a print statement showing execution environment
    print(f"Starting flood analysis on {platform.node()} with Python {sys.version}")
    print(f"Environment variables: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'Not set')}")
    print(f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Define dataset locations and parameters ---
    population_base_dir = '/hdrive/all_users/wiederkehr/analysis/population_files'
    income_base_dir = '/hdrive/all_users/wiederkehr/analysis/income_files'
    flood_file_dir = '/hdrive/all_users/wiederkehr/analysis/flood_files'

    ssps = args.ssps
    years = args.years
    flood_rps = args.flood_rps

    base_output_directory = '/hdrive/all_users/wiederkehr/analysis/analysis_runs_output_all_combinations'
    os.makedirs(base_output_directory, exist_ok=True)

    # --- Iterate through all combinations ---
    for ssp in ssps:
        ssp_folder_name = ssp # e.g., "SSP1"
        population_ssp_dir = os.path.join(population_base_dir, ssp_folder_name)

        for year in years:
            population_file_name = f"{ssp}_{year}.tif"
            population_file_path = os.path.join(population_ssp_dir, population_file_name)

            income_file_name = f"Europe_disp_inc_{year}_{ssp}.tif"
            income_file_path = os.path.join(income_base_dir, income_file_name)

            # Check if primary files exist before proceeding with flood events
            if not os.path.exists(population_file_path):
                print(f"SKIPPING: Population file not found: {population_file_path}")
                continue
            if not os.path.exists(income_file_path):
                print(f"SKIPPING: Income file not found: {income_file_path}")
                continue

            for rp in flood_rps:
                flood_file_name = f"Europe_RP{rp}_filled_depth.tif"
                flood_file_path = os.path.join(flood_file_dir, flood_file_name)

                if not os.path.exists(flood_file_path):
                    print(f"SKIPPING: Flood file not found: {flood_file_path} for SSP {ssp}, Year {year}")
                    continue
                
                print(f"\\n{'='*30} PROCESSING NEW COMBINATION {'='*30}")
                print(f"SSP: {ssp}, Year: {year}, Flood RP: {rp}")
                print(f"Population: {population_file_path}")
                print(f"Income: {income_file_path}")
                print(f"Flood: {flood_file_path}")
                print(f"{'='*80}\\n")

                try:
                    status = main(
                        population_file_path=population_file_path,
                        income_file_path=income_file_path,
                        flood_file_path=flood_file_path,
                        base_output_dir=base_output_directory,
                        nuts_level=3
                    )
                    if status is not None and status != 0:
                         print(f"WARNING: Main function returned status {status} for {ssp}_{year}_RP{rp}")
                except Exception as e:
                    print(f"CRITICAL ERROR processing {ssp}_{year}_RP{rp}: {str(e)}")
                    traceback.print_exc()
                    print("Continuing with the next combination...")
                
                print(f"\\n{'='*30} COMPLETED COMBINATION: SSP {ssp}, Year {year}, Flood RP {rp} {'='*30}\\n")

    print("\\nAll combinations processed.")
    sys.exit(0)