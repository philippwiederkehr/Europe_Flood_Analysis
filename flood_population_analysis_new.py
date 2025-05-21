# Standard library imports
import multiprocessing
import os
import platform
import re
import signal
import sys
import time
import traceback

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

print("Imported libraries...")
print("Starting script...")

def load_population_data(file_path, bounds=None):
    """
    Load population grid data using rasterio with efficient memory usage
    """
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
            
            # If bounds aren't provided or window reading failed, read the entire dataset
            print(f"Reading full population dataset")
            data = src.read(1)
            
            # Mask nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            # Create an xarray DataArray with corrected coordinates
            height, width = data.shape
            x_coords = np.linspace(global_bounds[0], global_bounds[2], width)
            y_coords = np.linspace(global_bounds[3], global_bounds[1], height)  # Note: y is from top to bottom
            
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
            
            # After creating the DataArray
            print(f"\nFinal population data statistics:")
            print(f"  Shape: {pop_data.shape}")
            print(f"  X coords range: {float(pop_data.x.min().values)} to {float(pop_data.x.max().values)}")
            print(f"  Y coords range: {float(pop_data.y.min().values)} to {float(pop_data.y.max().values)}")
            print(f"  Value range: {float(pop_data.min().values)} to {float(pop_data.max().values)}")
            print(f"  Mean value: {float(pop_data.mean().values)}")
            print(f"  Sum (total population): {float(pop_data.sum().values)}")
            print(f"  Count of pixels: {pop_data.count().values.item()}")
            
            print_memory_usage("After loading population data")
            return pop_data
            
    except Exception as e:
        print(f"Error loading population data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_flood_data(file_path, bounds=None):
    """
    Load flood extent data using rioxarray for simpler and more efficient loading
    """
    print(f"\n==== LOADING FLOOD DATA ====")
    print(f"Flood file: {file_path}")
    print(f"Requested bounds: {bounds}")
    
    try:
        # Use rioxarray's direct loading instead of manual coordinate creation
        flood_data = rioxarray.open_rasterio(file_path, masked=True)
        
        # Print basic metadata
        print(f"\nFlood dataset metadata:")
        print(f"  CRS: {flood_data.rio.crs}")
        print(f"  Shape: {flood_data.shape}")
        print(f"  Resolution: {flood_data.rio.resolution()}")
        print(f"  Bounds: {flood_data.rio.bounds()}")
        
        # Clip to bounds if provided
        if bounds:
            print(f"Clipping flood data to bounds: {bounds}")
            flood_data = flood_data.rio.clip_box(*bounds)
        
        # Print loaded data info
        print(f"Flood data loaded with shape: {flood_data.shape}")
        print(f"Flood data CRS: {flood_data.rio.crs}")
        print(f"Flood data bounds: {flood_data.rio.bounds()}")
        print(f"Flood Y range: {float(flood_data.y.min().values)} to {float(flood_data.y.max().values)}")
        
        # Additional flood-specific statistics
        valid_data = flood_data.values[~np.isnan(flood_data.values)]
        if len(valid_data) > 0:
            print(f"\nFlood data statistics:")
            print(f"  Value range: {np.nanmin(valid_data)} to {np.nanmax(valid_data)}")
            print(f"  Median depth: {np.nanmedian(valid_data)}")
            print(f"  Mean depth: {np.nanmean(valid_data)}")
            print(f"  Cells with depth > 0.5m: {np.sum(valid_data > 0.5)}")
            print(f"  Cells with depth > 1.0m: {np.sum(valid_data > 1.0)}")
            print(f"  NaN count: {np.isnan(valid_data).sum()}")
            
            # Print flood depth distribution
            flood_depth_bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
            print("Flood depth distribution:")
            for i in range(len(flood_depth_bins)-1):
                count = np.sum((valid_data > flood_depth_bins[i]) & (valid_data <= flood_depth_bins[i+1]))
                if flood_depth_bins[i+1] == float('inf'):
                    print(f"  > {flood_depth_bins[i]}m: {count} cells")
                else:
                    print(f"  {flood_depth_bins[i]}-{flood_depth_bins[i+1]}m: {count} cells")
        
        return flood_data
            
    except Exception as e:
        print(f"Error loading flood data: {str(e)}")
        traceback.print_exc()
        return None

def load_protection_data(file_path, bounds=None):
    """
    Load flood protection data using rioxarray.
    """
    print(f"\n==== LOADING PROTECTION DATA ====")
    print(f"Protection file: {file_path}")
    print(f"Requested bounds: {bounds}")
    
    try:
        protection_data = rioxarray.open_rasterio(file_path, masked=True)
        print(f"Protection data loaded successfully. Original CRS: {protection_data.rio.crs}")
        print(f"Protection data original bounds (in original CRS): {protection_data.rio.bounds()}")

        target_crs = "EPSG:4326" # Target CRS for alignment with other datasets and bounds

        # Reproject if CRS is different from target_crs
        if str(protection_data.rio.crs).upper() != target_crs and protection_data.rio.crs is not None:
            print(f"Reprojecting protection data from {protection_data.rio.crs} to {target_crs}...")
            protection_data = protection_data.rio.reproject(target_crs)
            print(f"Protection data reprojected. New CRS: {protection_data.rio.crs}")
            print(f"Protection data bounds after reprojection (in {target_crs}): {protection_data.rio.bounds()}")
        elif protection_data.rio.crs is None:
            print(f"WARNING: Protection data CRS is None. Assuming it's already {target_crs} or attempting to set it.")
            # Attempt to set CRS if it's None and we expect EPSG:3035 (ETRS89LAEA)
            # This is a fallback, ideally the file should have correct CRS info
            if "ETRS89LAEA" in str(file_path) or "3035" in str(file_path): # Heuristic
                 print(f"Attempting to set original CRS to EPSG:3035 then reprojecting to {target_crs}")
                 protection_data = protection_data.rio.write_crs("EPSG:3035", inplace=True)
                 protection_data = protection_data.rio.reproject(target_crs)
                 print(f"Protection data reprojected. New CRS: {protection_data.rio.crs}")
                 print(f"Protection data bounds after reprojection (in {target_crs}): {protection_data.rio.bounds()}")
            else:
                print(f"Could not determine original CRS to reproject from. Proceeding without reprojection, assuming {target_crs}.")
                protection_data = protection_data.rio.write_crs(target_crs, inplace=True)


        if bounds:
            print(f"Attempting to clip protection data (now in {protection_data.rio.crs}) with bounds: {bounds}")
            # The bounds are expected to be in EPSG:4326
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
    """
    Reproject and align all datasets to the same grid and CRS, using the finest resolution dataset as reference
    """
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
    
    # Determine which dataset has the finest resolution (smallest value)
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
    else: # income
        reference_data = income_data
    
    print(f"Using {finest_dataset} data as reference for all reprojections")
    
    # Reproject all datasets to match the reference
    try:
        # Reproject datasets that are not the reference
        reprojected_datasets = []
        
        if finest_dataset != 'population':
            # ... existing population reprojection code ...
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

def main(population_file_path: str, income_file_path: str, flood_file_path: str, region_bounds=None, base_output_dir=None, 
         nuts_level=None, cpu_cores=None, 
         flood_threshold=None, income_threshold=None): # flood_threshold here is for context if needed, vuln flood is >0m
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
    # Ensure setup_logging is defined and handles the log file (e.g., by setting a global 'log_file_handle')
    log_file = setup_logging(output_dir_for_this_run) 
    
    # Print the paths being used
    print(f"Using population data: {population_file_path}")
    print(f"Using flood data: {flood_file_path}")
    print(f"Using income data: {income_file_path}")
    print(f"Using protection data: {protection_file}")
    print(f"Using NUTS regions: {nuts_file}")
    print(f"Output will be saved to: {output_dir_for_this_run}")
    
    # Use the region bounds from arguments or default to Europe
    if region_bounds is None:
        region_bounds = (-10.0, 36.0, 30.0, 72.0)  # Default covers most of mainland Europe
        region_name = "Mainland Europe"
    else:
        region_name = "Custom Region"
    
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

    print_memory_usage("After loading all datasets")
    
    if pop_data is not None and flood_data is not None and income_data is not None and protection_data is not None:
        # Align all datasets
        align_start = print_timestamp("Starting dataset alignment")
        pop_data, flood_data, income_data, protection_data = align_datasets(pop_data, flood_data, income_data, protection_data)
        print_memory_usage("After dataset alignment")
        print_timestamp("Completed dataset alignment", align_start)
        
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
        
        # Convert to dask array for parallel processing
        if hasattr(pop_data, 'chunk'): # Ensure pop_data is an xarray DataArray
             pop_data = pop_data.chunk({'x': 1000, 'y': 1000})
        
        # Visualize data alignment
        vis_start = print_timestamp("Starting visualizations")
        visualize_data_alignment_with_nuts(pop_data, flood_data_protected, income_data, nuts_file, output_dir_for_this_run) 
        print_timestamp("Completed alignment visualization", vis_start)

        # Run analysis
        analysis_start = print_timestamp("Starting population analysis by depth and income")
        
        # Determine the thresholds to use, with defaults if None
        # current_flood_threshold_context is passed for context if needed elsewhere by process_region_with_income,
        # but process_region_with_income will use >0 for the vulnerability flood condition itself.
        current_flood_threshold_context = flood_threshold if flood_threshold is not None else 1.0 
        current_income_threshold_vuln = income_threshold if income_threshold is not None else 18000

        nuts_pop_by_depth_income = analyze_population_by_flood_depth_and_income(
            pop_data, flood_data_protected, income_data=income_data, nuts_file=nuts_file, 
            region_name=region_name,
            nuts_level=nuts_level,
            cpu_cores=cpu_cores,
            flood_threshold_vuln_context=current_flood_threshold_context, # Pass contextual flood threshold
            income_threshold_vuln=current_income_threshold_vuln # Pass vulnerability income threshold
        )
        print_timestamp("Completed population analysis", analysis_start)
        
        print_memory_usage("After main analysis")

        # Save income-stratified results to GeoJSON for GIS applications
        if output_dir_for_this_run and nuts_pop_by_depth_income is not None:
            print("Saving results to GeoJSON...")
            geojson_path = os.path.join(output_dir_for_this_run, f'analysis_results_{ssp_name}_{year}_{flood_event_name}.geojson')
            nuts_pop_by_depth_income.to_file(geojson_path, driver='GeoJSON')
            print(f"GeoJSON results with income analysis saved to {geojson_path}")
        
        if nuts_pop_by_depth_income is not None:
            visualize_population_by_income_and_depth(
                nuts_pop_by_depth_income, output_dir_for_this_run, current_flood_filename 
            )
        print_memory_usage("After visualization")

        # Call the refactored visualization function for vulnerable areas
        if nuts_pop_by_depth_income is not None:
            visualize_vulnerable_areas(
                results_gdf=nuts_pop_by_depth_income,
                flood_threshold=0.0, # Explicitly use 0.0 to signify "any flooding > 0m" for visualization title
                income_threshold=current_income_threshold_vuln,
                output_dir=output_dir_for_this_run, 
                flood_filename=current_flood_filename,
                nuts_file=nuts_file, # Optional, for context in plot
                nuts_level=nuts_level  # Optional, for context
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

def analyze_population_by_flood_depth_and_income(population_data, flooding_data, income_data, nuts_file=None, 
                                               region_name=None, nuts_level=None, cpu_cores=None,
                                               flood_threshold_vuln_context=1.0, income_threshold_vuln=18000): # Added vuln thresholds
    """
    Analyze how many people are affected by different flood depth ranges with optional income stratification
    Uses threading for parallel region processing (safer than multiprocessing)
    Also calculates vulnerability based on income_threshold_vuln and flood > 0m.
    flood_threshold_vuln_context is passed for potential other uses but not for the primary vuln flood condition.
    """
    print(f"\n==== ANALYZING POPULATION BY FLOOD DEPTH AND INCOME ====")
    
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
    
    # Load NUTS regions
    global nuts_gdf
    nuts_gdf = gpd.read_file(nuts_file)
    print(f"Loaded {len(nuts_gdf)} NUTS regions")
    
    # Get NUTS regions at the specified level
    print(f"Selecting all NUTS level {nuts_level} regions in the study area")
    nuts_level_regions = nuts_gdf[nuts_gdf['LEVL_CODE'] == nuts_level].copy()
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

    # Make data globally accessible for worker threads
    global pop_data, flood_data, income_data_global
    pop_data = population_data
    flood_data = flooding_data
    income_data_global = income_data
    
    # Set up threading pool parameters
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_cores = int(slurm_cpus)
        print(f"Using {num_cores} threads allocated by SLURM")
    else:
        num_cores = min(cpu_cores, multiprocessing.cpu_count()) if cpu_cores else multiprocessing.cpu_count()
        print(f"Using {num_cores} threads of {multiprocessing.cpu_count()} available CPU cores")
    
    # Prepare data for parallel processing
    # region_data = [(idx, region) for idx, region in study_regions.iterrows()] # Old
    region_processing_tuples = []
    for idx, region_obj_iter in study_regions.iterrows():
        # Pass the income threshold for vulnerability. The flood threshold for vulnerability is handled as >0 internally
        # in process_region_with_income. flood_threshold_vuln_context is passed along.
        region_processing_tuples.append( (idx, region_obj_iter, flood_threshold_vuln_context, income_threshold_vuln) )
    
    total_regions = len(region_processing_tuples)

    result_data = []

    start_time = time.time()

    from concurrent.futures import ThreadPoolExecutor  # Changed from ProcessPoolExecutor

    # Add before the parallel processing loop
    import gc
    gc.collect()  # Force garbage collection
    print_memory_usage("Before starting parallel processing")

    mp_start = print_timestamp("Starting parallel region processing")

    # Using ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Add robust exception handling
        try:
            print(f"👉 Starting executor with {num_cores} threads for {len(region_processing_tuples)} regions")
            
            # Process results as they complete
            result_data = []
            futures = []
            
            # Submit tasks to the executor
            print("Submitting tasks to executor...")
            # for i, region in enumerate(region_data): # Old
            #     idx, region_obj = region # Old
            #     region_name = region_obj.get('NAME_LATN', region_obj.get('NUTS_NAME', f'Unknown-{idx}')) # Old
            #     future = executor.submit(process_region_with_income, region) # Old
            #     futures.append((future, region_name)) # Old

            for i, task_data_tuple in enumerate(region_processing_tuples):
                idx, region_obj, _, _ = task_data_tuple # unpack to get region_obj for name
                region_name = region_obj.get('NAME_LATN', region_obj.get('NUTS_NAME', f'Unknown-{idx}'))
                future = executor.submit(process_region_with_income, task_data_tuple) # Pass the whole tuple
                futures.append((future, region_name))
                
                # Print only occasionally to avoid flooding logs
                if i % 10 == 0 or i == len(region_processing_tuples) - 1:
                    print(f"  Submitted {i+1}/{len(region_processing_tuples)} regions")
            
            # Process results with better error handling and timeouts
            from concurrent.futures import as_completed, TimeoutError
            completed = 0
            total = len(futures)
            
            print(f"\n👉 Waiting for {total} futures to complete...")
            
            # Track which future we're waiting on
            for i, (future, region_name) in enumerate(futures):
                try:
                    print(f"⏳ Waiting for result from region: {region_name} ({i+1}/{total})")
                    # Set a timeout per future to prevent hanging on any single one
                    result = future.result(timeout=6000)  # 100-minute timeout per region
                    
                    if result:
                        result_data.append(result)
                    
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"✅ Progress: {completed}/{total} regions completed ({completed/total*100:.1f}%)")
                        # Force garbage collection
                        gc.collect()
                        print_memory_usage("After processing batch")
                except TimeoutError:
                    print(f"⚠️ Timeout waiting for region {region_name} - skipping")
                except Exception as e:
                    print(f"⚠️ Error processing region {region_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"🏁 All regions processed. Completing analysis...")
            
        except Exception as e:
            print(f"⚠️ ERROR in processing: {str(e)}")
            traceback.print_exc()
            result_data = []

    # This code now runs AFTER the with block completes, which should happen reliably with ThreadPoolExecutor
    print("\n==== PARALLEL PROCESSING COMPLETED ====")
    print_timestamp("All parallel region processing finished", mp_start)

    # Memory cleanup
    print("\n==== FINAL CLEANUP ====")
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
    print(f"Failed or skipped: {len(missing_regions)} ({len(missing_regions)/len(all_regions)*100:.1f}%)")
    
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

def visualize_population_by_income_and_depth(nuts_pop_by_depth_income, output_dir=None, flood_filename=None, region_filter=None):
    """
    Create focused visualizations of population affected by different flood depths stratified by income
    For either specific regions or all regions in the dataset
    
    Parameters:
    - nuts_pop_by_depth_income: GeoDataFrame with analysis results by region
    - output_dir: Directory to save visualizations
    - flood_filename: Filename of flood data for title extraction
    """
    import gc
    print("\n==== VISUALIZING POPULATION BY INCOME AND FLOOD DEPTH ====")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract flood return period for title if available
    flood_title = "Flood Impact Analysis"
    if flood_filename:
        import re
        rp_match = re.search(r'RP(\d+)', flood_filename, re.IGNORECASE)
        if rp_match:
            rp_value = rp_match.group(1)
            flood_title = f"Flood Impact Analysis (RP{rp_value})"
    
    # Filter regions if specified, otherwise use all data
    if region_filter and 'field' in region_filter and 'values' in region_filter:
        field = region_filter['field']
        values = region_filter['values']
        filtered_regions = nuts_pop_by_depth_income[
            nuts_pop_by_depth_income[field].isin(values)
        ].copy()
        region_name = "_".join(values)  # For filenames
        region_display = ", ".join(values)  # For titles
    else:
        filtered_regions = nuts_pop_by_depth_income.copy()
        region_name = "all_regions"
        region_display = "All Regions"
    
    if len(filtered_regions) == 0:
        print(f"Warning: No data found for the specified filter criteria.")
        print("Using all data for visualization instead.")
        filtered_regions = nuts_pop_by_depth_income.copy()
        region_name = "all_regions"
        region_display = "All Regions"
    else:
        print(f"Found {len(filtered_regions)} regions matching filter criteria.")
    
    # 1. VISUALIZATION: Income-stratified flood impact summary chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # First subplot: Affected population by income level
    region_affected = filtered_regions[['region_name', 'low_income_affected', 'mid_income_affected', 'high_income_affected']].copy()
    total_affected = region_affected[['low_income_affected', 'mid_income_affected', 'high_income_affected']].sum()
    
    # Calculate percentages for annotation
    total_people_affected = total_affected.sum()
    percentages = (total_affected / total_people_affected * 100).round(1)
    
    # Plot the pie chart
    ax1.pie(
        total_affected, 
        labels=['Low Income', 'Middle Income', 'High Income'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff', '#99ff99'],
        explode=(0.05, 0, 0),  # Slightly emphasize low income
        shadow=True,
        startangle=90
    )
    ax1.set_title(f'Affected Population by Income Level\nTotal: {total_people_affected:,.0f} people')
    
    # Second subplot: Overall income distribution
    region_pop = filtered_regions[['region_name', 'low_income_population', 'mid_income_population', 'high_income_population']].copy()
    total_pop = region_pop[['low_income_population', 'mid_income_population', 'high_income_population']].sum()
    
    # Plot the pie chart for overall population
    ax2.pie(
        total_pop, 
        labels=['Low Income', 'Middle Income', 'High Income'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff', '#99ff99'],
        shadow=True,
        startangle=90
    )
    ax2.set_title(f'Overall Income Distribution\nTotal: {total_pop.sum():,.0f} people')
    
    plt.suptitle(f"{flood_title}\n{region_display} Income Distribution Analysis", fontsize=16)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{region_name}_income_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 2. VISUALIZATION: Detailed heatmap for selected regions
    # Identify depth ranges and income levels
    depth_ranges = ['0.0-0.5m', '0.5-1.0m', '1.0-2.0m', '2.0-4.0m', '4.0-6.0m', '>6.0m']
    income_levels = ['low', 'mid', 'high']
    
    # Create consolidated data for all selected regions
    consolidated_data = {depth: {'Low': 0, 'Mid': 0, 'High': 0} for depth in depth_ranges}
    
    # Debug to see available columns
    print("Available columns in filtered_regions:")
    print(filtered_regions.columns.tolist())
    
    # Sum up data across all filtered regions with CORRECTED column naming pattern
    for _, region_row in filtered_regions.iterrows():
        for depth in depth_ranges:
            # The actual column names have format "{range_name}_low", etc.
            # NOT "{depth}_{income}" as the current code tries to use
            low_col = f"{depth}_low"
            mid_col = f"{depth}_mid" 
            high_col = f"{depth}_high"
            
            if low_col in region_row:
                consolidated_data[depth]['Low'] += region_row[low_col]
            if mid_col in region_row:
                consolidated_data[depth]['Mid'] += region_row[mid_col]
            if high_col in region_row:
                consolidated_data[depth]['High'] += region_row[high_col]
                
    # After building consolidated data, add debug info
    print("Consolidated data before creating DataFrame:")
    for depth, values in consolidated_data.items():
        print(f"  {depth}: {values}")
    
    # Create DataFrame from consolidated data
    heatmap_data = []
    for depth, values in consolidated_data.items():
        row = {'Depth': depth}
        row.update(values)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('Depth', inplace=True)
    
    # Add percentage calculations across rows (what percentage of each depth is in each income level)
    percentage_df = heatmap_df.copy()
    for idx in percentage_df.index:
        row_sum = percentage_df.loc[idx].sum()
        if row_sum > 0:
            percentage_df.loc[idx] = (percentage_df.loc[idx] / row_sum * 100).round(1)
    
    # Create a more informative heatmap with both values and percentages
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with annotations showing both value and percentage
    sns.heatmap(
        heatmap_df,
        cmap='YlOrRd',
        annot=True,
        fmt='.0f',
        cbar_kws={'label': 'Population Affected'},
    )
    
    plt.title(f"{flood_title}\\n{region_display} Population by Income Level and Flood Depth")
    plt.tight_layout()
    
    # if output_dir:
    #     plt.savefig(os.path.join(output_dir, f'{region_name}_income_depth_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 3. VISUALIZATION: Percentage of income groups affected by each depth category
    # Calculate what percentage of each income group is affected at each depth
    income_totals = {
        'Low': filtered_regions['low_income_population'].sum(),
        'Mid': filtered_regions['mid_income_population'].sum(),
        'High': filtered_regions['high_income_population'].sum()
    }
    
    percentage_impact_df = heatmap_df.copy()
    for col in percentage_impact_df.columns:
        percentage_impact_df[col] = (percentage_impact_df[col] / income_totals[col] * 100).round(2)
    
    # plt.figure(figsize=(12, 8))
    
    # Use a diverging colormap to emphasize differences in vulnerability
    # sns.heatmap(
    #     percentage_impact_df,
    #     cmap='RdBu_r',  # Red for high impact (more vulnerable)
    #     annot=True,
    #     fmt='.2f',  # Show with 2 decimal places
    #     cbar_kws={'label': '% of Income Group Affected'},
    # )
    
    # plt.title(f"{flood_title}\\nPercentage of Each Income Group Affected by Flood Depth in {region_display}")
    # plt.tight_layout()
    
    # if output_dir:
    #     plt.savefig(os.path.join(output_dir, f'{region_name}_income_vulnerability.png'), dpi=300, bbox_inches='tight')
    
    # 4. Save data as CSV for further analysis
    if output_dir:
        csv_path = os.path.join(output_dir, f'{region_name}_population_by_income_depth.csv')
        filtered_regions.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        # Also save the heatmap data
        # heatmap_csv = os.path.join(output_dir, f'{region_name}_income_depth_matrix.csv')
        # heatmap_df.to_csv(heatmap_csv)
        
        # And the percentage impact data
        # impact_csv = os.path.join(output_dir, f'{region_name}_income_vulnerability_matrix.csv')
        # percentage_impact_df.to_csv(impact_csv)
        
        print(f"{region_display} income analysis results saved to {csv_path}")
    
    # Debug outputs to understand the data structure
    print("\nDebug information for heatmap data:")
    print(f"1. Depth ranges being used: {depth_ranges}")
    print(f"2. Income levels being used: {income_levels}")
    print(f"3. Sample of first region columns:")
    if len(filtered_regions) > 0:
        print(filtered_regions.iloc[0].index.tolist()[:20])  # Show first 20 columns
    print(f"4. Final heatmap data:")
    print(heatmap_df)
    print(f"5. Sum of all values in heatmap: {heatmap_df.values.sum()}")

    # Rest of function remains similar but with updated variable names and titles
    # ...

def visualize_data_alignment_with_nuts(pop_data, flood_data, income_data, nuts_file, output_dir=None):
    """
    Create visualizations of datasets with NUTS regions overlaid to verify alignment
    Focused only on Vienna region to save memory
    
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent
    - income_data: xarray DataArray with income data
    - nuts_file: path to NUTS regions GeoJSON file
    - output_dir: directory to save visualizations
    """
    print("\n==== VISUALIZING DATA ALIGNMENT FOR VIENNA REGION ONLY ====")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    print_memory_usage("Before Vienna alignment check")
    
    # Load NUTS regions
    nuts_gdf = None
    if nuts_file and os.path.exists(nuts_file):
        try:
            nuts_gdf = gpd.read_file(nuts_file)
            print(f"Loaded {len(nuts_gdf)} NUTS regions for Vienna alignment check")
        except Exception as e:
            print(f"Error loading NUTS regions: {str(e)}")
            return
    else:
        print("NUTS file not found, skipping alignment check")
        return
    
    print_memory_usage("After loading NUTS regions")

    
    # Create Vienna-specific visualization only
    print("Creating Vienna-specific alignment check...")
    
    # Find Vienna in the NUTS regions
    vienna_regions = nuts_gdf[nuts_gdf['NAME_LATN'].str.contains('Wien', case=False, na=False) | 
                             nuts_gdf['NUTS_NAME'].str.contains('Vienna', case=False, na=False) |
                             nuts_gdf['NUTS_ID'].str.startswith('AT13', na=False)]
    
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
        nuts_gdf.boundary.plot(ax=axs_vienna[0], color='gray', linewidth=0.3)
        
        # Plot 2: Flood data in Vienna
        flood_vienna = flood_data.rio.clip_box(*vienna_bounds)
        flood_vienna.plot(ax=axs_vienna[1], cmap='Blues', add_colorbar=True)
        axs_vienna[1].set_title('Vienna: Flood Data')
        vienna_regions.boundary.plot(ax=axs_vienna[1], color='red', linewidth=1.0)
        nuts_gdf.boundary.plot(ax=axs_vienna[1], color='gray', linewidth=0.3)
        
        # Plot 3: Income data in Vienna
        income_vienna = income_data.rio.clip_box(*vienna_bounds)
        income_vienna.plot(ax=axs_vienna[2], cmap='plasma', add_colorbar=True)
        axs_vienna[2].set_title('Vienna: Income Data')
        vienna_regions.boundary.plot(ax=axs_vienna[2], color='red', linewidth=1.0)
        nuts_gdf.boundary.plot(ax=axs_vienna[2], color='gray', linewidth=0.3)
        
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
    gc.collect()
    print_memory_usage("After Vienna visualization")

def visualize_vulnerable_areas(results_gdf, flood_threshold=0.0, income_threshold=15000, output_dir=None, flood_filename=None, nuts_file=None, nuts_level=None):
    """
    Visualize areas where flooding exceeds a threshold depth AND income is below a threshold,
    using pre-calculated data from results_gdf.
    
    Parameters:
    - results_gdf: GeoDataFrame with analysis results including vulnerability metrics
                   (e.g., 'vulnerable_pop_thresh', 'total_population', 'vulnerable_pct_thresh').
    - flood_threshold: Threshold used for vulnerability calculation (for titles/labels). 
                       A value of 0.0 indicates "any flooding (depth > 0m)".
    - income_threshold: Threshold used for vulnerability calculation (for titles/labels).
    - output_dir: directory to save visualizations
    - flood_filename: filename of the flood dataset to extract return period (for titles)
    - nuts_file: path to NUTS regions GeoJSON file (optional, for map context)
    - nuts_level: NUTS level used (optional, for context)
    """
    print("\n==== VISUALIZING VULNERABLE AREAS (FROM PRE-CALCULATED DATA) ====")
    print(f"Visualizing based on thresholds: Flood > {flood_threshold}m & Income < {income_threshold} PPP")

    if results_gdf is None or results_gdf.empty:
        print("No data provided for vulnerability visualization. Skipping.")
        return 0

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
        total_vulnerable_pop_all_regions = results_gdf['vulnerable_pop_thresh'].sum()
        print(f"Total vulnerable population across all regions (from GDF): {total_vulnerable_pop_all_regions:,.0f}")
        return total_vulnerable_pop_all_regions


    # Create a visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot the map
    vuln_gdf_plot.plot(
        column='vulnerable_pct_thresh', # This column is now calculated in process_region_with_income
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
    
    flood_condition_text = f"Flood Depth > {flood_threshold:.1f}m"
    if flood_threshold == 0.0:
        flood_condition_text = "Any Flooding (Depth > 0m)"

    ax.set_title(f"Vulnerable Population by Region\n({flood_condition_text} & Income < {income_threshold:,.0f} PPP)", 
                fontsize=14, pad=20)
    
    # Add explanatory text box
    ax.text(0.02, 0.02, 
            f"Percentage of each region's population vulnerable to:\n"
            f"• {flood_condition_text}\n"
            f"• Income < {income_threshold:,.0f} PPP",
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
        plot_filename = f'vulnerability_by_region_{flood_filename}.png' if flood_filename else 'vulnerability_by_region.png'
        plot_path = os.path.join(output_dir, plot_filename)
        
        csv_filename = f'vulnerability_summary_by_region_{flood_filename}.csv' if flood_filename else 'vulnerability_summary_by_region.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Vulnerability map saved to {plot_path}")
        
        # Save a summary CSV for vulnerability (only plotted data)
        vuln_gdf_plot[['region_name', 'nuts_id', 'total_population', 'vulnerable_pop_thresh', 'vulnerable_pct_thresh']].to_csv(
            csv_path, index=False)
        print(f"Vulnerability summary CSV saved to {csv_path}")
    
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

# def process_region_with_income(region_data): # Old signature
def process_region_with_income(task_data_tuple): # New signature
    """
    Process a single region with income analysis and vulnerability assessment in its own process.
    Returns the result data for that region with income stratification and vulnerability metrics.
    Vulnerability is defined as: income < income_threshold_vuln_arg AND flood_depth > 0.
    The flood_threshold_context_arg is available but ignored for the vulnerability flood condition.
    """
    
    # Declare global variables to be used in this worker function
    global pop_data, flood_data, income_data_global, nuts_gdf, depth_ranges # depth_ranges is for income stratification
    
    # Unpack task data
    idx, region, flood_threshold_context_arg, income_threshold_vuln_arg = task_data_tuple
    
    region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
    worker_pid = os.getpid()
    print(f"WORKER START [PID {worker_pid}]: Processing region {region_name}")
    print_memory_usage(f"Start processing region {region_name}")
    region_start = print_timestamp(f"Starting processing for region: {region_name}")
    
    # Buffer output instead of immediate printing
    output_buffer = [f"\nAnalyzing region: {region_name}"]
    
    # Get region geometry
    region_geom = region.geometry
    
    # Initialize result to return in case of error
    result_row = {
        'region_name': region_name,
        'nuts_id': region.get('NUTS_ID', 'Unknown'),
        'nuts_level': region.get('LEVL_CODE', 'Unknown')
    }
    
    try:
        # Convert region geometry to GeoDataFrame for clipping
        region_gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=nuts_gdf.crs)
        
        # Clip the data to the region
        pop_region = pop_data.rio.clip(region_gdf.geometry, region_gdf.crs)
        flood_region = flood_data.rio.clip(region_gdf.geometry, region_gdf.crs)
        income_region = income_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
        
        # Calculate total population in region
        total_pop = float(pop_region.sum().values)
        output_buffer.append(f"  Total population in region: {total_pop:,.2f}")
        
        # Calculate income thresholds for the region
        # We'll use the 33rd and 66th percentiles to divide into thirds
        income_values = income_region.values[~np.isnan(income_region.values)]
        if len(income_values) > 0:
            income_low_threshold = np.percentile(income_values, 33.33)
            income_high_threshold = np.percentile(income_values, 66.67)
            output_buffer.append(f"  Income thresholds:")
            output_buffer.append(f"    Low: <{income_low_threshold:.2f}")
            output_buffer.append(f"    Medium: {income_low_threshold:.2f}-{income_high_threshold:.2f}")
            output_buffer.append(f"    High: >{income_high_threshold:.2f}")
            
            # Create income masks
            low_income_mask = income_region <= income_low_threshold
            mid_income_mask = (income_region > income_low_threshold) & (income_region <= income_high_threshold)
            high_income_mask = income_region > income_high_threshold

        else:
            output_buffer.append("  Warning: No valid income data for this region")
            print("\n".join(output_buffer))
            print_memory_usage(f"End processing region {region_name}")
            print_timestamp(f"Completed processing for region: {region_name}", region_start)
            return result_row
        
        # Calculate population by income level
        pop_low_income = float(pop_region.where(low_income_mask, 0).sum().values)
        pop_mid_income = float(pop_region.where(mid_income_mask, 0).sum().values)
        pop_high_income = float(pop_region.where(high_income_mask, 0).sum().values)
        
        output_buffer.append(f"  Population by income level:")
        output_buffer.append(f"    Low income: {pop_low_income:,.2f} ({pop_low_income/total_pop*100:.2f}%)")
        output_buffer.append(f"    Medium income: {pop_mid_income:,.2f} ({pop_mid_income/total_pop*100:.2f}%)")
        output_buffer.append(f"    High income: {pop_high_income:,.2f} ({pop_high_income/total_pop*100:.2f}%)")
        
        # Initialize depth dictionaries for each income level
        depth_all_income = {}
        depth_low_income = {}
        depth_mid_income = {}
        depth_high_income = {}
        
        # Calculate population in each flood depth range for each income level
        for min_depth, max_depth in depth_ranges:
            # Create mask for this depth range
            if max_depth == float('inf'):
                depth_mask = flood_region > min_depth
                range_name = f">{min_depth}m"
            else:
:
                depth_mask = (flood_region > min_depth) & (flood_region <= max_depth)
                range_name = f"{min_depth}-{max_depth}m"
            
            # Calculate affected population (total and by income level)
            affected_pop = float(pop_region.where(depth_mask, 0).sum().values)
            depth_all_income[range_name] = affected_pop
            
            # Calculate for each income level - use efficient mask combination
            affected_low = float(pop_region.where(depth_mask & low_income_mask, 0).sum().values)
            affected_mid = float(pop_region.where(depth_mask & mid_income_mask, 0).sum().values)
            affected_high = float(pop_region.where(depth_mask & high_income_mask, 0).sum().values)
            
            depth_low_income[f"{range_name}_low"] = affected_low
            depth_mid_income[f"{range_name}_mid"] = affected_mid
            depth_high_income[f"{range_name}_high"] = affected_high
            
            # Fix division by zero errors with safe division
            low_pct = (affected_low/affected_pop*100) if affected_pop > 0 else 0.0
            mid_pct = (affected_mid/affected_pop*100) if affected_pop > 0 else 0.0
            high_pct = (affected_high/affected_pop*100) if affected_pop > 0 else 0.0
        
            output_buffer.append(f"  Population affected by {range_name} flooding:")
            output_buffer.append(f"    Total: {affected_pop:,.2f}")
            output_buffer.append(f"    Low income: {affected_low:,.2f} ({low_pct:.2f}% of affected)")
            output_buffer.append(f"    Medium income: {affected_mid:,.2f} ({mid_pct:.2f}% of affected)")
            output_buffer.append(f"    High income: {affected_high:,.2f} ({high_pct:.2f}% of affected)")
        
        # Calculate total affected population
        total_affected = float(pop_region.where(flood_region > 0, 0).sum().values)
        percentage_affected = (total_affected / total_pop * 100) if total_pop > 0 else 0
        output_buffer.append(f"  Total affected population: {total_affected:,.2f} ({percentage_affected:.2f}%)")
        
        # Calculate total affected by income level
        total_affected_low = float(pop_region.where((flood_region > 0) & low_income_mask, 0).sum().values)
        total_affected_mid = float(pop_region.where((flood_region > 0) & mid_income_mask, 0).sum().values)
        total_affected_high = float(pop_region.where((flood_region > 0) & high_income_mask, 0).sum().values)
        
        # Store results
        result_row = {
            'region_name': region_name,
            'nuts_id': region.get('NUTS_ID', 'Unknown'),
            'nuts_level': region.get('LEVL_CODE', 'Unknown'),
            'total_population': total_pop, # This is total_pop for the region
            'total_affected': total_affected,
            'percentage_affected': percentage_affected,
            'low_income_population': pop_low_income,
            'mid_income_population': pop_mid_income,
            'high_income_population': pop_high_income,
            'low_income_affected': total_affected_low,
            'mid_income_affected': total_affected_mid,
            'high_income_affected': total_affected_high,
            **depth_all_income,  # Original depth categories
            **depth_low_income,  # Low income by depth
            **depth_mid_income,  # Middle income by depth
            **depth_high_income  # High income by depth
        }

        # --- Add Vulnerability Analysis logic ---
        # Vulnerability: Income < income_threshold_vuln_arg AND Flood Depth > 0m
        output_buffer.append(f"  Calculating vulnerability (flood > 0m & income < {income_threshold_vuln_arg} PPP)")
        
        # Create masks for vulnerability
        vuln_flood_mask = flood_region > 0  # Use any flooding as the criterion
        vuln_income_mask = income_region < income_threshold_vuln_arg # income_region is already clipped
        vuln_combined_mask = vuln_flood_mask & vuln_income_mask
        
        # Calculate vulnerable population based on these specific thresholds
        # total_pop is already calculated for the region
        vulnerable_pop_specific = float(pop_region.where(vuln_combined_mask, 0).sum().values)
        vulnerable_pct_specific = (vulnerable_pop_specific / total_pop * 100) if total_pop > 0 else 0
        
        output_buffer.append(f"    Vulnerable population (flood > 0m, income < {income_threshold_vuln_arg:,.0f} PPP): {vulnerable_pop_specific:,.2f}")
        output_buffer.append(f"    Vulnerable percentage: {vulnerable_pct_specific:.2f}%")

        # Add vulnerability metrics to the result_row
        result_row['vulnerable_pop_thresh'] = vulnerable_pop_specific
        result_row['vulnerable_pct_thresh'] = vulnerable_pct_specific
        # 'total_population' is already in result_row and is the correct denominator for vulnerable_pct_thresh
        
    except Exception as e:
        output_buffer.append(f"  Error processing region: {str(e)}")
        # Import traceback only when needed
        import traceback
        output_buffer.append(traceback.format_exc())
    
    # Print all buffered output at once (should be more atomic)
    print("\n".join(output_buffer))
    
    # At the end of the function - note completion with PID
    print(f"WORKER COMPLETE [PID {worker_pid}]: Finished region {region_name}")
    print_memory_usage(f"End processing region {region_name}")
    print_timestamp(f"Completed processing for region: {region_name}", region_start)
    
    # Force garbage collection before returning
    import gc
    gc.collect()
    
    return result_row

# Function to be removed: process_vulnerable_region
# The functionality of process_vulnerable_region has been integrated into process_region_with_income.
# This function definition below will be removed.
# def process_vulnerable_region(region_data):
#     """Process a single region to identify vulnerable areas"""
#     global nuts_gdf, pop_data_global, flood_data_global, income_data_global
#     
#     idx, region, flood_threshold, income_threshold = region_data
#     region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
#     nuts_id = region.get('NUTS_ID', 'Unknown')
#     
#     result = {
#         'region_name': region_name,
#         'nuts_id': nuts_id,
#         'vulnerable_population': 0,
#         'total_population': 0,
#         'vulnerable_percentage': 0
#     }
#     
#     try:
#         # Clip data to region
#         region_gdf = gpd.GeoDataFrame(geometry=[region.geometry], crs=nuts_gdf.crs)
#         pop_region = pop_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
#         flood_region = flood_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
#         income_region = income_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
#         
#         # Calculate total population in region
#         total_pop = float(pop_region.sum().values)
#         print(f"  Total population in region: {total_pop:,.2f}")
#         
#         # Calculate income thresholds for the region - WITH ENHANCED DEBUGGING
#         # We'll use the 33rd and 66th percentiles to divide into thirds
#         income_values = income_region.values[~np.isnan(income_region.values)]
#         print(f"  Income data shape: {income_region.shape}")
#         print(f"  Valid income values count: {len(income_values)}")
#         
#         if len(income_values) > 0:
#             # Print detailed income statistics
#             print(f"  Income statistics:")
#             print(f"    Min: {income_values.min():.2f}")
#             print(f"    Max: {income_values.max():.2f}")
#             print(f"    Mean: {income_values.mean():.2f}")
#             print(f"    Median: {np.median(income_values):.2f}")
#             print(f"    5th percentile: {np.percentile(income_values, 5):.2f}")
#             print(f"    95th percentile: {np.percentile(income_values, 95):.2f}")
#             
#             # Calculate thresholds with clear debug info
#             income_low_threshold = np.percentile(income_values, 33.33)
#             income_high_threshold = np.percentile(income_values, 66.67)
#             print(f"  Income thresholds:")
#             print(f"    Low: <{income_low_threshold:.2f}")
#             print(f"    Medium: {income_low_threshold:.2f}-{income_high_threshold:.2f}")
#             print(f"    High: >{income_high_threshold:.2f}")
#             
#             # Verify distribution after applying thresholds
#             low_count = np.sum(income_values <= income_low_threshold)
#             mid_count = np.sum((income_values > income_low_threshold) & (income_values <= income_high_threshold))
#             high_count = np.sum(income_values > income_high_threshold)
#             
#             print(f"  Income distribution verification:")
#             print(f"    Low income cells: {low_count} ({low_count/len(income_values)*100:.2f}%)")
#             print(f"    Mid income cells: {mid_count} ({mid_count/len(income_values)*100:.2f}%)")
#             print(f"    High income cells: {high_count} ({high_count/len(income_values)*100:.2f}%)")

#         # Create masks
#         flood_mask = flood_region > flood_threshold
#         income_mask = income_region < income_threshold
#         combined_mask = flood_mask & income_mask
#         
#         # Calculate population
#         total_pop = float(pop_region.sum().values)
#         vulnerable_pop = float(pop_region.where(combined_mask, 0).sum().values)
#         vulnerable_pct = (vulnerable_pop / total_pop * 100) if total_pop > 0 else 0
#         
#         # Update result
#         result.update({
#             'vulnerable_population': vulnerable_pop,
#             'total_population': total_pop,
#             'vulnerable_percentage': vulnerable_pct
#         })
#         
#     except Exception as e:
#         print(f"Error processing region {region_name} for vulnerability: {str(e)}")
#     
#     return result

def chunk_data(data, chunk_size):
    """Split data into chunks of specified size"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def apply_protection_filter(hazard_data, protection_data, hazard_filename):
    """
    Apply the binary \"hold vs breach\" filter to the hazard data based on protection levels.

    Args:
        hazard_data (xr.DataArray): The original flood hazard GeoTIFF (water depths or binary mask).
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
        hazard_rp = 0
    else:
        hazard_rp = int(rp_match.group(1))
        print(f"Parsed Hazard Return Period (RP): {hazard_rp} years")

    # Ensure protection_data is in the same shape as hazard_data (it should be after alignment)
    if hazard_data.shape != protection_data.shape:
        print(f"ERROR: Hazard data shape {hazard_data.shape} and Protection data shape {protection_data.shape} do not match!")
        # Attempt to reproject protection_data to match hazard_data as a fallback
        # This should ideally not happen if align_datasets worked correctly.
        print("Attempting emergency reprojection of protection_data to match hazard_data...")
        try:
            protection_data = protection_data.rio.reproject_match(hazard_data, resampling=rasterio.enums.Resampling.nearest)
            print(f"Emergency reprojection successful. New protection_data shape: {protection_data.shape}")
        except Exception as e:
            print(f"Emergency reprojection failed: {e}. Returning original hazard data.")
            return hazard_data

    # Create a copy of the hazard data to modify
    protected_hazard_data = hazard_data.copy()

    # 2. Apply the binary "hold vs breach" filter
    # Where protection_data (P) >= hazard_rp (RP), defences hold -> set hazard to 0 (no flood)
    # Where protection_data (P) < hazard_rp (RP), defences breach -> keep original hazard value
    
    # Ensure protection_data has no NaNs where hazard_data is valid, fill with 0 (no protection)
    protection_values = protection_data.fillna(0).values 
    hazard_values = protected_hazard_data.values

    # Create the mask
    # Defences hold if protection_level >= hazard_return_period
    defences_hold_mask = protection_values >= hazard_rp
    
    # Apply the filter: set hazard to 0 where defences hold
    # Ensure we are operating on the correct band if multi-band (typically hazard is single band)
    if hazard_values.ndim == 3 and hazard_values.shape[0] == 1: # Single band raster
        hazard_values[0][defences_hold_mask[0]] = 0 
    elif hazard_values.ndim == 2: # Already 2D array
         hazard_values[defences_hold_mask[0]] = 0 # Assuming protection_mask is also 2D or compatible
    else:
        print(f"WARNING: Hazard data has unexpected dimensions {hazard_values.ndim}. Filter may not apply correctly.")
        # Attempt to apply to the first band if it's a multi-band array that wasn't caught
        if hazard_values.ndim > 2 and defences_hold_mask.ndim > 2:
             hazard_values[0][defences_hold_mask[0]] = 0
        elif hazard_values.ndim > 2 and defences_hold_mask.ndim == 2: # protection_mask might be 2D
             hazard_values[0][defences_hold_mask] = 0


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
    # For multiprocessing on Windows/Linux
    multiprocessing.freeze_support()
    
    # Start with a print statement showing execution environment
    print(f"Starting flood analysis on {platform.node()} with Python {sys.version}")
    print(f"Environment variables: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'Not set')}")
    print(f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Define dataset locations and parameters ---
    population_base_dir = '/hdrive/all_users/wiederkehr/analysis/population_files'
    income_base_dir = '/hdrive/all_users/wiederkehr/analysis/income_files'
    flood_file_dir = '/hdrive/all_users/wiederkehr/analysis/flood_files'

    # Define the SSPs, years, and flood events to iterate over
    ssps = ["SSP1", "SSP2"] # Add all your SSPs
    years = ["2020", "2030", "2040", "2050", "2100"] # Add all relevant years
    flood_rps = [30, 100, 500] # Return periods for flood events

    base_output_directory = os.path.join(script_dir, "analysis_runs_output_all_combinations")
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
                        nuts_level=3, # Or your desired default
                        cpu_cores=32, # Or your desired default
                        flood_threshold=0,
                        income_threshold=18000 # Or your desired default
                        # region_bounds can be set here if needed, e.g. region_bounds=(-10.0, 36.0, 30.0, 72.0)
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