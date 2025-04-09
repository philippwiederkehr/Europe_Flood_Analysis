import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
import geopandas as gpd
import matplotlib.patheffects as pe
from shapely.geometry import box
import seaborn as sns
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from itertools import islice
import signal
import sys
import re
import argparse
import traceback
import platform
import time
import dask.array as da

# Add near the top of your script
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

def parse_arguments():
    """Parse command-line arguments for the script"""
    parser = argparse.ArgumentParser(description='Flood population and income vulnerability analysis')
    parser.add_argument('--bounds', nargs=4, type=float, 
                        help='Analysis bounds: min_lon min_lat max_lon max_lat')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--nuts-level', type=int, default=2, 
                        help='NUTS level to analyze (0-3)')
    parser.add_argument('--no-vis', action='store_true', 
                        help='Skip visualization (data analysis only)')
    parser.add_argument('--cores', type=int, default=32,
                        help='Number of CPU cores to use (default: 32)')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Region processing chunk size (default: auto)')
    parser.add_argument('--flood-threshold', type=float, default=1.0,
                        help='Flood depth threshold for vulnerability analysis (default: 1.0m)')
    parser.add_argument('--income-threshold', type=int, default=18000,
                        help='Income threshold for vulnerability analysis (default: 18000 Euro)')
    return parser.parse_args()

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

def align_datasets(pop_data, flood_data, income_data):
    """
    Reproject and align all datasets to the same grid and CRS, using the finest resolution dataset as reference
    """
    print(f"\n==== ALIGNING DATASETS ====")
    print(f"Population data: {pop_data.shape} with CRS {pop_data.rio.crs}")
    print(f"Flood data: {flood_data.shape} with CRS {flood_data.rio.crs}")
    print(f"Income data: {income_data.shape} with CRS {income_data.rio.crs}")
    
    # Get resolutions of each dataset
    pop_res = pop_data.rio.resolution()
    flood_res = flood_data.rio.resolution()
    income_res = income_data.rio.resolution()
    
    print(f"Population data resolution: {pop_res}")
    print(f"Flood data resolution: {flood_res}")
    print(f"Income data resolution: {income_res}")
    
    # Calculate resolution as the average of x and y resolution
    pop_avg_res = (abs(pop_res[0]) + abs(pop_res[1])) / 2
    flood_avg_res = (abs(flood_res[0]) + abs(flood_res[1])) / 2
    income_avg_res = (abs(income_res[0]) + abs(income_res[1])) / 2
    
    # Determine which dataset has the finest resolution (smallest value)
    resolutions = {
        'population': pop_avg_res,
        'flood': flood_avg_res,
        'income': income_avg_res
    }
    
    finest_dataset = min(resolutions, key=resolutions.get)
    print(f"Dataset with finest resolution: {finest_dataset} ({resolutions[finest_dataset]} degrees/pixel)")
    
    # Set the reference dataset for reprojection
    if finest_dataset == 'population':
        reference_data = pop_data
    elif finest_dataset == 'flood':
        reference_data = flood_data
    else:
        reference_data = income_data
    
    print(f"Using {finest_dataset} data as reference for all reprojections")
    
    # Reproject all datasets to match the reference
    try:
        # Reproject datasets that are not the reference
        reprojected_datasets = []
        
        if finest_dataset != 'population':
            print(f"Reprojecting population data to match {finest_dataset} resolution...")
            
            # Calculate the scale factor based on resolution change
            pop_res = pop_data.rio.resolution()
            ref_res = reference_data.rio.resolution()
            area_ratio = (abs(ref_res[0]) * abs(ref_res[1])) / (abs(pop_res[0]) * abs(pop_res[1]))
            
            # Get population sum before reprojection
            pop_sum_before = float(pop_data.sum().values)

            # Reproject with bilinear interpolation
            pop_data = pop_data.rio.reproject_match(reference_data)
            
            # Apply the area ratio correction to maintain density
            pop_data = pop_data * area_ratio
            
            print(f"Population data shape after reprojection: {pop_data.shape}")
            print(f"Population sum before: {pop_sum_before}")
            print(f"Population sum after: {float(pop_data.sum().values)}")
            reprojected_datasets.append('population')
        
        if finest_dataset != 'flood':
            print(f"Reprojecting flood data to match {finest_dataset} resolution...")
            flood_data_before = flood_data.copy()
            flood_data = flood_data.rio.reproject_match(reference_data)
            print(f"Flood data shape after reprojection: {flood_data.shape}")
            print(f"Cells affected by flooding before: {(flood_data_before > 0).sum().values.item()}")
            print(f"Cells affected by flooding after: {(flood_data > 0).sum().values.item()}")
            reprojected_datasets.append('flood')
        
        if finest_dataset != 'income':
            print(f"Reprojecting income data to match {finest_dataset} resolution...")
            income_data_before = income_data.copy()
            income_data = income_data.rio.reproject_match(reference_data)
            print(f"Income data shape after reprojection: {income_data.shape}")
            reprojected_datasets.append('income')
        
        if reprojected_datasets:
            print(f"Successfully reprojected datasets: {', '.join(reprojected_datasets)}")
        else:
            print(f"No reprojection needed, all datasets already match the finest resolution")
        
        return pop_data, flood_data, income_data
            
    except Exception as e:
        print(f"Error aligning datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return pop_data, flood_data, income_data

def calculate_affected_population(pop_data, flood_data, flood_threshold=0.0):
    """
    Calculate population affected by flooding
    
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent (values above threshold are considered flooded)
    - flood_threshold: threshold above which an area is considered flooded
    
    Returns:
    - total_affected: total number of people affected
    - affected_pop_grid: grid showing affected population
    """
    print(f"\n==== CALCULATING AFFECTED POPULATION ====")
    print(f"Flood threshold: {flood_threshold}")
    print(f"Population data shape: {pop_data.shape}")
    print(f"Flood data shape: {flood_data.shape}")
    
    # Check for alignment issues
    if pop_data.shape != flood_data.shape:
        print(f"WARNING: Dataset shapes don't match!")
        print(f"  Population coords: x={len(pop_data.x)}, y={len(pop_data.y)}")
        print(f"  Flood coords: x={len(flood_data.x)}, y={len(flood_data.y)}")
    
    # Create a mask of flooded areas
    flood_mask = flood_data > flood_threshold
    flooded_cells = flood_mask.sum().values.item()
    total_cells = flood_mask.size
    print(f"Flooded cells: {flooded_cells} of {total_cells} ({flooded_cells/total_cells*100:.2f}%)")
    
    # Print flood depth distribution
    flood_data_valid = flood_data.values[~np.isnan(flood_data.values)]
    if len(flood_data_valid) > 0:
        flood_depth_bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
        print("Flood depth distribution:")
        for i in range(len(flood_depth_bins)-1):
            count = np.sum((flood_data_valid > flood_depth_bins[i]) & (flood_data_valid <= flood_depth_bins[i+1]))
            if flood_depth_bins[i+1] == float('inf'):
                print(f"  > {flood_depth_bins[i]}m: {count} cells")
            else:
                print(f"  {flood_depth_bins[i]}-{flood_depth_bins[i+1]}m: {count} cells")
    
    # Apply mask to population data
    affected_pop_grid = pop_data.where(flood_mask, 0)
    
    # Calculate total affected population
    total_affected = affected_pop_grid.sum().item()
    
    # Calculate total population and percentage
    total_population = pop_data.sum().item()
    percentage_affected = (total_affected / total_population) * 100 if total_population > 0 else 0
    
    # Detailed population statistics
    print(f"\nPopulation statistics:")
    print(f"  Max population in a cell: {float(pop_data.max().values)}")
    print(f"  Mean population in a cell: {float(pop_data.mean().values)}")
    print(f"  Median population in a cell: {float(np.nanmedian(pop_data.values))}")
    print(f"  Cells with population > 0: {(pop_data > 0).sum().values.item()}")
    print(f"  Cells with population > 100: {(pop_data > 100).sum().values.item()}")
    print(f"  Cells with population > 1000: {(pop_data > 1000).sum().values.item()}")
    
    print(f"\nAffected population statistics:")
    print(f"  Max affected population in a cell: {float(affected_pop_grid.max().values)}")
    print(f"  Mean affected population in a cell: {float(affected_pop_grid.mean().values)}")
    print(f"  Cells with affected population > 0: {(affected_pop_grid > 0).sum().values.item()}")
    print(f"  Cells with affected population > 100: {(affected_pop_grid > 100).sum().values.item()}")
    print(f"  Cells with affected population > 1000: {(affected_pop_grid > 1000).sum().values.item()}")
    
    print(f"\nTotal population: {total_population:.2f}")
    print(f"Affected population: {total_affected:.2f} ({percentage_affected:.2f}%)")
    
    # If total population seems low, provide a warning
    if total_population < 5000000 and 'austria' in str(pop_data.attrs).lower():
        print("\nWARNING: Total population appears low for Austria!")
        print("  - The SSP datasets might need a scaling factor (e.g., multiply by a constant)")
        print("  - Check the units in the dataset documentation")
        print("  - Current value is only about 20% of expected population")
    
    return total_affected, affected_pop_grid, percentage_affected

def visualize_results(pop_data, flood_data, affected_pop_grid, output_dir=None, nuts_file=None, flood_filename=None):
    """
    Create visualizations of the results with optional NUTS regions overlay
    
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent
    - affected_pop_grid: xarray DataArray with affected population
    - output_dir: directory to save visualizations
    - nuts_file: path to NUTS regions GeoJSON file
    - flood_filename: filename of the flood dataset to extract return period
    """
    print("Generating visualizations...")
    
    # Extract return period from flood filename if provided
    flood_title = "Flood Extent"
    if flood_filename:
        # Try to extract RP value (return period) from the filename
        import re
        rp_match = re.search(r'RP(\d+)', flood_filename, re.IGNORECASE)
        if rp_match:
            rp_value = rp_match.group(1)
            flood_title = f"Flood Extent (RP{rp_value})"
            print(f"Detected return period: RP{rp_value}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load NUTS regions if provided
    nuts_gdf = None
    if nuts_file and os.path.exists(nuts_file):
        print(f"Loading NUTS regions from: {nuts_file}")
        try:
            nuts_gdf = gpd.read_file(nuts_file)
            print(f"Loaded {len(nuts_gdf)} NUTS regions")
        except Exception as e:
            print(f"Error loading NUTS regions: {str(e)}")
    
    # Downsample large datasets before plotting to prevent memory errors
    print(f"Original data shapes - Population: {pop_data.shape}, Flood: {flood_data.shape}")
    
    # If population data is too large, downsample it
    if pop_data.shape[1] * pop_data.shape[2] > 1000000:  # If more than ~1 million pixels
        print("Population data is very large. Downsampling for visualization...")
        # Coarsen by factor of 20 in both dimensions
        sample_factor = 20
        pop_data_vis = pop_data.coarsen(y=sample_factor, x=sample_factor).mean()
        affected_pop_grid_vis = affected_pop_grid.coarsen(y=sample_factor, x=sample_factor).mean()
        print(f"Downsampled population data shape: {pop_data_vis.shape}")
    else:
        pop_data_vis = pop_data
        affected_pop_grid_vis = affected_pop_grid
    
    # If flood data is too large, downsample it too
    if flood_data.shape[1] * flood_data.shape[2] > 1000000:
        print("Flood data is very large. Downsampling for visualization...")
        sample_factor = 10
        flood_data_vis = flood_data.coarsen(y=sample_factor, x=sample_factor).mean()
        print(f"Downsampled flood data shape: {flood_data_vis.shape}")
    else:
        flood_data_vis = flood_data
    
    # Print data ranges for debugging
    print(f"Population data range: {float(pop_data_vis.min().values)} to {float(pop_data_vis.max().values)}")
    print(f"Flood data range: {float(flood_data_vis.min().values)} to {float(flood_data_vis.max().values)}")
    print(f"Affected population range: {float(affected_pop_grid_vis.min().values)} to {float(affected_pop_grid_vis.max().values)}")
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Population density
    pop_plot = pop_data_vis.plot(ax=axs[0], cmap='viridis', 
                            vmin=0, 
                            add_colorbar=True)
    axs[0].set_title('Population Density')
    
    # Plot 2: Flood extent - use a specific range for flood values
    # Only show positive values since negative are likely masked nodata
    flood_min = max(0.0, float(flood_data_vis.min().values))
    flood_max = float(flood_data_vis.max().values)
    if not np.isnan(flood_min) and not np.isnan(flood_max) and flood_max > flood_min:
        flood_plot = flood_data_vis.plot(ax=axs[1], cmap='Blues', 
                                   vmin=flood_min, vmax=flood_max,
                                   add_colorbar=True)
    else:
        # Fallback if we have invalid min/max
        flood_plot = flood_data_vis.plot(ax=axs[1], cmap='Blues', add_colorbar=True)
    axs[1].set_title(flood_title)
    
    # Plot 3: Affected population
    affected_plot = affected_pop_grid_vis.plot(ax=axs[2], cmap='hot_r', 
                                          vmin=0, 
                                          add_colorbar=True)
    axs[2].set_title('Population Affected by Flooding')
    
    # Overlay NUTS regions on all plots if available
    if nuts_gdf is not None:
        print("Overlaying NUTS regions on plots...")
        # For each plot, overlay NUTS boundaries
        for i, ax in enumerate(axs):
            nuts_gdf.boundary.plot(ax=ax, color='red', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'flood_population_analysis.png'), dpi=300)
        print(f"Visualization saved to {os.path.join(output_dir, 'flood_population_analysis.png')}")

def main(scale_factor=None, region_bounds=None, output_dir=None, skip_vis=False, 
         nuts_level=None, cpu_cores=None, chunk_size=None, 
         flood_threshold=None, income_threshold=None):
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
            print(f"\n[SLURM TERMINATION] Received {signal_name} in job {os.environ.get('SLURM_JOB_ID', 'unknown')}")
            print("[SLURM TERMINATION] Flushing and closing log file")
            sys.stdout.flush()
            
            # Safety: directly flush the global log file handle if available
            if 'log_file_handle' in globals() and log_file_handle and not log_file_handle.closed:
                log_file_handle.flush()
                log_file_handle.write("\n[SLURM TERMINATION] Job ended by SLURM\n")
                log_file_handle.close()
                
            sys.exit(1)
            
        # Register the SLURM-specific handler
        signal.signal(signal.SIGTERM, slurm_signal_handler)
    
    # Automatically determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Population file
    population_file = os.path.join(script_dir, 'SSP2_2020.tif')
    
    # Flood file
    flood_file = os.path.join(script_dir, 'Europe_RP500_filled_depth.tif')
    flood_filename = os.path.basename(flood_file)
    
    # Income file 
    income_file = os.path.join(script_dir, 'Europe_disp_inc_2015.tif')
    
    # NUTS regions file
    nuts_file = os.path.join(script_dir, 'NUTS_RG_01M_2024_4326.geojson')
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'flood_analysis_results_new')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to capture console output to a file
    log_file = setup_logging(output_dir)
    
    # Print the paths being used
    print(f"Using population data: {population_file}")
    print(f"Using flood data: {flood_file}")
    print(f"Using income data: {income_file}")
    print(f"Using NUTS regions: {nuts_file}")
    print(f"Output will be saved to: {output_dir}")
    
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
    pop_data = load_population_data(population_file, bounds=region_bounds)
    print_timestamp("Loaded population data", loading_start)

    flood_data = load_flood_data(flood_file, bounds=region_bounds)
    print_timestamp("Loaded flood data", loading_start)

    income_data = load_income_data(income_file, bounds=region_bounds)
    print_timestamp("Loaded income data", loading_start)
    
    print_memory_usage("After loading all datasets")
    
    if pop_data is not None and flood_data is not None and income_data is not None:
        # Align all datasets - income is now required
        align_start = print_timestamp("Starting dataset alignment")  # Add this line to define align_start
        pop_data, flood_data, income_data = align_datasets(pop_data, flood_data, income_data)
        print_memory_usage("After dataset alignment")
        print_timestamp("Completed dataset alignment", align_start)
        print("All datasets successfully loaded and aligned")
        
        # Convert to dask array for parallel processing
        pop_data = pop_data.chunk({'x': 1000, 'y': 1000})
        # Operations will now utilize multiple cores automatically
        
        # Visualize data alignment if not skipping visualization
        if not skip_vis:
            vis_start = print_timestamp("Starting visualizations")
            visualize_data_alignment_with_nuts(pop_data, flood_data, income_data, nuts_file, output_dir)
            print_timestamp("Completed alignment visualization", vis_start)

        # Run analysis always with income data
        analysis_start = print_timestamp("Starting population analysis by depth and income")
        nuts_pop_by_depth_income = analyze_population_by_flood_depth_and_income(
            pop_data, flood_data, income_data=income_data, nuts_file=nuts_file, 
            region_name=region_name,
            nuts_level=nuts_level,
            cpu_cores=cpu_cores,
            chunk_size=chunk_size
        )
        print_timestamp("Completed population analysis", analysis_start)
        
        print_memory_usage("After main analysis")
        
        if not skip_vis:
            visualize_population_by_income_and_depth(
                nuts_pop_by_depth_income, output_dir, flood_filename
            )
            
        # Run vulnerability hotspot analysis
        analyze_vulnerable_areas(
            pop_data, flood_data, income_data, 
            flood_threshold=flood_threshold if flood_threshold is not None else 1.0,
            income_threshold=income_threshold if income_threshold is not None else 18000,
            output_dir=output_dir, 
            flood_filename=flood_filename,
            nuts_file=nuts_file,
            nuts_level=nuts_level
        )
        
        # Save income-stratified results to GeoJSON for GIS applications
        if output_dir:
            geojson_path = os.path.join(output_dir, 'europe_flood_depth_income_analysis.geojson')
            nuts_pop_by_depth_income.to_file(geojson_path, driver='GeoJSON')
            print(f"GeoJSON results with income analysis saved to {geojson_path}")

        print_memory_usage("After visualization")
        
        print("\n====== ANALYSIS COMPLETE ======")
        print(f"Results saved to {output_dir}")
        print(f"Log file saved to {log_file}")
    else:
        print("Error: Failed to load required datasets.")
        if pop_data is None:
            print("  - Population data could not be loaded")
        if flood_data is None:
            print("  - Flood data could not be loaded")
        if income_data is None:
            print("  - Income data could not be loaded")
    print_timestamp("Finished main analysis", start_time)

def analyze_population_by_flood_depth_and_income(population_data, flooding_data, income_data=None, nuts_file=None, 
                                               region_name=None, nuts_level=None, cpu_cores=None, chunk_size=None):
    """
    Analyze how many people are affected by different flood depth ranges with optional income stratification
    Uses multiprocessing for parallel region processing
    
    Parameters:
    - population_data: Population raster data
    - flooding_data: Flood depth raster data
    - income_data: Optional income raster data (if None, only basic flood analysis is performed)
    - nuts_file: Path to NUTS regions GeoJSON
    - region_name: Name of the study region
    - nuts_level: NUTS level to analyze (0-3)
    - cpu_cores: Number of CPU cores to use
    - chunk_size: Size of region chunks for processing
    
    Returns:
    - GeoDataFrame with analysis results
    """
    print(f"\n==== ANALYZING POPULATION BY FLOOD DEPTH" + 
          (f" AND INCOME ====" if income_data is not None else " ===="))
    
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
    nuts_level_regions = nuts_gdf[nuts_gdf['LEVL_CODE'] == nuts_level]
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

    # If no regions found, fall back to using all regions
    if len(study_regions) == 0:
        print(f"Warning: No NUTS level {nuts_level} regions found in study area. Using all available regions.")
        study_regions = nuts_gdf
        print(f"Using {len(study_regions)} regions")
    
    # Make data globally accessible for multi-processing
    global pop_data, flood_data, income_data_global
    pop_data = population_data
    flood_data = flooding_data
    
    # Only set income data if provided
    if income_data is not None:
        income_data_global = income_data
    
    # Set up multiprocessing pool
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_cores = int(slurm_cpus)
        print(f"Using {num_cores} CPU cores allocated by SLURM")
    else:
        num_cores = min(cpu_cores, multiprocessing.cpu_count()) if cpu_cores else multiprocessing.cpu_count()
        print(f"Using {num_cores} of {multiprocessing.cpu_count()} available CPU cores")
    
    # Prepare data for parallel processing
    region_data = [(idx, region) for idx, region in study_regions.iterrows()]
    total_regions = len(region_data)

    # Calculate chunk size based on data size and CPU cores
    if chunk_size is None:
        chunk_size = max(1, len(region_data) // (num_cores * 4))
    
    print(f"Processing {len(region_data)} regions in chunks of {chunk_size}")
    result_data = []

    start_time = time.time()
    processed_count = 0

    from concurrent.futures import ProcessPoolExecutor

    mp_start = print_timestamp("Starting parallel region processing")
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_region, region_data))
    print_timestamp("Completed parallel region processing", mp_start)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Finished processing {len(result_data)}/{total_regions} regions ({100.0:.1f}%) in {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Add debug information before creating the GeoDataFrame
    print(f"Results count: {len(result_data)}")
    print(f"Geometries count: {len(study_regions.geometry.values)}")
    
    # Count duplicate IDs in results
    nuts_ids = [r.get('nuts_id') for r in result_data]
    unique_nuts_ids = set(nuts_ids)
    print(f"Unique region IDs in results: {len(unique_nuts_ids)} out of {len(nuts_ids)}")
    
    # Create result GeoDataFrame with all regions' data
    # First create a mapping from region ID to geometry to ensure alignment
    region_id_to_geom = {region.get('NUTS_ID', f'unknown_{idx}'): region.geometry 
                        for idx, region in study_regions.iterrows()}
    
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
    
    plt.title(f"{flood_title}\n{region_display} Population by Income Level and Flood Depth")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{region_name}_income_depth_heatmap.png'), dpi=300, bbox_inches='tight')
    
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
    
    plt.figure(figsize=(12, 8))
    
    # Use a diverging colormap to emphasize differences in vulnerability
    sns.heatmap(
        percentage_impact_df,
        cmap='RdBu_r',  # Red for high impact (more vulnerable)
        annot=True,
        fmt='.2f',  # Show with 2 decimal places
        cbar_kws={'label': '% of Income Group Affected'},
    )
    
    plt.title(f"{flood_title}\nPercentage of Each Income Group Affected by Flood Depth in {region_display}")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{region_name}_income_vulnerability.png'), dpi=300, bbox_inches='tight')
    
    # 4. Save data as CSV for further analysis
    if output_dir:
        csv_path = os.path.join(output_dir, f'{region_name}_population_by_income_depth.csv')
        filtered_regions.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        # Also save the heatmap data
        heatmap_csv = os.path.join(output_dir, f'{region_name}_income_depth_matrix.csv')
        heatmap_df.to_csv(heatmap_csv)
        
        # And the percentage impact data
        impact_csv = os.path.join(output_dir, f'{region_name}_income_vulnerability_matrix.csv')
        percentage_impact_df.to_csv(impact_csv)
        
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
    
    if output_dir:
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
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'vienna_alignment_check.png'), dpi=300)
            print(f"Vienna alignment check saved to {os.path.join(output_dir, 'vienna_alignment_check.png')}")
            
        # Close figure to free memory
        plt.close(fig_vienna)
        
    else:
        print("Vienna region not found in NUTS data - skipping Vienna-specific alignment check")
        
    # Free memory
    import gc
    gc.collect()
    print_memory_usage("After Vienna visualization")

def analyze_vulnerable_areas(pop_data, flood_data, income_data, flood_threshold=1.0, income_threshold=15000, output_dir=None, flood_filename=None, nuts_file=None, nuts_level=None):
    """
    Identify and visualize areas where flooding exceeds a threshold depth AND income is below a threshold
    These represent especially vulnerable populations
    
    Parameters:
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent
    - income_data: xarray DataArray with income data (in PPP - Purchasing Power Parity)
    - flood_threshold: threshold above which an area is considered flooded
    - income_threshold: threshold below which income is considered low (in PPP)
    - output_dir: directory to save visualizations
    - flood_filename: filename of the flood dataset to extract return period
    - nuts_file: path to NUTS regions GeoJSON file
    """
    print("\n==== VULNERABLE AREAS ANALYSIS ====")
    print(f"Identifying areas where:")
    print(f"  - Flood depth > {flood_threshold}m")
    print(f"  - Income < {income_threshold} PPP")

    # Initialize variables
    vulnerable_pop = 0
    total_vulnerable_pop = 0
    
    # Parallelized approach for larger regions
    if nuts_file:
        # Load NUTS regions
        global nuts_gdf, pop_data_global, flood_data_global, income_data_global
        
        nuts_gdf = gpd.read_file(nuts_file)
        pop_data_global = pop_data
        flood_data_global = flood_data
        income_data_global = income_data
        
        # Filter to NUTS2 regions
        study_regions = nuts_gdf[nuts_gdf['LEVL_CODE'] == nuts_level]
        
        # Make sure regions intersect with our data
        x_min, y_min, x_max, y_max = (
            float(pop_data.x.min().values),
            float(pop_data.y.min().values),
            float(pop_data.x.max().values),
            float(pop_data.y.max().values)
        )
        study_area_poly = box(x_min, y_min, x_max, y_max)
        study_regions = study_regions[study_regions.geometry.intersects(study_area_poly)]
        
        # Set up processing parameters
        region_data = [(idx, region, flood_threshold, income_threshold) 
                      for idx, region in study_regions.iterrows()]
        total_regions = len(region_data)

        start_time = time.time()
        processed_count = 0

        
        # Use multiprocessing to analyze vulnerable areas for each region
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            num_cores = int(slurm_cpus)
            print(f"Using {num_cores} CPU cores allocated by SLURM")
            # Add chunk_size definition here
            chunk_size = max(1, len(region_data) // (num_cores * 4))
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"Using {num_cores} CPU cores for parallel processing")
            chunk_size = max(1, len(region_data) // (num_cores * 4))

        print(f"Processing {len(region_data)} regions in chunks of {chunk_size}")
        results = []
        with Pool(processes=num_cores) as pool:
            for chunk_idx, chunk in enumerate(chunk_data(region_data, chunk_size)):
                chunk_results = pool.map(process_vulnerable_region, chunk)
                results.extend(chunk_results)
                
                # Update processed count and calculate progress
                processed_count += len(chunk)
                percent_complete = (processed_count / total_regions) * 100
                
                # Calculate ETA
                elapsed_time = time.time() - start_time
                if processed_count > 0:
                    time_per_region = elapsed_time / processed_count
                    regions_remaining = total_regions - processed_count
                    eta_seconds = time_per_region * regions_remaining
                    eta_str = f"{int(eta_seconds//3600):02d}:{int((eta_seconds%3600)//60):02d}:{int(eta_seconds%60):02d}"
                    print(f"Vulnerability analysis progress: {processed_count}/{total_regions} regions ({percent_complete:.1f}%) - ETA: {eta_str}")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"Vulnerability analysis completed: {len(results)}/{total_regions} regions in {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        
        # Combine results
        total_vulnerable_pop = sum(r['vulnerable_population'] for r in results if r is not None)
        total_pop = sum(r['total_population'] for r in results if r is not None)
        
        # Create a visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create a GeoDataFrame with vulnerability percentages
        vuln_gdf = gpd.GeoDataFrame(
            [{
                'region_name': r['region_name'],
                'vulnerable_population': r['vulnerable_population'],
                'total_population': r['total_population'],
                'vulnerable_percentage': r['vulnerable_percentage'],
                'geometry': next(reg.geometry for idx, reg in study_regions.iterrows() 
                               if reg.get('NUTS_ID', '') == r['nuts_id'])
            } for r in results if r is not None and r['total_population'] > 0],
            geometry='geometry',
            crs=study_regions.crs
        )
        
        # Plot the map
        vuln_plot = vuln_gdf.plot(
            column='vulnerable_percentage',
            ax=ax,
            legend=True,
            cmap='YlOrRd',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            legend_kwds={'label': 'Percentage of Population (%)', 
                         'orientation': 'horizontal',
                         'shrink': 0.8,
                         'pad': 0.05}
        )
        
        # Add map annotations
        ax.set_title(f"Vulnerable Population by Region\n(Areas with Flood Depth > {flood_threshold:.1f}m & Income < {income_threshold:,.0f} PPP)", 
                    fontsize=14, pad=20)
        
        # Add explanatory text box
        ax.text(0.02, 0.02, 
                f"This map shows the percentage of each region's population that is considered vulnerable.\n"
                f"Vulnerability is defined as people living in areas where:\n"
                f"• Flood depth exceeds {flood_threshold:.1f}m\n"
                f"• Income is below {income_threshold:,.0f} PPP\n\n"
                f"Darker colors indicate regions with higher percentages of vulnerable population.",
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                verticalalignment='bottom')
        
        # Add analysis info
        total_vuln_percent = sum(r['vulnerable_population'] for r in results if r is not None) / sum(r['total_population'] for r in results if r is not None and r['total_population'] > 0) * 100
        ax.text(0.98, 0.98, 
                f"Total: {total_vuln_percent:.2f}% of population\n"
                f"across all regions is vulnerable",
                transform=ax.transAxes, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                horizontalalignment='right', verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'vulnerability_by_region.png'), dpi=300, bbox_inches='tight')
            vuln_gdf.drop(columns=['geometry']).to_csv(
                os.path.join(output_dir, 'vulnerability_by_region.csv'), index=False)
        
            print(f"Vulnerability results saved to {os.path.join(output_dir, 'vulnerability_by_region.csv')}")
    
    # If you also want to calculate vulnerable population for the entire area (non-region-specific)
    # Create masks for the entire dataset
    flood_mask = flood_data > flood_threshold
    income_mask = income_data < income_threshold
    combined_mask = flood_mask & income_mask
    
    # Calculate total vulnerable population across entire area
    vulnerable_pop = float(pop_data.where(combined_mask, 0).sum().values)
    
    print(f"Total vulnerable population: {total_vulnerable_pop:,.2f} (from regions)")
    print(f"Global vulnerable population: {vulnerable_pop:,.2f} (from entire area)")
          
    
    return vulnerable_pop, total_vulnerable_pop

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
        print("\nCleaning up logging resources...")
        sys.stdout.flush()
        if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'log'):
            sys.stdout.log.flush()
            sys.stdout.log.close()
        print("Log file closed properly.")
    
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

def process_region(region_data):
    """
    Process a single region in its own process
    Returns the result data for that region
    """
    # Declare global variables to be used in this worker function
    global pop_data, flood_data, nuts_gdf, depth_ranges

    # At the start of the function
    print_memory_usage(f"Start processing region {region_data[1].get('NAME_LATN', 'Unknown')}")

    idx, region = region_data
    region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
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
        print_memory_usage("After region clipping")
        flood_region = flood_data.rio.clip(region_gdf.geometry, region_gdf.crs)
        
        # Calculate total population in region
        total_pop = float(pop_region.sum().values)
        output_buffer.append(f"  Total population in region: {total_pop:,.2f}")
        
        # Create all masks at once
        depth_masks = [(flood_region > min_depth) & (flood_region <= max_depth) 
                      for min_depth, max_depth in depth_ranges]
        # Apply all masks in one operation
        affected_pops = [float(pop_region.where(mask, 0).sum().values) for mask in depth_masks]
        
        # Calculate population in each flood depth range
        depth_populations = {}
        for (min_depth, max_depth), affected_pop in zip(depth_ranges, affected_pops):
            if max_depth == float('inf'):
                range_name = f">{min_depth}m"
            else:
                range_name = f"{min_depth}-{max_depth}m"
            
            depth_populations[range_name] = affected_pop
            
            output_buffer.append(f"  Population affected by {range_name} flooding: {affected_pop:,.2f}")
        
        # Calculate total affected (any flooding)
        total_affected = float(pop_region.where(flood_region > 0, 0).sum().values)
        percentage_affected = (total_affected / total_pop * 100) if total_pop > 0 else 0
        output_buffer.append(f"  Total affected population: {total_affected:,.2f} ({percentage_affected:.2f}%)")
        
        # Store results
        result_row = {
            'region_name': region_name,
            'nuts_id': region.get('NUTS_ID', 'Unknown'),
            'nuts_level': region.get('LEVL_CODE', 'Unknown'),
            'total_population': total_pop,
            'total_affected': total_affected,
            'percentage_affected': percentage_affected,
            **depth_populations  # Unpacks all depth category data
        }
        
    except Exception as e:
        output_buffer.append(f"  Error processing region: {str(e)}")
        # Import traceback only when needed
        import traceback
        output_buffer.append(traceback.format_exc())
    
    # Print all buffered output at once (should be more atomic)
    print("\n".join(output_buffer))
    
    # At the end of the function
    print_memory_usage(f"End processing region {region_name}")
    print_timestamp(f"Completed processing for region: {region_name}", region_start)
    return result_row

def process_region_with_income(region_data):
    """
    Process a single region with income analysis in its own process
    Returns the result data for that region with income stratification
    """
    # Declare global variables to be used in this worker function
    global pop_data, flood_data, income_data_global, nuts_gdf, depth_ranges
    
    idx, region = region_data
    print(f"\nAnalyzing region: {region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))}")
    
    # Get region geometry
    region_geom = region.geometry
    
    # Initialize result to return in case of error
    result_row = {
        'region_name': region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown')),
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
        print(f"  Total population in region: {total_pop:,.2f}")
        
        # Calculate income thresholds for the region
        # We'll use the 33rd and 66th percentiles to divide into thirds
        income_values = income_region.values[~np.isnan(income_region.values)]
        if len(income_values) > 0:
            income_low_threshold = np.percentile(income_values, 33.33)
            income_high_threshold = np.percentile(income_values, 66.67)
            print(f"  Income thresholds - Low: <{income_low_threshold:.2f}, Medium: {income_low_threshold:.2f}-{income_high_threshold:.2f}, High: >{income_high_threshold:.2f}")
            
            # Create income masks
            low_income_mask = income_region <= income_low_threshold
            mid_income_mask = (income_region > income_low_threshold) & (income_region <= income_high_threshold)
            high_income_mask = income_region > income_high_threshold
        else:
            print("  Warning: No valid income data for this region")
            return result_row
        
        # Calculate population by income level
        pop_low_income = float(pop_region.where(low_income_mask, 0).sum().values)
        pop_mid_income = float(pop_region.where(mid_income_mask, 0).sum().values)
        pop_high_income = float(pop_region.where(high_income_mask, 0).sum().values)
        
        print(f"  Population by income level:")
        print(f"    Low income: {pop_low_income:,.2f} ({pop_low_income/total_pop*100:.2f}%)")
        print(f"    Medium income: {pop_mid_income:,.2f} ({pop_mid_income/total_pop*100:.2f}%)")
        print(f"    High income: {pop_high_income:,.2f} ({pop_high_income/total_pop*100:.2f}%)")
        
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
                depth_mask = (flood_region > min_depth) & (flood_region <= max_depth)
                range_name = f"{min_depth}-{max_depth}m"
            
            # Calculate affected population (total and by income level)
            affected_pop = float(pop_region.where(depth_mask, 0).sum().values)
            depth_all_income[range_name] = affected_pop
            
            # Calculate for each income level
            affected_low = float(pop_region.where(depth_mask & low_income_mask, 0).sum().values)
            affected_mid = float(pop_region.where(depth_mask & mid_income_mask, 0).sum().values)
            affected_high = float(pop_region.where(depth_mask & high_income_mask, 0).sum().values)
            
            depth_low_income[f"{range_name}_low"] = affected_low
            depth_mid_income[f"{range_name}_mid"] = affected_mid
            depth_high_income[f"{range_name}_high"] = affected_high
            
            # Fix division by zero errors with safe division
            if affected_pop > 0:
                low_pct = affected_low/affected_pop*100
                mid_pct = affected_mid/affected_pop*100
                high_pct = affected_high/affected_pop*100
            else:
                low_pct = mid_pct = high_pct = 0.0
        
            print(f"  Population affected by {range_name} flooding:")
            print(f"    Total: {affected_pop:,.2f}")
            print(f"    Low income: {affected_low:,.2f} ({low_pct:.2f}% of affected)")
            print(f"    Medium income: {affected_mid:,.2f} ({mid_pct:.2f}% of affected)")
            print(f"    High income: {affected_high:,.2f} ({high_pct:.2f}% of affected)")

            print(f"PROGRESS: {idx+1}/{len(region_data)} regions processed")
        
        # Calculate total affected population
        total_affected = float(pop_region.where(flood_region > 0, 0).sum().values)
        percentage_affected = (total_affected / total_pop * 100) if total_pop > 0 else 0
        print(f"  Total affected population: {total_affected:,.2f} ({percentage_affected:.2f}%)")
        
        # Calculate total affected by income level
        total_affected_low = float(pop_region.where((flood_region > 0) & low_income_mask, 0).sum().values)
        total_affected_mid = float(pop_region.where((flood_region > 0) & mid_income_mask, 0).sum().values)
        total_affected_high = float(pop_region.where((flood_region > 0) & high_income_mask, 0).sum().values)
        
        # Store results
        result_row = {
            'region_name': region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown')),
            'nuts_id': region.get('NUTS_ID', 'Unknown'),
            'nuts_level': region.get('LEVL_CODE', 'Unknown'),
            'total_population': total_pop,
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
        
    except Exception as e:
        print(f"  Error processing region: {str(e)}")
        traceback.print_exc()
    
    return result_row

def process_vulnerable_region(region_data):
    """Process a single region to identify vulnerable areas"""
    global nuts_gdf, pop_data_global, flood_data_global, income_data_global
    
    idx, region, flood_threshold, income_threshold = region_data
    region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
    nuts_id = region.get('NUTS_ID', 'Unknown')
    
    result = {
        'region_name': region_name,
        'nuts_id': nuts_id,
        'vulnerable_population': 0,
        'total_population': 0,
        'vulnerable_percentage': 0
    }
    
    try:
        # Clip data to region
        region_gdf = gpd.GeoDataFrame(geometry=[region.geometry], crs=nuts_gdf.crs)
        pop_region = pop_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
        flood_region = flood_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
        income_region = income_data_global.rio.clip(region_gdf.geometry, region_gdf.crs)
        
        # Calculate total population in region
        total_pop = float(pop_region.sum().values)
        print(f"  Total population in region: {total_pop:,.2f}")
        
        # Calculate income thresholds for the region - WITH ENHANCED DEBUGGING
        # We'll use the 33rd and 66th percentiles to divide into thirds
        income_values = income_region.values[~np.isnan(income_region.values)]
        print(f"  Income data shape: {income_region.shape}")
        print(f"  Valid income values count: {len(income_values)}")
        
        if len(income_values) > 0:
            # Print detailed income statistics
            print(f"  Income statistics:")
            print(f"    Min: {income_values.min():.2f}")
            print(f"    Max: {income_values.max():.2f}")
            print(f"    Mean: {income_values.mean():.2f}")
            print(f"    Median: {np.median(income_values):.2f}")
            print(f"    5th percentile: {np.percentile(income_values, 5):.2f}")
            print(f"    95th percentile: {np.percentile(income_values, 95):.2f}")
            
            # Calculate thresholds with clear debug info
            income_low_threshold = np.percentile(income_values, 33.33)
            income_high_threshold = np.percentile(income_values, 66.67)
            print(f"  Income thresholds:")
            print(f"    Low: <{income_low_threshold:.2f}")
            print(f"    Medium: {income_low_threshold:.2f}-{income_high_threshold:.2f}")
            print(f"    High: >{income_high_threshold:.2f}")
            
            # Verify distribution after applying thresholds
            low_count = np.sum(income_values <= income_low_threshold)
            mid_count = np.sum((income_values > income_low_threshold) & (income_values <= income_high_threshold))
            high_count = np.sum(income_values > income_high_threshold)
            
            print(f"  Income distribution verification:")
            print(f"    Low income cells: {low_count} ({low_count/len(income_values)*100:.2f}%)")
            print(f"    Mid income cells: {mid_count} ({mid_count/len(income_values)*100:.2f}%)")
            print(f"    High income cells: {high_count} ({high_count/len(income_values)*100:.2f}%)")

        # Create masks
        flood_mask = flood_region > flood_threshold
        income_mask = income_region < income_threshold
        combined_mask = flood_mask & income_mask
        
        # Calculate population
        total_pop = float(pop_region.sum().values)
        vulnerable_pop = float(pop_region.where(combined_mask, 0).sum().values)
        vulnerable_pct = (vulnerable_pop / total_pop * 100) if total_pop > 0 else 0
        
        # Update result
        result.update({
            'vulnerable_population': vulnerable_pop,
            'total_population': total_pop,
            'vulnerable_percentage': vulnerable_pct
        })
        
    except Exception as e:
        print(f"Error processing region {region_name} for vulnerability: {str(e)}")
    
    return result

# Add missing chunk_data function
def chunk_data(data, chunk_size):
    """Split data into chunks of specified size"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

if __name__ == "__main__":
    # For multiprocessing on Windows/Linux
    multiprocessing.freeze_support()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Call main with arguments from command line
    try:
        # Start with a print statement showing execution environment
        print(f"Starting flood analysis on {platform.node()} with Python {sys.version}")
        print(f"Environment variables: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'Not set')}")
        print(f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'Not set')}")
        
        status = main(
            region_bounds=args.bounds,
            output_dir=args.output,
            skip_vis=args.no_vis,
            nuts_level=args.nuts_level,
            cpu_cores=args.cores,
            chunk_size=args.chunk_size,
            flood_threshold=args.flood_threshold,
            income_threshold=args.income_threshold
        )
        sys.exit(status or 0)
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        traceback.print_exc()
        sys.exit(1)