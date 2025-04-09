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

def load_population_data(file_path, bounds=None, downsample_factor=1):
    """
    Load population grid data using rasterio with efficient memory usage
    """
    print(f"\n==== LOADING POPULATION DATA ====")
    print(f"Population file: {file_path}")
    print(f"Requested bounds: {bounds}")
    print(f"Downsample factor: {downsample_factor}")
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
            
            # Read a small sample to understand value distribution
            sample_data = src.read(1, window=((0,100), (0,100)))
            print(f"\nSample data statistics:")
            print(f"  Sample min: {np.nanmin(sample_data)}")
            print(f"  Sample max: {np.nanmax(sample_data)}")
            print(f"  Sample mean: {np.nanmean(sample_data)}")
            print(f"  Sample median: {np.nanmedian(sample_data)}")
            print(f"  Sample non-zero values: {np.count_nonzero(sample_data)}/{sample_data.size} pixels")
            print(f"  Sample with value > 10: {np.sum(sample_data > 10)} pixels")
            print(f"  Sample NaN count: {np.isnan(sample_data).sum()} pixels")
            
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
                    data = src.read(1, window=window)  # Removed out_shape parameter
                    
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
                    
                    # If you need to downsample for visualization or processing, do it separately
                    if downsample_factor > 1:
                        # Optional: Create a downsampled version for visualization/analysis
                        pop_data_downsampled = pop_data.coarsen(
                            y=downsample_factor, 
                            x=downsample_factor
                        ).sum()  # Use sum() not mean() to preserve total population
                        
                        print(f"Downsampled shape: {pop_data_downsampled.shape}")
                        print(f"Downsampled total population: {float(pop_data_downsampled.sum().values)}")
                        
                        # Calculate the ratio to verify totals are preserved
                        ratio = float(pop_data.sum().values) / float(pop_data_downsampled.sum().values)
                        print(f"Original/downsampled ratio: {ratio:.4f} (should be close to 1.0)")
                    
                    return pop_data
                    
                except Exception as e:
                    print(f"Error reading window: {str(e)}")
                    print("Falling back to reading a subsample of the entire dataset")
            
            # If bounds aren't provided or window reading failed, read a subsampled version of the entire dataset
            print(f"Reading subsampled population dataset with factor: {downsample_factor*10}")
            # Use a higher downsample factor for the full dataset to keep memory usage reasonable
            higher_downsample = downsample_factor * 10
            out_shape = (
                int(src.height / higher_downsample),
                int(src.width / higher_downsample)
            )
            data = src.read(1, out_shape=out_shape)
            
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
            
            # Check for specific areas if using full data
            if bounds is None or window.width <= 0:
                print("\nLooking at specific regions in the dataset:")
                # Check Vienna region specifically (approximately)
                vienna_x_idx = np.abs(pop_data.x - 16.3).argmin()
                vienna_y_idx = np.abs(pop_data.y - 48.2).argmin()
                print(f"  Vienna area coordinates: x={pop_data.x[vienna_x_idx].values}, y={pop_data.y[vienna_y_idx].values}")
                # Extract a small region around Vienna
                vienna_region = pop_data.isel(
                    x=slice(max(0, vienna_x_idx-2), min(len(pop_data.x), vienna_x_idx+3)),
                    y=slice(max(0, vienna_y_idx-2), min(len(pop_data.y), vienna_y_idx+3))
                )
                print(f"  Vienna region shape: {vienna_region.shape}")
                print(f"  Vienna region sum: {float(vienna_region.sum().values)}")
                print(f"  Vienna region max value: {float(vienna_region.max().values)}")
                
            return pop_data
            
    except Exception as e:
        print(f"Error loading population data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_flood_data(file_path, bounds=None, downsample_factor=5):
    """
    Load flood extent data using rasterio with efficient memory usage
    """
    print(f"\n==== LOADING FLOOD DATA ====")
    print(f"Flood file: {file_path}")
    print(f"Requested bounds: {bounds}")
    print(f"Downsample factor: {downsample_factor}")
    
    try:
        with rasterio.open(file_path) as src:
            print(f"\nFlood dataset metadata:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height} pixels")
            print(f"  Resolution: {src.res} degrees per pixel")
            print(f"  Transform: {src.transform}")
            print(f"  Reported bounds: {src.bounds}")
            print(f"  Nodata value: {src.nodata}")
            print(f"  Data type: {src.dtypes[0]}")
            
            # Read a small sample to understand value distribution
            sample_data = src.read(1, window=((0,100), (0,100)))
            print(f"\nSample data statistics:")
            print(f"  Sample min: {np.nanmin(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            print(f"  Sample max: {np.nanmax(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            print(f"  Sample mean: {np.nanmean(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            print(f"  Sample with value > 0: {np.sum(sample_data > 0)} pixels")
            print(f"  Sample NaN count: {np.isnan(sample_data).sum()} pixels")
            print(f"  Sample nodata count: {np.sum(sample_data == src.nodata)} pixels")
            
            # If bounds are provided, use them to read only the window of interest
            if bounds:
                print(f"Reading flood data in bounds: {bounds}")
                min_lon, min_lat, max_lon, max_lat = bounds
                
                try:
                    # Create a window using the bounds
                    window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                    
                    # Calculate output shape after downsampling
                    out_shape = (
                        int(window.height / downsample_factor),
                        int(window.width / downsample_factor)
                    )
                    
                    print(f"Reading window with shape: {out_shape}")
                    
                    # Read the data for the specified window with downsampling
                    data = src.read(1, window=window, out_shape=out_shape)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Get the transform for the window
                    window_transform = src.window_transform(window)
                    
                    # Create an xarray DataArray
                    # Calculate coordinates for the new data
                    height, width = data.shape
                    x_coords = np.linspace(min_lon, max_lon, width)
                    y_coords = np.linspace(max_lat, min_lat, height)
                    
                    flood_data = xr.DataArray(
                        data[np.newaxis, :, :],  # Add band dimension
                        dims=['band', 'y', 'x'],
                        coords={
                            'band': [1],
                            'y': y_coords,
                            'x': x_coords
                        }
                    )
                    
                    # Add CRS information
                    flood_data.rio.write_crs(src.crs, inplace=True)
                    
                    print(f"Flood data loaded with shape: {flood_data.shape}")
                    print(f"Flood data range: {np.nanmin(data)} to {np.nanmax(data)}")
                    
                    # After reading data from window or full dataset
                    print(f"\nFlood data statistics:")
                    print(f"  Data shape: {data.shape}")
                    print(f"  Value range: {np.nanmin(data)} to {np.nanmax(data)}")
                    print(f"  Median depth: {np.nanmedian(data)}")
                    print(f"  Mean depth: {np.nanmean(data)}")
                    print(f"  Cells with depth > 0.5m: {np.sum(data > 0.5)}")
                    print(f"  Cells with depth > 1.0m: {np.sum(data > 1.0)}")
                    print(f"  NaN count: {np.isnan(data).sum()}")
                    
                    return flood_data
                
                except Exception as e:
                    print(f"Error reading window: {e}")
                    print("Falling back to downsampling the entire dataset")
            
            # If bounds aren't provided or window reading failed, read a downsampled version of the entire dataset
            print(f"Reading entire flood dataset with downsample factor: {downsample_factor}")
            out_shape = (
                int(src.height / downsample_factor),
                int(src.width / downsample_factor)
            )
            data = src.read(1, out_shape=out_shape)
            
            # Handle nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            # Create an xarray DataArray
            height, width = data.shape
            x_coords = np.linspace(src.bounds.left, src.bounds.right, width)
            y_coords = np.linspace(src.bounds.top, src.bounds.bottom, height)
            
            flood_data = xr.DataArray(
                data[np.newaxis, :, :],  # Add band dimension
                dims=['band', 'y', 'x'],
                coords={
                    'band': [1],
                    'y': y_coords,
                    'x': x_coords
                }
            )
            
            # Add CRS information
            flood_data.rio.write_crs(src.crs, inplace=True)
            
            print(f"Flood data loaded with shape: {flood_data.shape}")
            print(f"Flood data range: {np.nanmin(data)} to {np.nanmax(data)}")
            
            # After creating final DataArray
            print(f"\nFinal flood data statistics:")
            print(f"  Shape: {flood_data.shape}")
            print(f"  X coords range: {float(flood_data.x.min().values)} to {float(flood_data.x.max().values)}")
            print(f"  Y coords range: {float(flood_data.y.min().values)} to {float(flood_data.y.max().values)}")
            print(f"  Value range: {float(flood_data.min().values)} to {float(flood_data.max().values)}")
            
            return flood_data
            
    except Exception as e:
        print(f"Error loading flood data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_income_data(file_path, bounds=None, downsample_factor=5):
    """
    Load disposable income data using rasterio with efficient memory usage
    """
    print(f"\n==== LOADING INCOME DATA ====")
    print(f"Income file: {file_path}")
    print(f"Requested bounds: {bounds}")
    print(f"Downsample factor: {downsample_factor}")
    
    try:
        with rasterio.open(file_path) as src:
            print(f"\nIncome dataset metadata:")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height} pixels")
            print(f"  Resolution: {src.res} degrees per pixel")
            print(f"  Transform: {src.transform}")
            print(f"  Reported bounds: {src.bounds}")
            print(f"  Nodata value: {src.nodata}")
            print(f"  Data type: {src.dtypes[0]}")
            
            # Read a small sample to understand value distribution
            sample_data = src.read(1, window=((0,100), (0,100)))
            print(f"\nSample data statistics:")
            print(f"  Sample min: {np.nanmin(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            print(f"  Sample max: {np.nanmax(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            print(f"  Sample mean: {np.nanmean(sample_data) if not np.all(np.isnan(sample_data)) else 'All NaN'}")
            
            # If bounds are provided, use them to read only the window of interest
            if bounds:
                print(f"Reading income data in bounds: {bounds}")
                min_lon, min_lat, max_lon, max_lat = bounds
                
                try:
                    # Create a window using the bounds
                    window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                    
                    # Calculate output shape after downsampling
                    out_shape = (
                        int(window.height / downsample_factor),
                        int(window.width / downsample_factor)
                    )
                    
                    print(f"Reading window with shape: {out_shape}")
                    data = src.read(1, window=window, out_shape=out_shape)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    
                    # Create an xarray DataArray
                    height, width = data.shape
                    x_coords = np.linspace(min_lon, max_lon, width)
                    y_coords = np.linspace(max_lat, min_lat, height)
                    
                    income_data = xr.DataArray(
                        data[np.newaxis, :, :],  # Add band dimension
                        dims=['band', 'y', 'x'],
                        coords={
                            'band': [1],
                            'y': y_coords,
                            'x': x_coords
                        }
                    )
                    
                    # Add CRS information
                    income_data.rio.write_crs(src.crs, inplace=True)
                    
                    print(f"Income data loaded with shape: {income_data.shape}")
                    print(f"Income data range: {float(income_data.min().values)} to {float(income_data.max().values)}")
                    
                    return income_data
                    
                except Exception as e:
                    print(f"Error reading window: {e}")
                    print("Falling back to downsampling the entire dataset")
            
            # If bounds aren't provided or window reading failed, read the entire dataset
            print(f"Reading entire income dataset with downsample factor: {downsample_factor}")
            out_shape = (
                int(src.height / downsample_factor),
                int(src.width / downsample_factor)
            )
            data = src.read(1, out_shape=out_shape)
            
            # Handle nodata values
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            # Create an xarray DataArray
            height, width = data.shape
            x_coords = np.linspace(src.bounds.left, src.bounds.right, width)
            y_coords = np.linspace(src.bounds.top, src.bounds.bottom, height)
            
            income_data = xr.DataArray(
                data[np.newaxis, :, :],
                dims=['band', 'y', 'x'],
                coords={
                    'band': [1],
                    'y': y_coords,
                    'x': x_coords
                }
            )
            
            # Add CRS information
            income_data.rio.write_crs(src.crs, inplace=True)
            
            print(f"Income data loaded with shape: {income_data.shape}")
            print(f"Income data range: {float(income_data.min().values)} to {float(income_data.max().values)}")
            
            return income_data
            
    except Exception as e:
        print(f"Error loading income data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def align_datasets(pop_data, flood_data):
    """
    Reproject and align datasets to the same grid and CRS if needed
    """
    print(f"\n==== ALIGNING DATASETS ====")
    print(f"Population data: {pop_data.shape} with CRS {pop_data.rio.crs}")
    print(f"Flood data: {flood_data.shape} with CRS {flood_data.rio.crs}")
    
    print(f"Population data resolution: {pop_data.rio.resolution()}")
    print(f"Flood data resolution: {flood_data.rio.resolution()}")
    
    print(f"Population bounds: {pop_data.rio.bounds()}")
    print(f"Flood bounds: {flood_data.rio.bounds()}")
    
    # Check for coordinate overlap
    pop_x_min, pop_y_min, pop_x_max, pop_y_max = pop_data.rio.bounds()
    flood_x_min, flood_y_min, flood_x_max, flood_y_max = flood_data.rio.bounds()
    
    x_overlap = max(0, min(pop_x_max, flood_x_max) - max(pop_x_min, flood_x_min))
    y_overlap = max(0, min(pop_y_max, flood_y_max) - max(pop_y_min, flood_y_min))
    
    print(f"Coordinate overlap: X={x_overlap} degrees, Y={y_overlap} degrees")
    
    try:
        # Check if reprojection is needed
        if pop_data.rio.crs != flood_data.rio.crs:
            print(f"Reprojecting flood data from {flood_data.rio.crs} to {pop_data.rio.crs}")
            flood_data_before = flood_data.copy()
            flood_data = flood_data.rio.reproject_match(pop_data)
            print(f"Flood data shape after reprojection: {flood_data.shape}")
            print(f"Flood data range after reprojection: {float(flood_data.min().values)} to {float(flood_data.max().values)}")
            print(f"Cells affected by flooding before: {(flood_data_before > 0).sum().values.item()}")
            print(f"Cells affected by flooding after: {(flood_data > 0).sum().values.item()}")
        
        # If dimensions/resolutions don't match, resample
        if pop_data.rio.resolution() != flood_data.rio.resolution():
            print("Resampling datasets to match resolution...")
            flood_data_before = flood_data.copy()
            flood_data = flood_data.rio.reproject_match(pop_data)
            print(f"Flood data shape after resampling: {flood_data.shape}")
            print(f"Flood data range after resampling: {float(flood_data.min().values)} to {float(flood_data.max().values)}")
            print(f"Cells affected by flooding before: {(flood_data_before > 0).sum().values.item()}")
            print(f"Cells affected by flooding after: {(flood_data > 0).sum().values.item()}")
            
        return pop_data, flood_data
    except Exception as e:
        print(f"Error aligning datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        return pop_data, flood_data

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
    
    plt.show()

def main(scale_factor=None):
    """
    Main function to analyze population affected by flooding by depth category and income level
    for Vienna region
    """
    # Automatically determine file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Population file - using SSP2_2020.tif
    population_file = os.path.join(script_dir, 'SPP2', 'SSP2_2020.tif')
    
    # Flood file - using Europe_RP500_filled_depth.tif 
    flood_file = os.path.join(script_dir, 'Europe_RP500_filled_depth.tif')
    flood_filename = os.path.basename(flood_file)
    
    # Income file - Europe disposable income
    income_file = os.path.join(script_dir, 'dataverse_files', 'Europe_disp_inc_2015.tif')
    
    # NUTS regions file
    nuts_file = os.path.join(script_dir, 'NUTS_RG_01M_2024_4326.geojson')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, 'flood_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to capture console output to a file
    log_file = setup_logging(output_dir)
    
    # Print the paths being used
    print(f"Using population data: {population_file}")
    print(f"Using flood data: {flood_file}")
    print(f"Using income data: {income_file}")
    print(f"Using NUTS regions: {nuts_file}")
    print(f"Output will be saved to: {output_dir}")
    
    # Set Vienna-specific parameters
    region_bounds = (16.0, 47.9, 16.7, 48.4)
    region_name = "Vienna"
    
    print(f"Analysis region: {region_name}")
    print(f"Bounds: {region_bounds}")
    
    # Load all datasets
    pop_data = load_population_data(population_file, bounds=region_bounds, downsample_factor=1)
    flood_data = load_flood_data(flood_file, bounds=region_bounds, downsample_factor=1)
    income_data = load_income_data(income_file, bounds=region_bounds, downsample_factor=1)
    
    if pop_data is not None and flood_data is not None and income_data is not None:
        # Align all datasets
        pop_data, flood_data = align_datasets(pop_data, flood_data)
        income_data = income_data.rio.reproject_match(pop_data)
        
        print("All datasets successfully loaded and aligned")
        
        # Add this new line to check alignment visually
        visualize_data_alignment_with_nuts(pop_data, flood_data, income_data, nuts_file, output_dir)
        
        # Run standard analysis (keep existing functionality)
        nuts_pop_by_depth = analyze_population_by_flood_depth(
            pop_data, flood_data, nuts_file, region_name=region_name
        )
        
        visualize_population_by_flood_depth(
            nuts_pop_by_depth, output_dir, flood_filename
        )
        
        # Run new analysis with income stratification
        nuts_pop_by_depth_income = analyze_population_by_flood_depth_and_income(
            pop_data, flood_data, income_data, nuts_file, region_name=region_name
        )
        
        visualize_population_by_income_and_depth(
            nuts_pop_by_depth_income, output_dir, flood_filename
        )
        
        # NEW: Run vulnerability hotspot analysis
        # For initial query (may auto-adjust thresholds as needed)
        analyze_vulnerable_areas(
            pop_data, flood_data, income_data, 
            flood_threshold=1.0, income_threshold=18000,
            output_dir=output_dir, flood_filename=flood_filename,
            nuts_file=nuts_file  # Pass the nuts_file parameter
        )
        
        # Save results to GeoJSON for GIS applications
        if output_dir:
            geojson_path = os.path.join(output_dir, 'vienna_flood_depth_income_analysis.geojson')
            nuts_pop_by_depth_income.to_file(geojson_path, driver='GeoJSON')
            print(f"GeoJSON results with income analysis saved to {geojson_path}")
        
        print("\n====== ANALYSIS COMPLETE ======")
        print(f"Results saved to {output_dir}")
        print(f"Log file saved to {log_file}")
    else:
        print("Error: Failed to load required datasets.")

def analyze_population_by_flood_depth(pop_data, flood_data, nuts_file, region_name="Wien"):
    """
    Analyze how many people are affected by different flood depth ranges in a specific region
    """
    print(f"\n==== ANALYZING POPULATION BY FLOOD DEPTH ====")
    
    # Define the flood depth ranges (in meters)
    depth_ranges = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, float('inf'))  # Everything above 6 meters
    ]
    
    # Load NUTS regions
    nuts_gdf = gpd.read_file(nuts_file)
    print(f"Loaded {len(nuts_gdf)} NUTS regions")
    
    # Get Vienna regions - be more inclusive by checking all Wien-related areas
    vienna_area_names = ["Wien", "Vienna", "AT13", "Wiener"]
    
    # Use a list to collect matching regions instead of an empty GeoDataFrame
    matching_regions_indices = []
    
    # Find all Vienna-related regions
    for name in vienna_area_names:
        for field in ['NAME_LATN', 'NUTS_NAME', 'CNTR_CODE', 'NUTS_ID']:
            if field in nuts_gdf.columns:
                regions = nuts_gdf[nuts_gdf[field].str.contains(name, case=False, na=False)]
                if len(regions) > 0:
                    # Add indices to our list
                    matching_regions_indices.extend(regions.index.tolist())
                    print(f"Found {len(regions)} regions matching '{name}' in field '{field}'")
    
    # Remove duplicates and create vienna_regions from the original GeoDataFrame
    unique_indices = list(set(matching_regions_indices))
    vienna_regions = nuts_gdf.loc[unique_indices]
    print(f"Total of {len(vienna_regions)} unique Vienna-related regions found")
    
    # If no Vienna regions found, use a fallback approach
    if len(vienna_regions) == 0:
        print(f"Warning: Could not find Vienna in NUTS regions. Using all regions in the study area.")
        # Use all regions that intersect with our data extent
        x_min, y_min, x_max, y_max = (
            float(pop_data.x.min().values),
            float(pop_data.y.min().values),
            float(pop_data.x.max().values),
            float(pop_data.y.max().values)
        )
        study_area_poly = box(x_min, y_min, x_max, y_max)
        nuts_gdf['intersects_study_area'] = nuts_gdf.geometry.intersects(study_area_poly)
        vienna_regions = nuts_gdf[nuts_gdf['intersects_study_area']]
        print(f"Using {len(vienna_regions)} NUTS regions that intersect with the study area")
    
    # Initialize result dataframe
    result_data = []
    
    # Process each NUTS region
    for idx, region in vienna_regions.iterrows():
        print(f"\nAnalyzing region: {region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))}")
        
        # Get region geometry
        region_geom = region.geometry
        
        # Clip population and flood data to this region
        try:
            # Convert region geometry to GeoDataFrame for clipping
            region_gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=nuts_gdf.crs)
            
            # Clip the data to the region
            pop_region = pop_data.rio.clip(region_gdf.geometry, region_gdf.crs)
            flood_region = flood_data.rio.clip(region_gdf.geometry, region_gdf.crs)
            
            # Calculate total population in region
            total_pop = float(pop_region.sum().values)
            print(f"  Total population in region: {total_pop:,.2f}")
            
            # Calculate population in each flood depth range
            depth_populations = {}
            for min_depth, max_depth in depth_ranges:
                # Create mask for this depth range
                if max_depth == float('inf'):
                    depth_mask = flood_region > min_depth
                    range_name = f">{min_depth}m"
                else:
                    depth_mask = (flood_region > min_depth) & (flood_region <= max_depth)
                    range_name = f"{min_depth}-{max_depth}m"
                
                # Calculate affected population
                affected_pop = float(pop_region.where(depth_mask, 0).sum().values)
                depth_populations[range_name] = affected_pop
                
                print(f"  Population affected by {range_name} flooding: {affected_pop:,.2f}")
            
            # Calculate total affected (any flooding)
            total_affected = float(pop_region.where(flood_region > 0, 0).sum().values)
            percentage_affected = (total_affected / total_pop * 100) if total_pop > 0 else 0
            print(f"  Total affected population: {total_affected:,.2f} ({percentage_affected:.2f}%)")
            
            # Store results
            result_row = {
                'region_name': region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown')),
                'nuts_id': region.get('NUTS_ID', 'Unknown'),
                'nuts_level': region.get('LEVL_CODE', 'Unknown'),
                'total_population': total_pop,
                'total_affected': total_affected,
                'percentage_affected': percentage_affected,
                **depth_populations  # Unpacks all depth category data
            }
            result_data.append(result_row)
            
        except Exception as e:
            print(f"  Error processing region: {str(e)}")
            # Import traceback only when needed
            import traceback
            traceback.print_exc()
    
    # Create result GeoDataFrame with all regions' data
    result_gdf = gpd.GeoDataFrame(
        result_data,
        geometry=vienna_regions.geometry.values,
        crs=vienna_regions.crs
    )
    
    # Add after creating result_gdf
    vienna_total = sum(row['total_population'] for row in result_data)
    print(f"\n==== VIENNA REGION TOTAL POPULATION ====")
    print(f"Total population in study area: {vienna_total:,.2f}")

    # If you have AT13/AT130 Vienna specifically:
    vienna_city = next((row for row in result_data if row.get('nuts_id') in ['AT13', 'AT130']), None)
    if vienna_city:
        print(f"Vienna city population: {vienna_city['total_population']:,.2f}")
    print(f"Note: These figures represent the population within the study area bounds.")
    
    # Before returning the result_gdf, add this debug info:
    print("\n==== NUTS REGIONS DEBUG INFO ====")
    for idx, region in vienna_regions.iterrows():
        region_name = region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))
        nuts_id = region.get('NUTS_ID', 'Unknown')
        nuts_level = region.get('LEVL_CODE', 'Unknown')
        print(f"Region: {region_name}, ID: {nuts_id}, Level: {nuts_level}")
        
        # Check if this region intersects with our data
        region_geom = region.geometry
        x_min, y_min, x_max, y_max = (
            float(pop_data.x.min().values),
            float(pop_data.y.min().values),
            float(pop_data.x.max().values),
            float(pop_data.y.max().values)
        )
        data_box = box(x_min, y_min, x_max, y_max)
        
        intersects = region_geom.intersects(data_box)
        print(f"  Intersects with data extent: {intersects}")
        if intersects:
            overlap_area = region_geom.intersection(data_box).area
            region_area = region_geom.area
            print(f"  Overlap percentage: {overlap_area/region_area*100:.2f}%")
    
    return result_gdf

def visualize_population_by_flood_depth(nuts_pop_by_depth, output_dir=None, flood_filename=None):
    """
    Create visualizations of population affected by different flood depth ranges
    """
    print("\n==== VISUALIZING POPULATION BY FLOOD DEPTH ====")
    
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
    
    # 1. Create a map showing total affected population by region
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Create a copy for plotting to avoid SettingWithCopyWarning
    plot_gdf = nuts_pop_by_depth.copy()
    
    # Simple plot without complex legend keywords
    plot_gdf.plot(
        column='percentage_affected',
        ax=ax,
        legend=True,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add a simple title to the legend manually
    ax.get_figure().get_axes()[1].set_title('Affected Population (%)')
    
    # Add labels for regions
    for idx, row in plot_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.annotate(
            text=row['region_name'],
            xy=(centroid.x, centroid.y),
            ha='center',
            fontsize=8,
            color='black',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')]
        )
    
    ax.set_title(f"{flood_title}\nPercentage of Population Affected")
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'affected_population_map.png'), dpi=300, bbox_inches='tight')
    
    # 2. Create stacked bar chart of population by flood depth for each region
    depth_columns = [col for col in nuts_pop_by_depth.columns 
                     if any(col.startswith(f"{d}-") or col.startswith(f">{d}") 
                            for d in ['0', '1', '2', '4', '6'])]
    
    if depth_columns:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Prepare data for plotting
        plot_data = nuts_pop_by_depth[['region_name'] + depth_columns].copy()
        plot_data.set_index('region_name', inplace=True)
        
        # Plot stacked bar chart
        plot_data.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            colormap='viridis',
            rot=45
        )
        
        ax.set_title(f"{flood_title}\nPopulation Affected by Flood Depth Categories")
        ax.set_ylabel('Population')
        ax.set_xlabel('Region')
        ax.legend(title='Flood Depth')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'population_by_flood_depth.png'), dpi=300, bbox_inches='tight')
    
    # 3. Create a heatmap table visualization - only if we have many regions
    if len(nuts_pop_by_depth) > 5:  # Only create heatmap for larger datasets
        fig, ax = plt.subplots(1, 1, figsize=(14, len(nuts_pop_by_depth) * 0.8 + 2))
        
        # Prepare data for heatmap
        heatmap_data = nuts_pop_by_depth[['region_name'] + depth_columns].copy()
        heatmap_data.set_index('region_name', inplace=True)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Population Affected'},
            ax=ax
        )
        
        ax.set_title(f"{flood_title}\nDetailed Population Exposure by Flood Depth")
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'flood_depth_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Show plots if not in batch mode
    plt.show()
    
    # 4. Save data as CSV for further analysis
    if output_dir:
        csv_path = os.path.join(output_dir, 'population_by_flood_depth.csv')
        nuts_pop_by_depth.drop(columns=['geometry']).to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

def analyze_population_by_flood_depth_and_income(pop_data, flood_data, income_data, nuts_file, region_name="Wien"):
    """
    Analyze how many people are affected by different flood depth ranges and income levels
    """
    print(f"\n==== ANALYZING POPULATION BY FLOOD DEPTH AND INCOME ====")
    
    # Define the flood depth ranges (in meters)
    depth_ranges = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, float('inf'))  # Everything above 6 meters
    ]
    
    # Load NUTS regions
    nuts_gdf = gpd.read_file(nuts_file)
    print(f"Loaded {len(nuts_gdf)} NUTS regions")
    
    # Get Vienna regions - be more inclusive by checking all Wien-related areas
    vienna_area_names = ["Wien", "Vienna", "AT13", "Wiener"]
    matching_regions_indices = []
    
    # Find all Vienna-related regions (same as your existing code)
    for name in vienna_area_names:
        for field in ['NAME_LATN', 'NUTS_NAME', 'CNTR_CODE', 'NUTS_ID']:
            if field in nuts_gdf.columns:
                regions = nuts_gdf[nuts_gdf[field].str.contains(name, case=False, na=False)]
                if len(regions) > 0:
                    matching_regions_indices.extend(regions.index.tolist())
                    print(f"Found {len(regions)} regions matching '{name}' in field '{field}'")
    
    unique_indices = list(set(matching_regions_indices))
    vienna_regions = nuts_gdf.loc[unique_indices]
    print(f"Total of {len(vienna_regions)} unique Vienna-related regions found")
    
    # Fallback if no Vienna regions found (same as your existing code)
    if len(vienna_regions) == 0:
        print(f"Warning: Could not find Vienna in NUTS regions. Using all regions in the study area.")
        x_min, y_min, x_max, y_max = (
            float(pop_data.x.min().values),
            float(pop_data.y.min().values),
            float(pop_data.x.max().values),
            float(pop_data.y.max().values)
        )
        study_area_poly = box(x_min, y_min, x_max, y_max)
        nuts_gdf['intersects_study_area'] = nuts_gdf.geometry.intersects(study_area_poly)
        vienna_regions = nuts_gdf[nuts_gdf['intersects_study_area']]
        print(f"Using {len(vienna_regions)} NUTS regions that intersect with the study area")
    
    # Initialize result dataframe
    result_data = []
    
    # Process each NUTS region
    for idx, region in vienna_regions.iterrows():
        print(f"\nAnalyzing region: {region.get('NAME_LATN', region.get('NUTS_NAME', 'Unknown'))}")
        
        # Get region geometry
        region_geom = region.geometry
        
        try:
            # Convert region geometry to GeoDataFrame for clipping
            region_gdf = gpd.GeoDataFrame(geometry=[region_geom], crs=nuts_gdf.crs)
            
            # Clip the data to the region
            pop_region = pop_data.rio.clip(region_gdf.geometry, region_gdf.crs)
            flood_region = flood_data.rio.clip(region_gdf.geometry, region_gdf.crs)
            income_region = income_data.rio.clip(region_gdf.geometry, region_gdf.crs)
            
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
                continue
            
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
                
                print(f"  Population affected by {range_name} flooding:")
                print(f"    Total: {affected_pop:,.2f}")
                print(f"    Low income: {affected_low:,.2f} ({affected_low/affected_pop*100:.2f}% of affected)")
                print(f"    Medium income: {affected_mid:,.2f} ({affected_mid/affected_pop*100:.2f}% of affected)")
                print(f"    High income: {affected_high:,.2f} ({affected_high/affected_pop*100:.2f}% of affected)")
            
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
            result_data.append(result_row)
            
        except Exception as e:
            print(f"  Error processing region: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create result GeoDataFrame with all regions' data
    result_gdf = gpd.GeoDataFrame(
        result_data,
        geometry=vienna_regions.geometry.values,
        crs=vienna_regions.crs
    )
    
    return result_gdf

def visualize_population_by_income_and_depth(nuts_pop_by_depth_income, output_dir=None, flood_filename=None):
    """
    Create focused visualizations of population affected by different flood depths stratified by income
    Specifically for Vienna proper (NUTS ID: AT13 or AT130)
    """
    print("\n==== VISUALIZING POPULATION BY INCOME AND FLOOD DEPTH FOR VIENNA ====")
    
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
    
    # Filter for Vienna proper (AT13 or AT130)
    vienna_proper = nuts_pop_by_depth_income[
        nuts_pop_by_depth_income['nuts_id'].isin(['AT13', 'AT130'])
    ].copy()
    
    if len(vienna_proper) == 0:
        print("Warning: No data found for Vienna proper (NUTS ID: AT13 or AT130).")
        print("Using all data for visualization instead.")
        vienna_proper = nuts_pop_by_depth_income.copy()
    else:
        print(f"Found {len(vienna_proper)} regions matching Vienna proper.")
    
    # 1. VISUALIZATION: Income-stratified flood impact summary chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # First subplot: Affected population by income level
    vienna_affected = vienna_proper[['region_name', 'low_income_affected', 'mid_income_affected', 'high_income_affected']].copy()
    total_affected = vienna_affected[['low_income_affected', 'mid_income_affected', 'high_income_affected']].sum()
    
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
    vienna_pop = vienna_proper[['region_name', 'low_income_population', 'mid_income_population', 'high_income_population']].copy()
    total_pop = vienna_pop[['low_income_population', 'mid_income_population', 'high_income_population']].sum()
    
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
    
    plt.suptitle(f"{flood_title}\nVienna Income Distribution Analysis", fontsize=16)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vienna_income_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 2. VISUALIZATION: Detailed heatmap for Vienna only
    # Identify depth ranges and income levels
    depth_ranges = ['0.0-0.5m', '0.5-1.0m', '1.0-2.0m', '2.0-4.0m', '4.0-6.0m', '>6.0m']
    income_levels = ['low', 'mid', 'high']
    
    # Create consolidated data for all Vienna regions
    consolidated_data = {depth: {'Low': 0, 'Mid': 0, 'High': 0} for depth in depth_ranges}
    
    # Sum up data across all Vienna regions
    for _, region_row in vienna_proper.iterrows():
        for depth in depth_ranges:
            for income in income_levels:
                col_name = f"{depth}_{income}"
                if col_name in region_row:
                    income_cap = income.capitalize()
                    consolidated_data[depth][income_cap] += region_row[col_name]
    
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
    
    plt.title(f"{flood_title}\nVienna Population by Income Level and Flood Depth")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vienna_income_depth_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 3. VISUALIZATION: Percentage of income groups affected by each depth category
    # Calculate what percentage of each income group is affected at each depth
    income_totals = {
        'Low': vienna_proper['low_income_population'].sum(),
        'Mid': vienna_proper['mid_income_population'].sum(),
        'High': vienna_proper['high_income_population'].sum()
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
    
    plt.title(f"{flood_title}\nPercentage of Each Income Group Affected by Flood Depth in Vienna")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vienna_income_vulnerability.png'), dpi=300, bbox_inches='tight')
    
    # 4. Save data as CSV for further analysis (but only the Vienna data)
    if output_dir:
        csv_path = os.path.join(output_dir, 'vienna_population_by_income_depth.csv')
        vienna_proper.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        # Also save the heatmap data
        heatmap_csv = os.path.join(output_dir, 'vienna_income_depth_matrix.csv')
        heatmap_df.to_csv(heatmap_csv)
        
        # And the percentage impact data
        impact_csv = os.path.join(output_dir, 'vienna_income_vulnerability_matrix.csv')
        percentage_impact_df.to_csv(impact_csv)
        
        print(f"Vienna income analysis results saved to {csv_path}")
    
    plt.show()

def visualize_data_alignment_with_nuts(pop_data, flood_data, income_data, nuts_file, output_dir=None):
    """
    Create visualizations of datasets with NUTS regions overlaid to verify alignment
    
    Parameters:
    - pop_data: xarray DataArray with population counts
    - flood_data: xarray DataArray with flood extent
    - income_data: xarray DataArray with income data
    - nuts_file: path to NUTS regions GeoJSON file
    - output_dir: directory to save visualizations
    """
    print("\n==== VISUALIZING DATA ALIGNMENT WITH NUTS REGIONS ====")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load NUTS regions
    nuts_gdf = None
    if nuts_file and os.path.exists(nuts_file):
        try:
            nuts_gdf = gpd.read_file(nuts_file)
            print(f"Loaded {len(nuts_gdf)} NUTS regions for alignment check")
        except Exception as e:
            print(f"Error loading NUTS regions: {str(e)}")
            return
    else:
        print("NUTS file not found, skipping alignment check")
        return
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Population density with NUTS overlay
    pop_data.plot(ax=axs[0], cmap='viridis', vmin=0, add_colorbar=True)
    axs[0].set_title('Population Data with NUTS Regions')
    nuts_gdf.boundary.plot(ax=axs[0], color='red', linewidth=0.7)
    
    # Add text showing the CRS
    axs[0].text(0.01, 0.01, f"CRS: {pop_data.rio.crs}", 
                transform=axs[0].transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot 2: Flood data with NUTS overlay
    flood_data.plot(ax=axs[1], cmap='Blues', add_colorbar=True)
    axs[1].set_title('Flood Data with NUTS Regions')
    nuts_gdf.boundary.plot(ax=axs[1], color='red', linewidth=0.7)
    
    # Add text showing the CRS
    axs[1].text(0.01, 0.01, f"CRS: {flood_data.rio.crs}", 
                transform=axs[1].transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot 3: Income data with NUTS overlay
    income_data.plot(ax=axs[2], cmap='plasma', add_colorbar=True)
    axs[2].set_title('Income Data with NUTS Regions')
    nuts_gdf.boundary.plot(ax=axs[2], color='red', linewidth=0.7)
    
    # Add text showing the CRS
    axs[2].text(0.01, 0.01, f"CRS: {income_data.rio.crs}", 
                transform=axs[2].transAxes, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'data_alignment_check.png'), dpi=300)
        print(f"Alignment check visualization saved to {os.path.join(output_dir, 'data_alignment_check.png')}")
    
    plt.show()

def analyze_vulnerable_areas(pop_data, flood_data, income_data, flood_threshold=1.0, income_threshold=15000, output_dir=None, flood_filename=None, nuts_file=None):
    """
    Identify and visualize areas where flooding exceeds a threshold depth AND income is below a threshold
    These represent especially vulnerable populations
    
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
    
    # Extract return period from flood filename if provided
    flood_title = f"Flood Vulnerability Analysis (Depth > {flood_threshold:.2f}m, Income < {income_threshold:,.0f} PPP)"
    if flood_filename:
        import re
        rp_match = re.search(r'RP(\d+)', flood_filename, re.IGNORECASE)
        if rp_match:
            rp_value = rp_match.group(1)
            flood_title = f"Flood Vulnerability Analysis (RP{rp_value}, Depth > {flood_threshold:.2f}m, Income < {income_threshold:,.0f} PPP)"
    
    # Check data alignment
    print(f"Dataset shapes - Population: {pop_data.shape}, Flood: {flood_data.shape}, Income: {income_data.shape}")
    
    # Create masks for our criteria
    flood_mask = flood_data > flood_threshold
    income_mask = income_data < income_threshold
    
    # Get counts of cells matching each criterion
    flood_cells = int(flood_mask.sum().values)
    income_cells = int(income_mask.sum().values)
    
    # Combined mask where both conditions are true
    combined_mask = flood_mask & income_mask
    combined_cells = int(combined_mask.sum().values)
    
    # Calculate statistics
    total_cells = flood_mask.size
    print(f"\nHotspot analysis results:")
    print(f"  Total cells in study area: {total_cells}")
    print(f"  Cells with flood depth > {flood_threshold}m: {flood_cells} ({flood_cells/total_cells*100:.2f}%)")
    print(f"  Cells with income < {income_threshold} PPP: {income_cells} ({income_cells/total_cells*100:.2f}%)")
    print(f"  Vulnerability hotspots (both conditions): {combined_cells} ({combined_cells/total_cells*100:.2f}%)")
    
    # If no cells match both criteria, try adjusting thresholds
    if combined_cells == 0:
        print("\nNo cells match both criteria. Attempting to adjust thresholds...")
        
        # Get flood depth percentiles
        flood_values = flood_data.values[~np.isnan(flood_data.values)]
        flood_75th = np.percentile(flood_values[flood_values > 0], 75)
        
        # Get income percentiles
        income_values = income_data.values[~np.isnan(income_data.values)]
        income_25th = np.percentile(income_values, 25)
        
        print(f"  Adjusted flood threshold: {flood_threshold} -> {flood_75th:.2f}m (75th percentile)")
        print(f"  Adjusted income threshold: {income_threshold} -> {income_25th:.2f} PPP (25th percentile)")
        
        # Apply new thresholds
        flood_mask = flood_data > flood_75th
        income_mask = income_data < income_25th
        combined_mask = flood_mask & income_mask
        combined_cells = int(combined_mask.sum().values)
        
        print(f"  With adjusted thresholds, found {combined_cells} vulnerability hotspots")
        
        # Update thresholds for title and results
        flood_threshold = flood_75th
        income_threshold = income_25th
        
        # Update the title with the new thresholds
        flood_title = f"Flood Vulnerability Analysis (Depth > {flood_threshold:.2f}m, Income < {income_threshold:,.0f} PPP)"
        if flood_filename:
            import re
            rp_match = re.search(r'RP(\d+)', flood_filename, re.IGNORECASE)
            if rp_match:
                rp_value = rp_match.group(1)
                flood_title = f"Flood Vulnerability Analysis (RP{rp_value}, Depth > {flood_threshold:.2f}m, Income < {income_threshold:,.0f} PPP)"
    
    # Calculate affected population in vulnerable areas
    vulnerable_pop = pop_data.where(combined_mask, 0)
    total_vulnerable_pop = float(vulnerable_pop.sum().values)
    total_pop = float(pop_data.sum().values)
    
    print(f"\nPopulation in vulnerability hotspots:")
    print(f"  Total population in hotspots: {total_vulnerable_pop:,.2f}")
    print(f"  Percentage of total population: {total_vulnerable_pop/total_pop*100:.2f}%")
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Population density
    pop_data.plot(ax=axs[0, 0], cmap='viridis', vmin=0, add_colorbar=True)
    axs[0, 0].set_title('Population Density')
    
    # Plot 2: Flood depth with threshold highlighted
    flood_data.plot(ax=axs[0, 1], cmap='Blues', add_colorbar=True)
    flood_highlight = flood_data.where(flood_mask, np.nan)
    flood_highlight.plot(ax=axs[0, 1], cmap='Reds', add_colorbar=False, alpha=0.5)
    axs[0, 1].set_title(f'Flood Depth (Highlighted > {flood_threshold:.2f}m)')
    
    # Plot 3: Income with threshold highlighted
    income_data.plot(ax=axs[1, 0], cmap='plasma', add_colorbar=True)
    income_highlight = income_data.where(income_mask, np.nan)
    income_highlight.plot(ax=axs[1, 0], cmap='Reds', add_colorbar=False, alpha=0.5)
    axs[1, 0].set_title(f'Income in PPP (Highlighted < {income_threshold:,.0f})')
    
    # Plot 4: Vulnerability hotspots
    vulnerable_pop.plot(ax=axs[1, 1], cmap='hot_r', add_colorbar=True)
    axs[1, 1].set_title('Vulnerability Hotspots\n(High Flood Risk + Low Income)')
    
    # Overlay NUTS regions on all plots
    if nuts_file and os.path.exists(nuts_file):
        try:
            nuts_gdf = gpd.read_file(nuts_file)
            for ax in axs.flatten():
                nuts_gdf.boundary.plot(ax=ax, color='darkgreen', linewidth=0.7)
        except Exception as e:
            print(f"Error adding NUTS overlay: {str(e)}")
    
    plt.suptitle(f"{flood_title}", fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        hotspot_path = os.path.join(output_dir, 'vulnerability_hotspots.png')
        plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
        print(f"Vulnerability hotspots visualization saved to {hotspot_path}")
    
    plt.show()
    
    return vulnerable_pop, total_vulnerable_pop

def setup_logging(output_dir):
    """
    Set up logging to capture both console output and log to a file
    """
    import sys
    from datetime import datetime
    
    log_file_path = os.path.join(output_dir, f"flood_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create a custom logger that outputs to both console and file
    class Logger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w", encoding="utf-8")
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Set up the logger
    sys.stdout = Logger(log_file_path)
    
    print(f"Logging to file: {log_file_path}")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 80)
    
    return log_file_path

if __name__ == "__main__":
    # Call main without any parameters since Vienna is now the default
    main()