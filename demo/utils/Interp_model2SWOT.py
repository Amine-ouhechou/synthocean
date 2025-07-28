#!/usr/bin/env python3
import argparse
import xarray as xr
import numpy as np
import pyinterp
import os
from scipy import interpolate
from inpoly.inpoly2 import inpoly2



def read_netcdf_files(file1, file2, file3, time_name = "time_counter", time_step=0):
    """
    Reads three NetCDF files and returns the datasets.
    
    Parameters:
        file1 (str): Path to the first NetCDF file (Model).
        file2 (str): Path to the second NetCDF file (Mask for model).
        file3 (str): Path to the third NetCDF file (SWOT data).
        
    Returns:
        tuple: A tuple containing three xarray datasets.
    """
    
    # file extensions ...
    extension1 = os.path.splitext(file1)[1]
    extension2 = os.path.splitext(file2)[1]
    
    # Open datasets depending on format
    ds1 = xr.open_dataset(file1) if extension1 == ".nc" else xr.open_zarr(file1) # model
    
    if file1 == file2:
        ds2 = ds1         # save memory if data and mask are in the same file
    else:
        ds2 = xr.open_dataset(file2) if extension2 == ".nc" else xr.open_zarr(file2) #mask model 
        
    ds3 = xr.open_dataset(file3)

    # Select first time step if a time dimension exists (SWOT only needs 2D fields)
    ds1 = ds1.isel({time_name: int(time_step)}) if time_name in ds1.dims else ds1  # indeed, if the dataset has time dimension. For the moment only one time step is needed
    
    return ds1, ds2, ds3


def select_area(lat, lon, var, latitude_array, longitude_array):
    """
    Selects the subset of model data (lon, lat, var) located within a swath defined by satellite coordinates.
    
    Parameters:
    - lat, lon: 2D arrays of model grid coordinates.
    - var: 2D array of the variable to interpolate.
    - latitude_array, longitude_array: 2D arrays of SWOT swath coordinates.
    
    Returns:
    - lon_in, lat_in, var_in: 1D arrays of the filtered model data inside the swath.
    """
    
    # Remove invalid points (0, 0)
    mask_valid = ~((lon.flatten() == 0) & (lat.flatten() == 0))
    lon_clean = lon.flatten()[mask_valid]
    lat_clean = lat.flatten()[mask_valid]
    var_clean = var.flatten()[mask_valid]
    
    points = np.column_stack((lon_clean, lat_clean))
        
    # Adjust longitudes to [-180, 180]
    X = xr.where(longitude_array <= 180, longitude_array, longitude_array - 360)
    Y = latitude_array
    
    # Polygon with margin
    dy = Y.values[-1,0] - Y.values[0,0]
    
    k = 2 if dy > 0 else -2
    k1 = abs(k)
    
    xx = np.concatenate([
        X.isel(num_lines=0).values,
        X.isel(num_pixels=-1).values+k,
        X.isel(num_lines=-1).values[::-1],
        X.isel(num_pixels=0).values[::-1]-k
    ])

    yy = np.concatenate([
        Y.isel(num_lines=0).values-k,
        Y.isel(num_pixels=-1).values-k1,
        Y.isel(num_lines=-1).values[::-1]+k,
        Y.isel(num_pixels=0).values[::-1]+k1
    ])
    
    
    
    
    polygon = np.column_stack((xx, yy))  
    inside, on_edge = inpoly2(points, polygon)
    mask = inside | on_edge

    lon_in = lon_clean[mask]
    lat_in = lat_clean[mask]
    var_in = var_clean[mask]
     
    return lon_in, lat_in, var_in



def open_model_data(ds_var, ds_coords,interpolator, var, latitude_array, longitude_array, lat_name="latitude", lon_name="longitude",time_step=0):
    """
    Creates an interpolator from a model dataset containing the ssh variable.
    The spatial coordinates (latitude and longitude) are provided as 2D variables in a separate dataset.

    Parameters:
    - ds_var (xarray.Dataset): Dataset containing the model ssh to interpolate.
    - ds_coords (xarray.Dataset): Dataset containing latitude and longitude as 2D variables.
    - var (str): Name of the variable to interpolate.
    - latitude_array (xarray.DataArray or np.array): Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    - longitude_array (xarray.DataArray or np.array): Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    - lat_name (str, optional): Name of the latitude variable in ds_coords (default: "latitude").
    - lon_name (str, optional): Name of the longitude variable in ds_coords (default: "longitude").
    
    
    Returns:
    - finterp (LinearNDInterpolator): Interpolator for irregular 2D (latitude, longitude) grid.
    """
    
    # Check if the variable exists in ds_var
    if var not in ds_var:
        raise ValueError(f"Variable '{var}' is not present in the provided dataset.")

    # Extract latitude and longitude from ds_coords (as 2D arrays)
    try:
        lat_values = ds_coords[lat_name].values  # Shape (x, y)
        lon_values = ds_coords[lon_name].values  # Shape: x, y)
    except KeyError:
        raise ValueError(f"Could not find '{lat_name}' or '{lon_name}' in the coordinates dataset.")

    # Extract variable values
    var_values = ds_var[var].values  
    
    # Ensure the variable has the correct dimensions (latitude, longitude)
    if var_values.ndim == 3:  # If an extra time dimension exists
        var_values = var_values[time_step]  # Take only the first time step

    lon_in, lat_in, var_in = select_area(lon=lon_values,
                                         lat=lat_values,
                                         var=var_values,
                                         latitude_array=latitude_array,
                                         longitude_array=longitude_array
                                        )
    
    
    
    # Flatten the 2D grid into 1D arrays
    lat_flat = lat_in # lat_values.flatten()
    lon_flat = lon_in # lon_values.flatten()
    var_flat = var_in # var_values.flatten()
    
    if np.size(var_flat)!=0: #Checking is not empty
        
        # Create a scattered data interpolator  !!!!!!! car c'est irregular 2D grids)
        if interpolator == "scipy_interpolator":
            
            finterp = interpolate.LinearNDInterpolator(
               list(zip(lat_flat, lon_flat)),
                var_flat,
                fill_value=np.nan
            )
        
        elif interpolator == "pyinterp_interpolator":

            points = np.column_stack((lon_flat, lat_flat))
            finterp = pyinterp.RTree()
            finterp.packing(points, var_flat)
        else:
            raise ValueError(f"Unknown interpolator: {interpolator}")
    

        return finterp
    else:
        return 0



def interp_satellite(latitude_array, longitude_array,cross_dist,quality_flag, interpolator, interp, var):
    """
    Interpolates the modeled SSH at satellite observation points (wide swath only).

    Parameters:
    - latitude_array (xarray.DataArray or np.array): Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    - longitude_array (xarray.DataArray or np.array): Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    - interp (scipy.interpolate.LinearNDInterpolator): Interpolator from `open_model_data`
    - var (str): Name of the ssh variable (e.g., "ssh_debug")
    - cross_dist and quality_flag are used to build the mask

    Returns:
    - ds (xarray.Dataset): Dataset of interpolated SSH values, structured for wide swath data.
    """
    
    # Ensure latitude and longitude are NumPy arrays before flattening
    longitude_array = xr.where(longitude_array>180 , longitude_array-360, longitude_array)  # swot longitude conversion from 0/360 to 180/-180
    
    latitude_array = np.asarray(latitude_array)
    longitude_array = np.asarray(longitude_array)

    # Flatten the satellite lat/lon arrays to feed into the interpolator
    points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))

    # Apply the interpolator to get SSH values at satellite positions
    
    print("Interpolation in progress ...")
    
    if interpolator == "scipy_interpolator":
        # Flatten the satellite lat/lon arrays to feed into the interpolator
        points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))       
        ssh_interp = interp(points).reshape(latitude_array.shape)
         
    elif interpolator == "pyinterp_interpolator":
        points = np.column_stack((longitude_array.flatten(), latitude_array.flatten()))
        ssh_interp = interp.inverse_distance_weighting(
            coordinates=points,                                  
            k=5,    # We are looking for at most ' neighbours
            num_threads=0, # parallel computing                         
            p=2,  #The power to be used by the interpolator inverse_distance_weighting.
            within=True
            )[0].reshape(latitude_array.shape) 
               
    # Rename variable if needed
    if var != "ssh":
        var = "ssh"

    # Create an xarray dataset for wide swath data
    ds = xr.Dataset({
        var: (["num_lines", "num_pixels"], ssh_interp)
    }, coords={
        "latitude": (["num_lines", "num_pixels"], latitude_array),
        "longitude": (["num_lines", "num_pixels"], longitude_array)
    })
    
    
    # removing data from where swot does not have any (inter-swath and periphery areas) and from the continent (eventually) [see quality flag in the documentation]
    # Only values between 10 and 60 km to the nadir are considered as valid data. https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/hdbk_duacs_SWOT_L3.pdf                                                                                               
    mask = xr.where((abs(cross_dist)<=60.0) & (abs(cross_dist)>=10.0) & (quality_flag<101) ,cross_dist,np.nan)                                                                             
    ds["ssh"] = ds["ssh"].where(~np.isnan(mask))
    
    # swot longitude conversion back to 0/360
    ds.coords['longitude']= xr.where(ds.longitude<0 , ds.longitude+360, ds.longitude)  
    return ds


def save_netcdf(result, output_file):
    """
    Save the resulting dataset to a NetCDF file.
    """
    result.to_netcdf(output_file)


ALLOWED_INTERPOLATORS = ['scipy_interpolator', 'pyinterp_interpolator']

def validate_interpolator(value):
    """function to validate the interpolator argument."""
    if value.lower() not in ALLOWED_INTERPOLATORS:
        raise argparse.ArgumentTypeError(
            f"Invalid interpolator method: '{value}'. "
            f"Allowed methods are: {', '.join(ALLOWED_INTERPOLATORS)}"
        )
    return value.lower() 


def main():
    parser = argparse.ArgumentParser(description="A tool to process and interpolate ocean model to SWOT satellite swath",
        formatter_class=argparse.RawTextHelpFormatter # for multiline help messages
                                    )
    parser.add_argument("-m", "--model_file", required=True, help="Path of the input model file (NETCDF or zarr)")
    parser.add_argument("-k", "--mask_file", required=True, help="Path of the input mask file (NETCDF or zarr)")
    parser.add_argument("-s", "--swot_file", required=True, help="Path of the input SWOT file (NETCDF or zarr)")
    parser.add_argument("-o", "--output_file", required=True, help="Path of the output NETCDF file")

    parser.add_argument("-i", "--interpolator", type=validate_interpolator, required=True,
                        help=f"The interpolation method to use. "
                             f"Supported methods are: {', '.join(ALLOWED_INTERPOLATORS)}.\n"
                             f"Choose one from the list based on your needs.")
    
    # Arguments for model's (lat/lon) variable names
    parser.add_argument("--model_lat_var", default="latitude",
                        help="Name of the latitude variable in the model NetCDF file (default: latitude).")
    parser.add_argument("--model_lon_var", default="longitude",
                        help="Name of the longitude variable in the model NetCDF file (default: longitude).")
    parser.add_argument("--model_time_var", default="time_counter",
                        help="Name of the time variable in the model NetCDF file (default: time_counter).")
    parser.add_argument("--model_ssh_var", default="ssh", 
                        help="Name of the variable in the model file to interpolate (e.g., 'ssh', 'sossheig', ...)")
    parser.add_argument("--model_timestep_index", default="ssh", 
                        help="Time step index in the model file to interpolate (by default is the first time-step)")
    args = parser.parse_args()

    # Pre-check existence of input files
    for f in [args.model_file, args.mask_file, args.swot_file]:
        if not os.path.exists(f):
            parser.error(f"Error: Input file not found: {f}")
        if not os.path.isfile(f):
            parser.error(f"Error: Path is not a file: {f}")

    # read files
    interpolator = args.interpolator
    print(f"Processing with interpolator: {interpolator}")
    ds_model, ds_mask, ds_swot = read_netcdf_files(args.model_file, args.mask_file, args.swot_file, args.model_time_var,args.model_timestep_index)
    # Analyse
    finterp = open_model_data(ds_model, ds_mask,interpolator, args.model_ssh_var, ds_swot.latitude, ds_swot.longitude, args.model_lat_var, args.model_lon_var,args.model_timestep_index)
    if finterp !=0: # Checking finterp is not empty
        output_ds = interp_satellite(ds_swot.latitude, ds_swot.longitude, ds_swot.cross_track_distance, ds_swot.quality_flag, interpolator, finterp, var="ssh")
        # Sauvegarder le fichier
        save_netcdf(output_ds, args.output_file)
        print("Script finished successfully")
    else:
        print("The model has no information for the SWOT path")

if __name__ == "__main__":
    main()

 