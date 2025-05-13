#!/usr/bin/env python3
import os
import argparse
import xarray as xr
import numpy as np
import pyinterp
from scipy import interpolate

# For zarr collection we need dask configuration !!! (to be discussed)

import dask_jobqueue, dask
import warnings
warnings.filterwarnings('ignore')


dashboard_link = "https://jupyterhub.cnes.fr/user/{JUPYTERHUB_USER}/proxy/{port}/status"
dask.config.set({"distributed.dashboard.link": dashboard_link})

def start_slurm_cluster(nb_cores=8, workers=5,
                        local_directory='$TMPDIR',
                      account='account',
                      walltime='00:10:00',
                    log_directory='/work/scratch/data/aguedjh/slurm_out'):

    memory = "{}GB".format(nb_cores*8)

    cluster = dask_jobqueue.SLURMCluster(
    # Dask-worker specific keywords
    cores=nb_cores,                    # Number of cores per job
    memory=memory,              # Amount of memory per job
    processes=1,                # Number of Python processes to cut up each job
    local_directory=local_directory,  # Location to put temporary data if necessary
    account=account,
    walltime=walltime,
    interface='ib0',
    log_directory=log_directory,
    #job_extra_directives=['--qos="cpu_2019_80"','--partition="cpu2019_qual"']
    )

    # To precise the cluster size (dask workers number)
    cluster.scale(jobs=workers) # launch 4 jobs

    # To uncomment to adapt cluster size to the cluster's available ressources
#    cluster.adapt(maximum_jobs=16)

    return cluster

try : client.close()
except: pass
cluster = start_slurm_cluster(workers=10,account = "cnes_level2", walltime="00:20:00")

client = dask.distributed.Client(cluster, timeout=600)

#_________________________________________________________________


def read_netcdf_files(file1, file2, file3):
    """
    Reads three NetCDF files and returns the datasets.
    
    Parameters:
        file1 (str): Path to the first NetCDF file (Model).
        file2 (str): Path to the second NetCDF file (Mask for model).
        file3 (str): Path to the third NetCDF file (SWOT data).
        - interpolator (str) : the used interpolator
        
    Returns:
        tuple: A tuple containing three xarray datasets.
    """
    
    variable_name = input(f"Enter the variable to be interpolated name : " )
    
    latlon_input = input("Enter coordinate names separated by a comma (time, lat, lon): ").strip()
    
    try:
        time_name, lat_name, lon_name = [name.strip() for name in latlon_input.split(",")]
       
        
    except ValueError:
        sys.exit("Please enter exactly three dimension names separated by a comma:")
        
    
    interpolator = input(f"Enter the interpolator you wish to use, choose 'scipy_interpolator' or 'pyinterp_interpolator' : " )
    
    if interpolator in ('scipy_interpolator', 'pyinterp_interpolator'):
        pass
    else:
        sys.exit("Invalid interpolator! Please choose either 'scipy_interpolator' or 'pyinterp_interpolator'.")

    
    extension1 = os.path.splitext(file1)[1]
    extension2 = os.path.splitext(file2)[1]
    extension3 = os.path.splitext(file3)[1]
    
    # Open datasets depending on format
    ds1 = xr.open_dataset(file1) if extension1 == ".nc" else xr.open_zarr(file1, consolidated=True) # model
    
    if file1 == file2:
        ds2 = ds1         # save memory 
    else:
        ds2 = xr.open_dataset(file2) if extension2 == ".nc" else xr.open_zarr(file2) #mask model
    ds3 = xr.open_dataset(file3) if extension3 == ".nc" else xr.open_zarr(file3) #swot
    
    if ds3.longitude.max() > 180: # convert longitude to [- 180, 180], according to the input model coordinates
        ds3.coords['longitude'] = (ds3.coords['longitude'] + 180) % 360 - 180 
        
    
    if variable_name not in ds1:
        sys.exit(f"Variable '{variable_name}' not found in the model dataset.")
    
    ds1["ssh"] = ds1[variable_name]
    
    if len(ds1[variable_name].dims)==3:
        ds1 = ds1.isel({time_name: 0}) #(pourquoi prendre uniquement un seul pas de temps, il faudra généraliser pour la dimension temps)
        
    if variable_name!="ssh":
        del ds1[variable_name]
    
    # ds3["ssh"] = ds3["ssha"]  + ds3["mdt"] 
    
    return ds1, ds2, ds3 , interpolator, lat_name, lon_name



import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
import pyinterp

def open_model_data(ds_var, ds_coords, latitude_array, longitude_array, var, interpolator, lat_name="latitude", lon_name="longitude"):
    """
    Creates an interpolator from a model dataset containing the ssh variable.
    The spatial coordinates (latitude and longitude) are provided as 2D variables in a separate dataset.
    Parameters:
    - ds_var (xarray.Dataset): Dataset containing the model ssh to interpolate.
    - ds_coords (xarray.Dataset): Dataset containing latitude and longitude as 2D variables.
    - latitude_array, longitude_array (xarray.Dataset): Satellite (SWOT) grid to restrict the area around the target swath
    - var (str): Name of the variable to interpolate.
    - lat_name (str, optional): Name of the latitude variable in ds_coords (default: "latitude").
    - lon_name (str, optional): Name of the longitude variable in ds_coords (default: "longitude").
    - interpolator (str): the used interpolator
    Returns:
    - finterp (LinearNDInterpolator): Interpolator for irregular 2D (latitude, longitude) grid.
    """
    # Check if the variable exists in ds_var
    if var not in ds_var:
        raise ValueError(f"Variable '{var}' is not present in the provided dataset.")

    # Restrict the area around the swath
    lon_min, lon_max, lat_min, lat_max = np.min(longitude_array), np.max(longitude_array), np.min(latitude_array), np.max(latitude_array)
    print("The domain limits: ", [lon_min.values, lon_max.values, lat_min.values, lat_max.values])

    condition = ((ds_coords[lon_name] <= lon_max) & (ds_coords[lon_name] >= lon_min) & (ds_coords[lat_name] <= lat_max) & (ds_coords[lat_name] >= lat_min))
    print("ok")
    ds_var = ds_var.where(condition, drop=True)
    ds_coords = ds_coords.where(condition, drop=True)

    # Extract latitude and longitude from ds_coords (as 2D arrays)
    try:
        lat_values = ds_coords[lat_name].compute().values  # Shape (x, y)
        lon_values = ds_coords[lon_name].compute().values  # Shape (x, y)
    except KeyError:
        raise ValueError(f"Could not find '{lat_name}' or '{lon_name}' in the coordinates dataset.")

    # Extract variable values
    var_values = ds_var[var].compute().values
    print(var_values.shape)
    print(lon_values.shape)
    print(lat_values.shape)

    # Ensure the variable has the correct dimensions (latitude, longitude)
    if var_values.ndim == 3:  # If an extra time dimension exists
        var_values = var_values[0]  # Take only the first time step

    mask = np.isfinite(var_values) & np.isfinite(lon_values) & np.isfinite(lat_values)

    # Flatten the 2D grid into 1D arrays
    lat_flat = lat_values[mask]
    lon_flat = lon_values[mask]
    var_flat = var_values[mask]

    if interpolator == "scipy_interpolator":
        # Create a scattered data interpolator
        finterp = LinearNDInterpolator(
            list(zip(lat_flat, lon_flat)),
            var_flat,
            fill_value=np.nan
        )
    elif interpolator == "pyinterp_interpolator":
        # Create an interpolator
        points = np.vstack((lon_flat, lat_flat)).T
        finterp = pyinterp.RTree()
        finterp.packing(points, var_flat)
    else:
        raise ValueError(f"Unknown interpolator: {interpolator}")

    return finterp



def interp_satellite(latitude_array, longitude_array, interp, interpolator, var):
    """
    Interpolates the modeled SSH at satellite observation points (wide swath only).

    Parameters:
    - latitude_array (xarray.DataArray or np.array): Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    - longitude_array (xarray.DataArray or np.array): Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    - interp (scipy.interpolate.LinearNDInterpolator): Interpolator from `open_model_data` 
    - interpolator (str) : the used interpolator
     - var (str): Name of the ssh variable (e.g., "ssh_debug")

    Returns:
    - ds (xarray.Dataset): Dataset of interpolated SSH values, structured for wide swath data.
    """

    # Ensure latitude and longitude are NumPy arrays before flattening
    latitude_array = np.asarray(latitude_array)
    longitude_array = np.asarray(longitude_array)

    # Flatten the satellite lat/lon arrays to feed into the interpolator
    points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))

    # Apply the interpolator to get SSH values at satellite positions
    
    if interpolator == "scipy_interpolator":
        print("Interpolation in progress ...")
        ssh_interp = interp(points).reshape(latitude_array.shape)
        print('interplation done')
    elif interpolator == "pyinterp_interpolator":
        ssh_interp = interp.inverse_distance_weighting(
            points,                        
            k=8,    # We are looking for at most 8 neighbours
            num_threads=0 # parallel computing                         
            p=2  #The power to be used by the interpolator inverse_distance_weighting.
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

    ds["ssh"] = ds["ssh"].where(ds["ssh"] != 0.0, np.nan)
    
    return ds


def save_netcdf(result, output_file):
    """
    Save the resulting dataset to a NetCDF file.
    """
    result.to_netcdf(output_file)

def main():
    parser = argparse.ArgumentParser(description="Processing workflow")
    parser.add_argument("file1", help="Path of the model NetCDF file")
    parser.add_argument("file2", help="Path of the mask NetCDF file")
    parser.add_argument("file3", help="Path of the SWOT NetCDF file")
    parser.add_argument("output", help="Path of the output nc file")

    args = parser.parse_args()

    # read files NetCDF
    ds_model, ds_mask, ds_swot, interpolator, lat_name, lon_name = read_netcdf_files(args.file1, args.file2, args.file3)
    print("All dataset are opened ...")
    # Analyse
    finterp  = open_model_data(ds_model, ds_mask, ds_swot.latitude, ds_swot.longitude, "ssh", interpolator, lat_name, lon_name)
    print('OK')
    output_ds = interp_satellite(ds_swot.latitude, ds_swot.longitude, finterp, interpolator, var="ssh")
    
    # Sauvegarder le fichier
    save_netcdf(output_ds, args.output)

if __name__ == "__main__":
    main()
