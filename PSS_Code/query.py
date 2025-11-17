import os, time
from datetime import date
import argparse
import ast
import shutil
from IPython.display import Image
from astroquery.mast import Catalogs
from astroquery.mast import Observations
import pandas as pd
from astropy.table import QTable
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
from sh import gunzip

home_dir = '/xdisk/hamden/benyurek/env39'

### Input parameters ### 
r = 4.0 # Search radius in degrees for query
project_lis = ['AIS','NGS', 'MIS', 'DIS']  # list of GALEX surveys

# Swicth to the raw file directory
path_init = home_dir  + '/raw_files'
os.chdir( path_init )

today = date.today()
print("Today's date:", today)

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pair-list", nargs=4, type=str, help='four integers (BATCH, INDEX, RA, DEC)'
)
args = parser.parse_args()
print(args)

if args.pair_list:
    BATCH, INDEX, RA, DEC = map(int, args.pair_list)
    print("BATCH:", BATCH)
    print("INDEX:", INDEX)
    print("RA:", RA)
    print("DEC:", DEC)
    try:
        # Define the sky coordinates & directory name using the input information
        c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
        dir_name = "Batch_" + str(BATCH) + "/Index" + str(INDEX) + "_RA" + str(RA) + "_DEC" + str(DEC)
        
        # Create a new directory under raw_files
        os.makedirs(dir_name, exist_ok=True)
        
        # Switch to the new directory where files will be downloaded
        path_new = path_init + '/' + dir_name
        os.chdir( path_new )
 
    except FileNotFoundError:
        print("Error")
        

def coord_query(coord, radius, project):
    """
    Criteria based astropy query. 
    Criteria: coordinate, obs_collection ( = GALEX ), project ( AIS, MIS, DIS, NGS ), filter ( = FUV ), file

    Parameters
    ----------
    coord : astropy.coordinates.sky_coordinate.SkyCoord
        The desired coordinate, RA & DEC in degrees, (center) of query.
    radius : float
        The desired radius for query.
    project : list 
        The desired GALEX Science Surveys (e.g., ["AIS", "MIS"]).

    Returns
    -------
    DataProduct_final : Pandas DataFrame
        The table of files found via query.

    Description
    -----------
    This function performs criteria based astropy query. 
    It returns a catalog of files, allowing you to extract and download files
    in the next function (download_dp).
    
    """
    Observations.TIMEOUT = 1200
    Observations.timeout = 1200
    
    # Run GALEX coordinate query 
    result_table = Observations.query_criteria(coordinates=c, obs_collection='GALEX', radius=r*u.degree, project=project_lis, filters=["*FUV"])
    print("\n\nquery.py obsid/result table direct from MAST")
    print(result_table)
    print("\n\n")
    obsid = result_table['obsid']
    print(len(obsid), "obsids available")
    data_products = Observations.get_product_list(obsid)
    data_products =  data_products.to_pandas()
    
    # Only keep the -fd (FUV) files and toss the -nd (NUV) files;
    data_products = data_products[(data_products['productFilename'].str.contains('-fd'))]
    DataProduct_final = data_products[
    ((data_products['productFilename'].str.contains('_00' + r'\d{1}' + r'\d{1}' + '-' )) == False) &
    ((data_products['productFilename'].str.contains('_00' + r'\d{1}' + r'\d{1}' + '_' )) == False) &
    (data_products['productFilename'].str.endswith(('-cnt.fits.gz', '-rrhr.fits.gz', '-int.fits.gz', '-skybg.fits.gz', 'flags.fits.gz')))
    ]    
    print(int(len(DataProduct_final))/5, 'files total per type available to download')
    print(DataProduct_final['productFilename'])    

    return DataProduct_final

def download_dp(data_products):
    """
    Download files. 
    
    Parameters
    ----------
    data_products : Pandas DataFrame
        The table of files found via query ( output from coord_query )
        
    Returns
    -------
    None.

    Description
    -----------
    This function takes a catalog from query and downloads the files
    inside the folders "mastDownload/GALEX" 
    
    (e.g., raw_files/2024-07-08-RA195-DEC25/mastDownload/GALEX/6378621151228198912/AIS_219_sg17-fd-int.fits.gz ).
    
    """   

    data_products = QTable.from_pandas(data_products) # Convert data frame back to QTable
    manifest = Observations.download_products(data_products) # Download the files
    print("Installed: " , data_products['productFilename'])
    
    return None

def move_files(path_new):
    """
    organizes folders and files obtained from download_dp.
    
    Parameters
    ----------
    path_new : str
        The path to the working directory for this coordinate

    Returns
    -------
    None.

    Description
    -----------
    1. This function transfers the downloaded files located in
    directories " mastDownload/GALEX " to our default directory 
    and removes the empty parent directories afterward. 
    
    2. This function also gunzips the files ( fits.gz -> .fits ).
    
    Example
    -----------
    Before:
    raw_files/2024-07-08-RA195-DEC25/mastDownload/GALEX/6378621151228198912/AIS_219_sg17-fd-int.fits.gz
    
    After:
    raw_files/2024-07-08-RA195-DEC25/AIS_219_sg17-fd-int.fits
    
    """
    
    ext = '/mastDownload/GALEX'
    os.chdir( path_new + ext )
    
    current_folder = os.getcwd() 
    list_dir = os.listdir(current_folder)  # All the subdirectories
    
    # enumerate on list_dir to get the content of all the folders 
    # and store it in a dictionary
    content_dict = {}
    for index, val in enumerate(list_dir):
        path = os.path.join(current_folder, val)
        # os.listdir lists all everything (files/dir) in the specified directory
        content_dict [ list_dir[index] ] = os.listdir(path) # Assigns the files as keys in the dictionary 
    
    merge_folder_path = path_new # Distination of the files; default is the newly created directory 
   
    for sub_dir in content_dict: # loop through the sub folders under /mastDownload/GALEX
        for contents in content_dict[sub_dir]: # iterate through files 
            file_path = path_new + ext + "/" + sub_dir + "/" + contents # path of the file to move
            shutil.move(file_path, path_new) # move the file, file_path, to path_new

        os.rmdir(sub_dir) # remove the empty subdirectories
    
    # Removes empty folders: GALEX and mastDownload
    os.chdir( path_new + '/mastDownload' )
    os.rmdir('GALEX')
    os.chdir( path_new )
    os.rmdir('mastDownload') 
    
    merge_folder_path = path_new
    sourcepath = merge_folder_path
    sourcefiles = os.listdir(sourcepath)
    destinationpath = sourcepath # Moves all the files to the raw directory
    
    for file in sourcefiles:
        if file.endswith('.gz'):
            shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))
    os.chdir(destinationpath)
    
    for file in glob.glob("*.gz"):
        try:
            gunzip(file)
        except:
            print('already exists')
            
    return None


if __name__=="__main__":
    # Run query & extract data product
    DataProduct_final = coord_query(c, r, project_lis)
    download_dp( DataProduct_final ) # Download the files
    move_files( path_new ) # Organize the files

