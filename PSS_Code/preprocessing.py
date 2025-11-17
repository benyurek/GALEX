import os, time
from multiprocessing import Pool
from datetime import date
import argparse
import ast
import sys
import shutil
from IPython.display import Image
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
from glob import glob
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.utils.data import get_pkg_data_filename
from photutils.segmentation import make_2dgaussian_kernel

from reproject import reproject_interp

home_dir = "/xdisk/hamden/benyurek/env39"

### Input parameters ### 
radius = int(1400)
area_circle = np.pi * (radius ** 2)
artifacts_cutoff = 20
edge_thickness = 500

num_cpus = int(os.environ["SLURM_CPUS_ON_NODE"])
print("Running a pool with %s workers" % num_cpus)

# Swicth to the raw file directory
path_init = home_dir + "/raw_files"
os.chdir(path_init)

today = date.today()

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pair-list", nargs=4, type=str, help="four integers (BATCH, INDEX, RA, DEC)"
)
args = parser.parse_args()

if args.pair_list:
    # Split args into a directory and a filename
    BATCH, INDEX, RA, DEC = map(int, args.pair_list)
    print("BATCH:", BATCH)
    print("INDEX:", INDEX)
    print("RA:", RA)
    print("DEC:", DEC)
    try:
        # Define the directory name using the input information
        dir_name = "Batch_" + str(BATCH) + "/Index" + str(INDEX) + "_RA" + str(RA) + "_DEC" + str(DEC)

        # Switch to the new directory
        path_new = str(os.getcwd()) + "/" + dir_name
        os.chdir(path_new)

        print("New location: ", path_new)
    except FileNotFoundError:
        print("Error")

#os.chdir("/xdisk/hamden/hina0830/venv39/raw_files/NGC55")
print("New location: ", os.getcwd())

def make_mask(data):  
    """
    Creates a mask for an image 
    
    Parameters
    ----------
    data : numpy array (float 64)
        The image to mask
        
    Returns
    -------
    mask : numpy array (float 64)
        The generated mask

    Description
    -----------
    This function creates an array that 
    masks pixels outside the circle ( radius = 1400 ).
    The size of the mask and the image 
    should be the same.
    
    """  
    
    # Create a grid of coordinates
    x, y = np.arange(0, int(len(data))), np.arange(0, int(len(data[0])))
    x_grid, y_grid = np.meshgrid(x, y)
    x_cent, y_cent = int(round(len(data) / 2)), int(round(len(data[0]) / 2))

    # Calculate distances for all pixels at once
    distances = np.sqrt((x_cent - x_grid) ** 2 + (x_cent - y_grid) ** 2)

    # Create a mask for pixels outside the circle
    mask = distances > radius
    
    return mask


def process_file(filename):
    
    """
    Preprocessing for cnt, rrhr, ant int files.
    
    Parameters
    ----------
    filename : str
        The radius of the bounds of the circe 
        
    Returns
    -------
    data : numpy array (float 64)
        The final image (edited)

    Description
    -----------
    This function preprocesses the following types of files accordingly 
    
    cnt:
        Replace the pixels outside of the circle with Numpy Nans
        ( r = 1400 [pix] )
    
    rrhr:
        Replace negative values with Numpy Nans.
    
    int:
        Replace negative values with Numpy Nans.
        Apply a Gaussian smoothing filter.
        ( kernel size: FWHM = 7.0 [pix])   
    
    All:
        Set the bounds of the data to be the bounds of the circle
        aka cut 500 blank pixels from each side of the image 
        ( 3840 [pix] x 3840 [pix] - > 2840 [pix]  x 2840 [pix] )
    
    """   
    
    print("Process_file at: ", filename)
    # Open the file and convert the data type
    hdu = fits.open(filename)
    data = (hdu[0].data).astype("float64")

    if "rrhr" in filename:
        print("rrhr file: ", filename)
        data[data < 0] = np.nan

    elif "int" in filename:
        print("int file: ", filename)
        data[data < 0] = np.nan
        
        # Apply Gaussian filter to intensity file
        kernel = make_2dgaussian_kernel(7.0, size=21)  # FWHM = 7
        data = convolve(data, kernel)
    
    # Replace the outer pixels with nans
    mask = make_mask(data)
    data[mask] = np.nan
    
    # Trim 500 blank pixels from each side
    data = data[edge_thickness:len(data)-edge_thickness, edge_thickness:len(data[0])-edge_thickness]

    # Save the file
    hdu[0].data = data
    hdu.writeto(filename.removesuffix(".fits") + "_preprocessed.fits", overwrite=True)
    print(f"File saved for {filename}")

    return data

def preprocess(file_tuple):
    """
    Assesses the quality of an image based on its flags file. 
    Bad images won't be preprocessed. 
    
    Parameters
    ----------
    file_tuple : tuple
        Set of filenames for the following files:
        cnt, rrhr, int, skybg, flags
        
    Returns
    -------
    None. 
    
    Description
    -----------
    1. This function reprojects and resizes flags files
    to be the same size as other files 
    ( resizing from 480 x 480 - > 3840 x 3840 )
    
    2. This function calculates the fraction of bad pixels 
    ( >= 128 ) in flags file 
    
    Formula : 
    100 * ( number of pixels above 127 ) / ( pi * r ^ 2, where r = 1400 pix ) 
    
    2. If less than 20%, this function preprocesses files in the tuple.
    
    cnt, rrhr, int:
        Run process_file function for preprocessing.
        
    flags ( Reprojected ):
        Set the bounds of the data to be the bounds of the circle.
        
    skybg :
        Set the bounds of the data to be the bounds of the circle
    
    """   
    
    
    print("Processing tuple ", file_tuple)
    cnt_file, rrhr_file, int_file, skybg_file, flags_file = file_tuple

    cnt_data, rrhr_data, int_data, skybg_data, flags_data = (
        fits.getdata(cnt_file).astype("float64"),
        fits.getdata(rrhr_file).astype("float64"),
        fits.getdata(int_file).astype("float64"),
        fits.getdata(skybg_file).astype("float64"),
        fits.getdata(flags_file).astype("float64"))
    
    hdu1 = fits.open(cnt_file)[0]
    hdu2 = fits.open(flags_file)[0] # flag, 480 by 480
    
    flags_data, footprint = reproject_interp(hdu2, hdu1.header, order = "nearest-neighbor") 
    flag_largeval = np.where(flags_data < 128, 0 , flags_data)
    nonzero_pix = np.count_nonzero(flag_largeval)
    artifacts_frac = (nonzero_pix / area_circle) * 100
    
    print("frac: " + (f"{artifacts_frac:.2f}") + "%")
    
    # Preprocess the files only if fraction of the flagged pixels is above our cutoff value 
    if artifacts_frac < artifacts_cutoff:
        prerocessed_cnt = process_file(cnt_file)
        prerocessed_rrhr = process_file(rrhr_file)
        prerocessed_int = process_file(int_file)
        
        flags_data = flags_data[edge_thickness:len(flags_data)-edge_thickness, edge_thickness:len(flags_data[0])-edge_thickness]
        
        # Save reprojected flags
        fits.writeto(flags_file.removesuffix('.fits') + '_wcs.fits', flags_data, hdu1.header, overwrite=True)
        print(flags_file.removesuffix('.fits') + '_wcs.fits', " reprojected and saved. ")
        
        hdu = fits.open(skybg_file)
        print("preprocessing.py, before trimming. ", np.shape(skybg_data))
        skybg_data = skybg_data[edge_thickness:len(skybg_data)-edge_thickness, edge_thickness:len(skybg_data[0])-edge_thickness]
        print("preprocessing.py, after trimming. ", np.shape(skybg_data))
        
        hdu[0].data = skybg_data
        hdu.writeto(skybg_file.removesuffix(".fits") + "_preprocessed.fits", overwrite=True)
        
    else:
        print(cnt_file.removesuffix('.fits') + " discarded at " + (f"{artifacts_frac:.2f}") + "%")
        
    return None

if __name__ == "__main__":
    with Pool(num_cpus) as p:
        fn_cnt, fn_rrhr, fn_int, fn_skybg, fn_flags = (
            sorted(glob("*-cnt.fits")),
            sorted(glob("*rrhr.fits")),
            sorted(glob("*int.fits")),
            sorted(glob("*skybg.fits")),
            sorted(glob("*flags.fits")),
        )
        
        mapped_new = list(zip(fn_cnt, fn_rrhr, fn_int, fn_skybg, fn_flags))
        print(mapped_new) 
        if len(fn_cnt) == len(fn_rrhr) == len(fn_int) == len(fn_skybg) == len(fn_flags):
            print(len(mapped_new), "files total.")
            output = p.map(preprocess, mapped_new)

        
