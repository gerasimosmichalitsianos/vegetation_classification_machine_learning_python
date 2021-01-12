import os
import sys
import numpy as np
import warnings as warn
import subprocess
from Misc import RunProcess
from distutils.spawn import find_executable 
from Misc import WriteGeotiff
from osgeo import osr,gdal
from scipy.ndimage.filters import gaussian_filter

def CreateImageGaussianFiltered(InputArray):
  '''function GaussianFilter( arr ):
  This function calls scipy.ndimage's gaussian_filter 
  function. In particular, an array or 2D NumPy array 
  is passed-into this function, and the gaussian_filter 
  function is called to create a "blurred" or filtered 
  2D NumPy array. This array is returned. 

  This function is called by CreateImageryBackground() 
  several times below to craate "background" imagery for 
  several satellite band-combinations for use in 
  classification of vegetation/forest in satellite imagery.
  Some of these metrics include background (moving average)
  for NDVI, Blue, Green, Red, and the Panchromatic image.

  Args:
    InputArray (numpy.ndarray): Input NumPy array. Should be two-dimensional.
  Returns: 
    np.ndarray: Output filtered 2D NumPy array (float).
  '''
  return np.array(gaussian_filter(InputArray,sigma=5,mode='nearest'), dtype=np.float32)

def CreateImageRGB(RedGeotiff,GreenGeotiff,BlueGeotiff,OutputDirectory):
  '''function CreateImageRGB( RedGeotiff,GreenGeotiff,BlueGeotiff,OutputDirectory):
  This function takes in filename strings for 
  Geotiffs that contain pixel data for the Red, 
  Green, and Blue channels for a set of satellite 
  imagery. It calls the gdal_merge.py command-line tool 
  (That comes with a standard GDAL installation) to 
  merge these 3 Geotiffs into a single RGB mosaic. 
  This 3-band RGB Geotiff filename string is returned.

  Args:
    RedGeotiff (str): Geotiff filename for Geotiff containing "Red" channel. 
    GreenGeotiff (str): Geotiff filename for Geotiff containing "Green" channel. 
    BlueGeotiff (str): Geotiff filename for Geotiff containing "Blue" channel. 
    OutputDirectory (str): Geotiff string for Geotiff containing "Red" channel. 
  Returns:
    str: Name of RGB-combination 3-band Geotiff.
  '''

  # get full path of gdal_merge.py ... if it is not installed,
  # then we exit.
  # ----------------------------------------------------------
  GDAL_Merge_Path  = find_executable('gdal_merge.py')
  if GDAL_Merge_Path is None:
    print('  \n   GDAL gdal_merge.py script was not found. Exiting ... ')
    return None
  
  # Create output filename for RGB Geotiff.
  # If the file already exists, then we remove it.
  # ----------------------------------------------
  OutnameRGB = os.path.join( OutputDirectory , 'RGB.jp2' )
  if os.path.isfile(OutnameRGB): os.remove(OutnameRGB)

  # Create command to create RGB mosaic (that will be run
  # by the shell). Then we call misc.py's RunProcess command
  # to create the file. We return output 
  # RGB filename Geotiff string.
  # ----------------------------------------------
  GDAL_Merge_Command = ' '.join([GDAL_Merge_Path,' -q -separate -o ',
    OutnameRGB, RedGeotiff,GreenGeotiff, BlueGeotiff ])
  RunProcess( GDAL_Merge_Command )
  return OutnameRGB

def CreateImageryBackground( FileArrayPointers,OutputDirectory,ReferenceDataset ): 
  '''
  function CreateImageryBackground( FileArrayPointers,OutputDirectory,ReferenceDataset ):
   This function creates "background" or gaussian-filtered imagery (Geotiffs) 
   for the following band(s) or band combinations: 
    (1) Red 
    (2) Green 
    (3) Blue
    (4) NIR
    (5) Panchromatic (Red+Green+Blue+NIR/4.0)
    (6) NDVI (Normalized Difference Vegetation Index)
   To this end, this function calls CreateImageGaussianFiltered() above 
   to compute this "blurred" imagery for these bands.
  Args: 
    FileArrayPointers (list): List of NumPy memory-map objects for bands listed above.
    OutputDirecotry (str): Output directory.
    ReferenceDataset (osgeo.gdal.Dataset): GDAL dataset for reference.
  Returns: 
    dict: Python dictionary{} holding filenames for "background" imagery.
  '''

  # Get File pointers (memory map objects) for NumPy arrays
  # for the following:
  #   (1) Red
  #   (2) Green
  #   (3) Blue 
  #   (4) NIR 
  #   (5) Panchromatic 
  #   (6) NDVI 
  # --------------------------------------------------------
  FilePointerRed   = FileArrayPointers[0]
  FilePointerGreen = FileArrayPointers[1]
  FilePointerBlue  = FileArrayPointers[2]
  FilePointerNIR   = FileArrayPointers[3]
  FilePointerPan   = FileArrayPointers[4]
  FilePointerNDVI  = FileArrayPointers[5]

  # initialize a dictionary to hold keys and 
  # corresponding filenames for the following 
  # imagery (for vegetation/non-vegetation 
  # classification:
  #   (1) Background (Gaussian-filtered) Red Image File
  #   (2) Background (Gaussian-filtered) Green Image File
  #   (3) Background (Gaussian-filtered) Blue Image File
  #   (4) Background (Gaussian-filtered) NIR File
  #   (5) Background (Gaussian-filtered) Panchromatic Image File
  #   (6) Background (Gaussian-filtered) NDVI Image File
  # ------------------------------------------------------------
  BackgroundImageryFilenameDict={}

  # Create output filename strings for "background" 
  # gaussian-filtered imagery
  # ------------------------------------------------
  BackgroundFileNameRed   = os.path.join(
    OutputDirectory,'BackgroundRed.tif') 
  BackgroundFileNameGreen = os.path.join(
    OutputDirectory,'BackgroundGreen.tif') 
  BackgroundFileNameBlue  = os.path.join(
    OutputDirectory,'BackgroundBlue.tif') 
  BackgroundFileNameNIR   = os.path.join(
    OutputDirectory,'BackgroundNIR.tif') 
  BackgroundFileNamePan   = os.path.join(
    OutputDirectory,'BackgroundPan.tif') 
  BackgroundFileNameNDVI   = os.path.join(
    OutputDirectory,'BackgroundNDVI.tif') 

  # If any "Background" or Guasian-blurred imagery
  # already exists ... then remove it.
  # -----------------------------------------------
  if os.path.isfile( BackgroundFileNameRed ) : os.remove( BackgroundFileNameRed )
  if os.path.isfile( BackgroundFileNameGreen ): os.remove( BackgroundFileNameGreen )
  if os.path.isfile( BackgroundFileNameBlue  ): os.remove( BackgroundFileNameBlue )
  if os.path.isfile( BackgroundFileNameNIR   ): os.remove( BackgroundFileNameNIR )
  if os.path.isfile( BackgroundFileNamePan   ): os.remove( BackgroundFileNamePan )
  if os.path.isfile( BackgroundFileNameNDVI  ): os.remove( BackgroundFileNameNDVI )

  # Write Geotiffs for each of the 6 "background" image files
  # named above 
  # ---------------------------------------------------------
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNameRed   , CreateImageGaussianFiltered(FilePointerRed)   )
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNameGreen , CreateImageGaussianFiltered(FilePointerGreen) )
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNameBlue  , CreateImageGaussianFiltered(FilePointerBlue)  )
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNameNIR   , CreateImageGaussianFiltered(FilePointerNIR)   )
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNamePan   , CreateImageGaussianFiltered(FilePointerPan)   )
  WriteGeotiff( ReferenceDataset, 
    BackgroundFileNameNDVI  , CreateImageGaussianFiltered(FilePointerNDVI)  )
 
  # Append the output dict{} holding the "background" filenames
  # -----------------------------------------------------------
  BackgroundImageryFilenameDict['bg_red']   = BackgroundFileNameRed 
  BackgroundImageryFilenameDict['bg_green'] = BackgroundFileNameGreen 
  BackgroundImageryFilenameDict['bg_blue']  = BackgroundFileNameBlue 
  BackgroundImageryFilenameDict['bg_nir']   = BackgroundFileNameNIR
  BackgroundImageryFilenameDict['bg_pan']   = BackgroundFileNamePan
  BackgroundImageryFilenameDict['bg_ndvi']  = BackgroundFileNameNDVI
  return BackgroundImageryFilenameDict

def CreateImageryNDVI( FileArrayPointers,OutputDirectory,ReferenceDataset ):
  '''
  function CreateImageryNDVI( FileArrayPointers,OutputDirectory,ReferenceDataset ):
  This function computes NDVI (Normalized Diff. Vegetation Index)
  as well as SAVI (Soil-Adjusted NDVI) for 10 different thresholds 
  L = 0.1,0.2,...1.0. To this end, this function takes in file 
  pointers 

  Args:
    FileArrayPointers (list): List of NumPy memory map objects
    OutputDirectory (str): Output directory to write NDVI,SAVI imagery.
    ReferenceDataset (osgeo.gdal.Dataset): GDAL dataset for reference to write Geotiffs.
  Returns:
    dict: Dictionary{} holding filename(s) of SAVI/SAVI imagery, used for veg. classifiaction. 
  '''

  # Get NumPy memory map objects (File Pointers) for 
  # the following NumPy arrays for the following bands:
  #   (1) Red Band
  #   (2) Green Band
  #   (3) Blue Band
  #   (4) NIR Band (Near-Infrared)
  # ---------------------------------------------------
  FilePointerRed   = FileArrayPointers[0]
  FilePointerGreen = FileArrayPointers[1]
  FilePointerBlue  = FileArrayPointers[2]
  FilePointerNIR   = FileArrayPointers[3]

  # Initialize dict{} (HASH) to hold filenames 
  # for NDVI, as well as Soil-Adjusted NDVI (SAVI)
  # for L = 0.1,0.2,...1.0
  # ----------------------------------------------
  NDVI_Imagery_Dict={}

  # Use File Pointers to NumPy memory maps to 
  # NIR and Red Bands to compute NDVI
  # (Normalized Difference Vegetation Index)
  # ----------------------------------------------

  with warn.catch_warnings():
    warn.filterwarnings('ignore',category=RuntimeWarning)
    nrows,ncols = ReferenceDataset.RasterYSize,ReferenceDataset.RasterXSize
    FilePointerNDVI  = ( FilePointerNIR - FilePointerRed ) / \
      ( FilePointerNIR + FilePointerRed )
    FilePointerNDVI[np.isnan(FilePointerNDVI)]=-1.0

    OutnameNDVI = os.path.join( OutputDirectory, 'NDVI.tif' )
    if os.path.isfile( OutnameNDVI ) : os.remove( OutnameNDVI) 
    WriteGeotiff( ReferenceDataset, OutnameNDVI, FilePointerNDVI )
    NDVI_Imagery_Dict['ndvi'] = OutnameNDVI

  # compute soil-adjusted NDVI (SAVI) for L = 0.1, 0.2 ... 1.0
  # formula: 
  #   SAVI = [ ( NIR - red ) / ( NIR + red + L ) ] * (1+L)
  # ----------------------------------------------------------

  Threshes = [ 0.1 ,0.2, 0.3 , 0.4 , 0.5, 0.6, 0.7, 0.8, 0.9, 1.0   ]
  Labels   = [ '01','02','03','04' ,'05','06','07', '08','09', '10' ]

  #L = 0.1
  #while(L<1.0):
  for L,Label in zip( Threshes,Labels ):

    with warn.catch_warnings():
      warn.filterwarnings('ignore',category=RuntimeWarning)
      savi = ((FilePointerNIR - FilePointerRed )/(FilePointerNIR + FilePointerRed + L )) * (1+L)

    #thresholdStringSAVI = '%02d'%int(str(L).replace('.',''))
    thresholdStringSAVI  = Label
    outnameSAVI = os.path.join( OutputDirectory , 'SAVI_' +thresholdStringSAVI+'.tif')
    if os.path.isfile(outnameSAVI): os.remove(outnameSAVI)
    WriteGeotiff( ReferenceDataset , outnameSAVI , savi )
    NDVI_Imagery_Dict['savi'+thresholdStringSAVI] = outnameSAVI 
    del savi
    L+=0.1
  return ( NDVI_Imagery_Dict , FilePointerNDVI )

def ComputeSimulatedPanchromaticBand( FileArrayPointers,OutDir,ReferenceDataset ):
  '''
  function ComputeSimulatedPanchromaticBand( FileArrayPointers,OutDir,ReferenceDataSet ):
  This function computes a simulated panchromatic band by taking an average of 
  the Red,Green,Blue, and NIR bands. This function is called if the user does not 
  pass-in a Panchromatic image filename at the command-line (see VegetationClassification.py).
  To this end, this function adds all four of these main multispectral bands (RGB,NIR), 
  and divides the sum by 4. The array 2D NumPy is returned as a memory map object.

  Args:
    FileArrayPointers (list): List of NumPy memory objects for Red,Green,Blue,NIR NumPy arrays.
    OutDir (str): Output directory where RGB,NIR Geotiffs are located.
    ReferenceDataset (osgeo.gdal.Dataset): GDAL dataset for reference to get projection and geotransform.
  Returns:
    np.memmap: Memory map (File Pointer) newly-created 1-band Panchromatic array.
  '''

  # Get NumPy memory-map objects (File Pointers) to following arrays:
  #   (1) Red Band
  #   (2) Green Band
  #   (3) Blue Band
  #   (4) NIR band 
  # -------------------------------------

  FilePointerRed   = FileArrayPointers[0]
  FilePointerGreen = FileArrayPointers[1]
  FilePointerBlue  = FileArrayPointers[2]
  FilePointerNIR   = FileArrayPointers[3]

  # Get 2D dimensions of RGB,NIR imagery set
  # ----------------------------------------

  nrows,ncols = ReferenceDataset.RasterYSize,ReferenceDataset.RasterXSize

  # Create simulated Panchromatic Image: 
  #   ( Red + Green + Blue + NIR ) / 4.0
  # as a NumPy MemoryMap object.
  # ----------------------------------------

  FilePointerPan = ( FilePointerRed + FilePointerGreen + FilePointerBlue + FilePointerNIR ) / 4.0
  OutnamePanGeotiff = os.path.join( OutDir ,'Pan.tif')
  
  # If panchromatic image file already exists, then remove it.
  # ----------------------------------------------------------
  if os.path.isfile(OutnamePanGeotiff): os.remove(OutnamePanGeotiff)
  
  # Write the Panchromatic Band, just created, to Geotiff
  # -----------------------------------------------------
  WriteGeotiff( ReferenceDataset, OutnamePanGeotiff, FilePointerPan)
  return ( OutnamePanGeotiff , FilePointerPan )
