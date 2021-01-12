import glob
import getopt
import os
import sys
import numpy as np
import warnings as warn
import shutil
from osgeo import osr,gdal
from osgeo import gdalconst
from distutils.spawn import find_executable
from TrainingImagery import *
from TrainingPoints import CreateTrainingPointsCSV
from ImageClassification import RandomForestClassification
from Misc import RunProcess,ResampleImage

def usage(message=None):

  # print out optional input message 
  # --------------------------------

  if message is not None: 
    print(message)

  # print out primary usage message
  # -------------------------------

  print('''
    NAME:
      VegetationClassification
    DESCRIPTION
      It is common for satellite imagery to include bands for the Red, Green, Blue, 
      and Near-Infrared (NIR) bands, among others. Each of these would correspond to 
      a frequency and/or reflectance (or irradiance) channel for a particular satellite 
      (i.e. Landsat, MODIS, Sentinel, and so on). These bands can be used to compute 
      various combinations that would help to classify vegetation in the satellite 
      imagery (i.e. Normalized Difference Vegetation Index, NDVI).

      This command-line program uses Python and machine-learning to classify a set of 
      satellite imagery into a vegetation and non-vegetation. To this end, this program 
      computes various parameters, Normalized Difference Vegetation Index (NDVI), SAVI 
      (Soil-Adjusted NDVI) for 10 different thresholds, as well as a Panchromaic image 
      (if needed). The final output is a Geotiff with 1s and 0s, whereas 1s mark 
      trees/woods and 0s mark non-trees (non-forest/woods).
    USAGE:
      To use this program, run it at command-line with the 
      following possible arguments:

      $ python3 VegetationClassification.py
         Command-Line Options:
          { --help, --h, -h }
            Display this help usage message
          { --outdir, -o    } 
            Output directory path (optional)
          { --ntrees, --numtrees, --numbertrees, -n }
            Number of trees used in ExtraTreesClassifier (optional, default is 3)
          { --panchromatic, -p  }
            Panchromatic Image Filename (optional)
          { --red, -r           }
            Image Filename for "Red" Band or Channel (required)
          { --green, -g         }
            Image Filename for "Green" Band or Channel (required)
          { --blue, -b          }
            Image filename for "Blue" Band or Channel (required)
          { --nir, -n           }
            Image Filename for "NIR" or Near-Infrared Band or Channel (required)
          { -z, --background, --nontrees, --nonvegetation }
            Shapefile (.shp extension) for points marking NOT trees/vegetation (required)
          { -t, --targets, --trees, --vegetation           }
            Shapefile (.shp extension) for points marking trees/vegetation (woods) (required)
          { -i, --ignore, --nodata                         }
            Imagery pixel value to ignore. NoData value. Usually 0 or -9999 (optional).
    EXAMPLE USAGE:

      This example shows how to use this program on the 
      Linux command-line (typically a BASH shell).

      Using Sentinel 2A imagery:
      
      $ red=T34SDH_20180830T093029_B04_10m.jp2
      $ green=T34SDH_20180830T093029_B03_10m.jp2
      $ blue=T34SDH_20180830T093029_B02_10m.jp2
      $ NIR=T34SDH_20180830T093029_B08_10m.jp2

      $ python VegetationClassification.py 
        --red $red 
        --green $green 
        --blue $blue 
        --nir $NIR
        --trees trees.shp 
        --nontrees not_trees.shp 
        --ntrees 3 
        --nodata 0
    AUTHOR: 
      Gerasioms A. Michalitsianos
      gerasimosmichalitsianos@gmail.com
      Last Updated: 12 January 2021 
  ''')
  sys.exit(1)

def main(): 
 
  # ---------------------------------------------------------------------
  # define empty strings to initialize for the 
  # following required input flag arguments:
  # 
  #   (1) Number of Trees for Random Forest Classification
  #   (2) Output Directory
  #   (3) "Red" Visible Channel Input Filename
  #   (4) "Green" Visible Channel Input Filename
  #   (5) "Blue" Visible Channel Input Filename
  #   (6) Near-Infrared or "NIR" Input Filename
  #   (7) Panchromatic Input Filename
  #   (8) Name of POINTS shapefile that defines "background" 
  #       or non-trees (non-woods/forest)
  #   (9) Name of POINTS shapefile that defines "trees" (woods/forest)
  #   (10) NoData value 
  # ----------------------------------------------------------------------
  args = [
    'help',
    'ntrees=','numtrees=','numbertrees=', 
    'outdir=',
    'red=',
    'green=',
    'blue=',
    'nir=',
    'pan=',
    'background=','nontrees=','nonvegetation=',
    'targets=','trees=','vegetation=',
    'ignore=','nodata='
  ]

  # Initialize Image Filenames (i.e. JPEG,Geotiff) for 
  # Red,Green,Blue,NIR, and optional Panchromatic 
  # channels as empty strings
  # ----------------------------------------------------
  RedImageFileName             = ''
  GreenImageFileName           = ''
  BlueImageFileName            = ''
  NIRImageFileName             = ''
  PanchromaticImageFileName    = ''

  # Initialize output directory string, filenames for POINTS
  # shapefiles for forest/non-forest (woods/non-woods), number
  # of trees to use for random-forest classification, 
  # a NoData number string, and NoData value 
  # ------------------------------------------------------------
  TargetPointsShapefile        = ''
  BackgroundPointsShapefile    = ''
  NumberTreesForClassification = 3
  NoDataString = ''
  NoDataValue  = 0

  try:
    Options,Arguments = getopt.getopt(
      sys.argv[1:],'h:p:g:b:r:n:z:t:n:i:',args )
  except getopt.GetoptError:
    usage()

  # ----------------------------------
  # iterate through command-line args. 
  # ----------------------------------
  for Option,Argument in Options:
    if Option in   ('-h','--h','--help'):
      usage()
    elif Option in ('-p','--pan','--panchromatic'):
      PanchromaticImageFileName    = Argument
    elif Option in ('-g','--green'):
      GreenImageFileName           = Argument
    elif Option in ('-b','--blue'):
      BlueImageFileName            = Argument
    elif Option in ('-r','--red'):
      RedImageFileName             = Argument
    elif Option in ('-n','--nir'):
      NIRImageFileName             = Argument
    elif Option in ('-z','--background','--nontrees','--nonvegetation'):
      BackgroundPointsShapefile    = Argument
    elif Option in ('-t','--targets','--trees','--vegetation'):
      TargetPointsShapefile        = Argument
    elif Option in ('-n','--ntrees','--numtrees','--numbertrees'):
      NumberTreesForClassification = Argument
    elif Option in ('-i','--ignore','--nodata'):
      NoDataString                 = Argument
    else: pass

  # if user did not pass-in NoData value (usually 0 or -999)
  # then it will remain as  0 (zero) as a default
  # --------------------------------------------------------

  if NoDataString != '':
    try:
      NoDataValue = float(NoDataString)
    except:
      print('  \n    Invalid nodata value: "'+ str(NoDataString) + '". \n   '+\
        ' Pass-in valid no data pixel value (int or float) '+\
        ' using --nodata, -i, or --ignore flags.')
      usage()

  # make sure number of trees for classification 
  # is an integer 
  # --------------------------------------------

  try:
    NumberTreesForClassification = int(NumberTreesForClassification)
  except:
    usage('  \n    Number of trees should be an integer.')
  np.random.seed(NumberTreesForClassification)

  # make sure user passed-in valid shapefile(s) 
  # for target points (i.e. trees/vegetation)
  # and background (i.e. non-trees or non-vegetation)
  # -------------------------------------------------

  if TargetPointsShapefile == '':
    print('  \n    Pass-in valid shapefile with target points + '\
    '(i.e. trees). Use -t or other listed flags.')
    usage()
  elif BackgroundPointsShapefile == '':
    print('  \n    Pass-in valid shapefile with background points '+\
    ' (i.e. non-trees). Use -b or other listed flags. ')
    usage()

  # --------------------------------------------------
  # define output directory as same as input directory
  # --------------------------------------------------
  OutputDirectory = os.path.dirname( RedImageFileName )
  if not os.path.isdir( OutputDirectory):
    usage('  \n  Not an existing directory: '+OutputDirectory )

  # ---------------------------------------------------
  # make sure user at least passed-in filename strings
  # for the Red,Green,Blue and NIR bands. Panchromatic
  # is optional.
  # ---------------------------------------------------
  if '' in [ RedImageFileName,GreenImageFileName,BlueImageFileName,NIRImageFileName ]:
    usage('  \n    Please pass in filenames for Red,Green,Blue,NIR bands (JPEG/Geotiff).')

  # Make sure that Red,Green,Blue,NIR image filenames
  # passed-in actually exist on the file-system
  # --------------------------------------------------- 

  if not os.path.isfile( RedImageFileName ):
    usage('  \n    Not an existing file: '+RedImageFileName)
  if not os.path.isfile( GreenImageFileName ):
    usage('  \n    Not an existing file: '+GreenImageFileName)
  if not os.path.isfile( BlueImageFileName ):
    usage('  \n    Not an existing file: '+BlueImageFileName)
  if not os.path.isfile( NIRImageFileName ):
    usage('  \n    Not an existing file: '+NIRImageFileName)
  
  # ----------------------------------------------------------------
  # open up red,green,blue,nir image files ... store their arrays
  # ----------------------------------------------------------------
  DatasetRed   = gdal.Open( RedImageFileName   )
  DatasetNIR   = gdal.Open( NIRImageFileName   )
  DatasetBlue  = gdal.Open( BlueImageFileName  )
  DatasetGreen = gdal.Open( GreenImageFileName )

  # Make sure Red,NIR,Blue,Green GDAL file-readers are 
  # valid GDAL objects or GDAL datasets (
  # ----------------------------------------------------------------

  for FileName,ImgDataset in zip( [ 
      RedImageFileName,
      NIRImageFileName,
      BlueImageFileName,
      GreenImageFileName ] , 
      [ DatasetRed,DatasetNIR,DatasetBlue,DatasetGreen ] ):
    if 'none' in str(type(ImgDataset)).lower():
      usage('  \n    Not a valid GDAL image (raster) dataset or file: '+FileName )

  # Make sure that all 4 multispectral image file GDAL datasets
  # all have SAME dimensions 
  # -----------------------------------------------------------
  InputColumnDimensions = np.array( [DatasetRed.RasterXSize,
    DatasetNIR.RasterXSize,
    DatasetBlue.RasterXSize,
    DatasetGreen.RasterXSize] )

  if np.unique( InputColumnDimensions ).size>1:
    usage('  \n    All multispectral input imagery (RGB,NIR) should have same x dimension. Exiting ... ')
  
  InputRowDimensions = np.array( [DatasetRed.RasterYSize,
    DatasetNIR.RasterYSize,
    DatasetBlue.RasterYSize,
    DatasetGreen.RasterYSize] )
  
  if np.unique( InputRowDimensions ).size>1:
    usage('  \n    All multispectral input imagery (RGB,NIR) should have same y dimension. Exiting ... ')

  # Read multspectral image files (RGB,NIR) as NumPy arrays 
  # -------------------------------------------------------
  nrows,ncols = DatasetRed.RasterYSize, DatasetRed.RasterXSize
  FilePointerRed   = DatasetRed.GetRasterBand(1).ReadAsArray().astype(float)
  FilePointerNIR   = DatasetNIR.GetRasterBand(1).ReadAsArray()
  FilePointerBlue  = DatasetBlue.GetRasterBand(1).ReadAsArray()
  FilePointerGreen = DatasetGreen.GetRasterBand(1).ReadAsArray()

  # if panchromatic (gray-scale) image file (Geotiff/JPEG) was NOT 
  # passed-in at command-line, then compute a simulated panchromatic
  # band by taking an average of the Red,Green,Blue,NIR bands.
  # ----------------------------------------------------------------
  if PanchromaticImageFileName == '':

    # if panchromatic image filename was not passed-in at command-line,
    # then create it
    # ------------------------------------------------------------------

    ( PanchromaticImageFileName, FilePointerPan ) = ComputeSimulatedPanchromaticBand(
      [ FilePointerRed,FilePointerGreen,FilePointerBlue,FilePointerNIR ],
      OutputDirectory,
      DatasetRed
    )

  else: 

    # if the user DID pass-in the panchromatic image filename at the 
    # the command-line, then COPY IT to the output directory with 
    # the name "Pan.tif" , but make sure the file actually exists 
    # first 
    # --------------------------------------------------------------
    if not os.path.isfile( PanchromaticImageFileName ):
      usage('  \n    Not an existing file: '+PanchromaticImageFileName)
    
    # make sure panchromatic image filename passed-in is vaild 
    # GDAL dataset (osgeo.gdal.Dataset)
    # ---------------------------------------------------------
    TempPanDataset = gdal.Open( PanchromaticImageFileName )
    if 'none' in str(type(TempPanDataset)):
      usage('  \n    Not a valid GDAL raster dataset: '+PanchromaticImageFileName)
    TempPanDataset = None

    OutFileNamePan = os.path.join( 
      OutputDirectory, 'Pan.tif')
    shutil.copyfile( PanchromaticImageFileName, OutFileNamePan )
    PanchromaticImageFileName = OutFileNamePan
 
    # if panchromatic image file passed-in DOES NOT have the same 
    # dimension as multispectral imagery passed-in (i.e. the 
    # low-res. RGB,NIR imagery), then resample the panchromatic 
    # image file to same dimensions as multispectral imagery 
    # --------------------------------------------------------------
    PanchromaticDataset = gdal.Open( PanchromaticImageFileName )
    PanDims = (PanchromaticDataset.RasterYSize,PanchromaticDataset.RasterXSize)
    MSDims = FilePointerRed.shape
    
    if ( MSDims[0] != PanDims[0] ) or ( MSDims[1] != PanDims[1] ): 
      ( PanchromaticFileName, FilePointerPan ) = ResampleImage( 
        PanchromaticImageFileName,
        PanchromaticDataset,
        DatasetRed, 
        PanchromaticImageFileName, 
        gdalconst.GRA_NearestNeighbour
      )

  # Make sure the computer in which this program is run 
  # has gdal_translate installed (command-line tool from GDAL)
  # ----------------------------------------------------------

  GDAL_Translate_Path = find_executable( 'gdal_translate' )
  if GDAL_Translate_Path is None:
    usage('  \n    Unable to find gdal_translate command-line tool. Exiting ... ')
  
  # copy the Red,Green,Blue,NIR image files (that were REQUIRED
  # to be passed-in at the command-line to the output directory
  # with file-names Red.tif,Green.tif,Blue.tif,NIR.tif
  # -------------------------------------------------------------
  OutFileNameNIR    = os.path.join( 
    OutputDirectory, 'NIR.tif' )
  OutFileNameRed    = os.path.join( 
    OutputDirectory, 'Red.tif' )
  OutFileNameGreen  = os.path.join( 
    OutputDirectory, 'Green.tif' )
  OutFileNameBlue   = os.path.join( 
    OutputDirectory, 'Blue.tif' )

  RunProcess( GDAL_Translate_Path+' -q -of GTiff '+NIRImageFileName+' '+OutFileNameNIR     )
  NIRImageFileName = OutFileNameNIR

  RunProcess( GDAL_Translate_Path+' -q -of GTiff '+RedImageFileName+' '+OutFileNameRed     )
  RedImageFileName = OutFileNameRed
  
  RunProcess( GDAL_Translate_Path+' -q -of GTiff '+GreenImageFileName+' '+OutFileNameGreen )
  GreenImageFileName = OutFileNameGreen

  RunProcess( GDAL_Translate_Path+' -q -of GTiff '+BlueImageFileName+' '+OutFileNameBlue   )
  BlueImageFileName = OutFileNameBlue

  # store the following into a dictionary: 
  #  (1) Red band filename
  #  (2) Green band filename
  #  (3) Blue band filename
  #  (4) NIR band filename
  #  (5) Panchromatic (Pan) band filename
  # ----------------------------------------

  ClassificationImageryDict={}
  ClassificationImageryDict['pan']   = PanchromaticImageFileName
  ClassificationImageryDict['red']   = RedImageFileName
  ClassificationImageryDict['green'] = GreenImageFileName
  ClassificationImageryDict['blue']  = BlueImageFileName
  ClassificationImageryDict['nir']   = NIRImageFileName

  # compute Normalized Difference Vegetation Index (NDVI), 
  # as well as Soil-Adjusted NDVI (SAVI). Write these to Geotiffs
  # and store the filenames into a dictionary{} to be appended
  # to the dictionary{} imageryDict above
  # ----------------------------------------------------------------

  ( NDVI_FileName_Dict, FilePointerNDVI ) = CreateImageryNDVI( 
    [ FilePointerRed,FilePointerGreen,FilePointerBlue,FilePointerNIR ],
    OutputDirectory,
    DatasetRed
  )
  ClassificationImageryDict.update( NDVI_FileName_Dict )

  # compute Gaussian-filtered "background" imagery for the 
  # following bands: 
  #   (1) Panchormatic band
  #   (2) Red band
  #   (3) Green band
  #   (4) Blue band 
  #   (5) NIR band
  #   (6) NDVI
  # Then update our master dict{} holding all filename 
  # strings for imagery that will be used in final 
  # forest/vegetation image classification
  # --------------------------------------------------------
  ClassificationImageryDict.update(CreateImageryBackground(
    [FilePointerRed,FilePointerGreen,FilePointerBlue,FilePointerNIR,FilePointerPan,FilePointerNDVI],
    OutputDirectory,
    DatasetRed
  ))

  # Create output dataset  holding RGB bands 
  # ----------------------------------------
  ImageFileNameRGB = CreateImageRGB(
    ClassificationImageryDict['red'],
    ClassificationImageryDict['green'],
    ClassificationImageryDict['blue'],
    OutputDirectory
  )

  if ImageFileNameRGB is None: usage()
  ClassificationImageryDict['rgb'] = ImageFileNameRGB 

  # use the following to create a CSV holding satellite pixel value 
  # training data: 
  #   (1) Shapefile containing "target" points 
  #       i.e. latitude/longitude points of trees (vegetation)
  #   (2) Shapefile containing "background" points 
  #       i.e. latitude/longitude points of non-trees
  #   (3) Python dictionary{} containing filenames (Geotiffs/JPEGs)
  #       of satellite training data 
  #       (i.e. NDVI,SAVI,Red,Green,Blue,Background Red,
  #        Background NDVI,Panchromatic band, ... )
  #   (4) Output directory string
  # ------------------------------------------------------------------
  Classification_CSV_FileName = CreateTrainingPointsCSV(
    BackgroundPointsShapefile,
    TargetPointsShapefile,
    ClassificationImageryDict, 
    OutputDirectory,
    NoDataValue
  )

  if Classification_CSV_FileName is None:
    usage()

  # use CSV to write out an image classification to the 
  # output directory
  # ---------------------------------------------------
  RandomForestClassification(
    ClassificationImageryDict,
    Classification_CSV_FileName,
    OutputDirectory,
    NumberTreesForClassification
  ) 

if __name__ == '__main__':
  main()
