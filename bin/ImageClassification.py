import os
import sys
import csv
import pandas
import numpy as np
import sklearn
from matplotlib.pylab import *
from osgeo import osr,gdal
from osgeo.gdalnumeric import ravel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from Misc import WriteGeotiff,WritePNG

def ExtractSpectralValues( ImageDataset,BandNumber,StartRow,EndRow ):
  '''function ExtractSpectralValues( ImageDataset,BandNumber,StartRow,Endrow ):
  
  This function reads a GDAL image dataset and returns a portion of 
  one of its bands (2D numpy arrays) as a 1D array. This subset 
  includes all columns and those rows starting and ending with 
  the input parameters StartRow and EndRow.

  Args:
    ImageDataset (osgeo.gdal.Dataset): GDAL Image dataset.
    BandNumber (int): NumPy array. Should be greater than or equal to 1.
    StartRow (int): Start row. Should be greater than or equal to 0.
    EndRow (int): End row to extract. Should be less than number of columns.
  Returns:
    numpy.ndarray: A flattened (1D) array of the array data.
  '''
  raster = ImageDataset.GetRasterBand(BandNumber+1).ReadAsArray()[StartRow:EndRow,:]
  return raster.flatten()

def PrepareTrainingDataFromCSV( TrainingPixelValueDataCSV ):
  '''function PrepareTrainingDataFromCSV( training_csv ):
  This function reads the input CSV that contains spectral pixel values 
  used for classifiying forest (trees/woods) in a set of satellite 
  imagery. In particular, this function reads the input CSV into a 
  pandas dataframe, randomizes the columns, and returns two separate 
  pandas dataframes in a tuple(). The first dataframe will contain 
  spectral pixel values corresponding to all those tree/non-tree points
  contained in the input shapefile(s) (see VegetationClassfiication.py). 
  These pixel values include NDVI, SAVI for L = 0.1,0.2,...,1.0, as well
  as RGB, Panchromatic, and background NDVI, Panchromatic, and RGB 
  spectral pixel values. The second output dataframe contains one 
  single columns for "tree" and "non-tree". 

  Args:
    TrainingPixelValueDataCSV (str): Filename string containing training pixel value data.
  Returns:
    tuple: Two Pandas dataframes: one for tree/nontree (1 or 0), one with spectral pixel values.
  '''

  # Read input CSV columns containing spectral pixel values into 
  # a Python pandas dataframe 
  # ---------------------------------------------------------------
  TrainingDataframe = pandas.read_csv(TrainingPixelValueDataCSV, header=0)

  # Randomize columns in dataframe, then 
  # rename "Label" column as "tree_binary" 
  # i.e. woods/non-woods (forest/non-forest)
  # -----------------------------------------
  TrainingDataframe = TrainingDataframe.reindex( np.random.permutation(TrainingDataframe.index) )
  TrainingDataframe.rename(columns={'Label' : 'tree_binary'},inplace=True)

  # Read the following into two SEPARATE pandas dataframes:
  #   (1) First 22 columns containing spectral pixel values
  #   (2) Final column (tree/non-tree) that is 1 or 0
  # Return these two dataframes.
  # ---------------------------------------------------------
  TrainingSpectralValuesDataframe = TrainingDataframe.iloc[:,0:22]#ix[:,0:22]
  TrainingTreeNonTreeDataframe = TrainingDataframe['tree_binary']
  return ( TrainingSpectralValuesDataframe,TrainingTreeNonTreeDataframe )

def BuildRandomForestModel( NTrees,SpectralValuesDataFrame,TreeNonTreeDataFrame ):
  '''function BuildRandomForestModel( NTrees,SpectralValuesDataFrame,TreeNonTreeDataFrame ):
  This function takes in two dataframes and builds a random-forest 
  model using sklearn.ensemble's ExtraTreesClassifier() function. 
  The first dataframe contains 22 columns for all imagery spectral 
  pixel values corresponding to trees/non-trees, and the second 
  contains a single column for tree/non-tree (woods/non-woods, 
  1 or 0). It initializes an sklearn ExtraTreesClassifier object, 
  and fits this to the two input dataframes.

  Args: 
    NTrees (int): 
    SpectralValuesDataFrame : 
    TreeNonTreeDataFrame : 
  Returns: 
    sklearn.ensemble.forest.ExtraTreesClassifier: Output fitted classifier. 
  '''

  # Initialize ExtraTreesClassifer()
  # --------------------------------

  ClassifierRandomForest = ExtraTreesClassifier( 
    n_estimators=NTrees,
    max_depth=None,
    min_samples_split=1.0,
    random_state=0 
  )

  # Fit ExtraTreesClassifier() object to both input 
  # dataframes and return this object.
  # ------------------------------------------------
  
  return ClassifierRandomForest.fit( 
    SpectralValuesDataFrame,
    TreeNonTreeDataFrame
  )

def ReadPixelDataIntoRandomForestModel(SpectralImageryDict,StartRow,EndRow):
  '''ReadPixelDataIntoRandomForestModel(
  This function matches the image filename containing pixel data with 
  its corresponding column name in the input CSV (dataframe) used for 
  vegetation (woods/trees) classification. To this end, this function
  iterates through all the image filenames and creates a list[] of the 
  matching variable names that match that filename. This list[] of 
  variable names is used to gather a strip of pixel data confined between
  a starting and ending row (StartRow,EndRow inputs). This pixel data 
  becomes part of the output Dataframe of this function.

  Args:
    SpectralImageryDict (dict): Dictionary holding names of imagery (i.e. NDVI,SAVI,RGB,...)
    StartRow (int): Starting row in imagery, greater than or equal to 0. 
    EndRow (int): Ending row in imagery.
  Returns:
    pandas.core.frame.DataFrame: Output dataframe containing pixel values and variable names.
  '''

  # Initialize output dataframe that will hold variable names 
  # (columns in dataframe) as well as ALL spectral pixel values 
  # for a strip of data in input imagery
  # ------------------------------------------------------------
  OutDataFrame = pandas.DataFrame()

  for EachFile in SpectralImageryDict.values(): 

    # Based on filename string, create list[] of appropriate 
    # variable name string(s) that will be column names in output
    # dataframe.
    # ------------------------------------------------------------

    if EachFile.endswith('Red.tif')     and 'Background' not in EachFile: 
      VariableNames = ['R']
    elif EachFile.endswith('Green.tif') and 'Background' not in EachFile: 
      VariableNames = ['G']
    elif EachFile.endswith('Blue.tif')  and 'Background' not in EachFile: 
      VariableNames = ['B']
    elif EachFile.endswith('NDVI.tif')  and 'Background' not in EachFile: 
      VariableNames = ['NDVI']
    elif EachFile.endswith('NIR.tif')   and 'Background' not in EachFile: 
      VariableNames = ['NIR']
    elif EachFile.endswith('SAVI_01.tif'): 
      VariableNames = ['SAVI01']
    elif EachFile.endswith('SAVI_02.tif'): 
      VariableNames = ['SAVI02']
    elif EachFile.endswith('SAVI_03.tif'): 
      VariableNames = ['SAVI03']
    elif EachFile.endswith('SAVI_04.tif'): 
      VariableNames = ['SAVI04']
    elif EachFile.endswith('SAVI_05.tif'): 
      VariableNames = ['SAVI05']
    elif EachFile.endswith('SAVI_06.tif'): 
      VariableNames = ['SAVI06']
    elif EachFile.endswith('SAVI_07.tif'): 
      VariableNames = ['SAVI07']
    elif EachFile.endswith('SAVI_08.tif'): 
      VariableNames = ['SAVI08']
    elif EachFile.endswith('SAVI_09.tif'): 
      VariableNames = ['SAVI09']
    elif EachFile.endswith('SAVI_10.tif'): 
      VariableNames = ['SAVI10']
    elif EachFile.endswith('BackgroundBlue.tif')  : 
      VariableNames = ['Background_Blue']
    elif EachFile.endswith('BackgroundGreen.tif') : 
      VariableNames = ['Background_Green']
    elif EachFile.endswith('BackgroundNDVI.tif')  : 
      VariableNames = ['Background_NDVI']
    elif EachFile.endswith('BackgroundNIR.tif')   : 
      VariableNames = ['Background_NIR']
    elif EachFile.endswith('BackgroundPan.tif')   : 
      VariableNames = ['Background_Pan']
    elif EachFile.endswith('BackgroundRed.tif')   : 
      VariableNames = ['Background_Red']
    elif EachFile.endswith('Pan.tif') and 'background' not in EachFile: 
      VariableNames = ['Pan']
    else: continue

    # OPEN current Geotiff/JPEG in imagery dataset
    # (for current iteration)
    # --------------------------------------------
    RasterImageDataset = gdal.Open(EachFile)

    # Iterate through all bands in image dataset, as well
    # as variable names from if-statement above , and 
    # append output Pandas dataframe with variable column
    # name string and an array (1D) of pixel values
    # ---------------------------------------------------

    for Band,VariableName in enumerate(VariableNames):
      OutDataFrame[VariableName] = ExtractSpectralValues(
        RasterImageDataset,Band,StartRow,EndRow
      )

    # CLOSE current Geotiff/JPEG in imagery dataset
    # (for current iteration)
    # --------------------------------------------
    RasterImageDataset=None
    del RasterImageDataset

  return OutDataFrame

def GetClassification( FullVariablesDataFrame,ClassifierFitRandomForest,dims):
  '''function GetClassification( FullVariablesDataFrame,
  For the entire image area, or a strip of it, this function performs 
  the actual classification, returning a 2D array of 1s and 0s marking
  trees (forest/woods) and non-trees (non-forest). To this end, it 
  formats the input dataframe and removes rows with NoData (np.nan), and 
  calls an input ExtraTreesClassifier object's predict() method with 
  the dataframe as input to return the predictive mask of 1s and 0s.
  This mask is reshaped into a 2D array and returned.

  Args:
    FullVariablesDataFrame (pandas.core.frame.DataFrame): Input dataframe with pixel values.
    ClassifierFitRandomForest (sklearn.ensemble.ExtraTreesClassifier): Classifier object.
    dims (tuple): number of rows and columns of image area or strip.
    dims (tuple): 2D dimensions of image domain or subset (strip).
  Returns:
    np.ndarray: Output vegetation classification.
  '''
  
  # Format input spectral-values dataframe ... replace 
  # mask with NoData (np.nan)
  # -----------------------------------------------------
  fullVariablesDataFrame = FullVariablesDataFrame.dropna(axis=1)
  fullVariablesDataFrame = FullVariablesDataFrame.replace('-',np.nan)
  
  # Perform acutal prediction ... creating output array 
  # or sub-array (strip) of 1s and 0s. Return the 
  # reshaped array (1D to 2D) 
  # -----------------------------------------------------
  classifierPredictRandomForest = ClassifierFitRandomForest.predict(FullVariablesDataFrame)
  classifierPredictRandomForest = np.array(classifierPredictRandomForest,dtype=np.int8)
  return np.reshape(classifierPredictRandomForest,dims)

def RandomForestClassification( ImgDict,CSV,OutDir,NTrees ):
  '''function RandomForestClassification( ImgDict,CSV,OutDir,NTrees ):
  This is the primary method for creating our final output Geotiff image 
  that contains our vegetation/forest classification. To this end, it does
  the following:
    (1) Reads input CSV of training points into two separate dataframes.
    (2) Builds a random-forest model using an ExtraTreesClassifier
    (3)  
    (4) Writes final result (2D array of 1s and 0s) to Geotiff. 
  Args:
    ImgDict (dict): Dictionary{} containing imagery for vegetation classification.
    CSV (str): Name of CSV containing all relevant pixel value training data.
    OutDir (str): Output directory.
    NTrees (int): Number of trees for ExtraTreesClassifier() object. For classification.
  '''

  # Open up panchromatic image file 
  # as a GDAL dataset. Read its 2D dimensions.
  # ---------------------------------------------
  
  ds=gdal.Open(ImgDict['pan'])
  NROWS,NCOLS = ds.RasterYSize,ds.RasterXSize
  ds=None
  del ds

  # create SEPARATE randomized pandas data-frames containing:
  #   (1) 22 columns for spectral values (NDVI,SAVI,RGB,...) with data from input CSV
  #   (2) 1 column for tree/nontree (woods/non-woods) 1 or 0 label with data from CSV
  # ---------------------------------------------------------------------------------
  ( TrainingSpectralValueDataframe,TrainingTreeValueDataframe ) = PrepareTrainingDataFromCSV( CSV )
  
  # Create ExtraTreesClassifier() object from sklearn.ensemble
  # using two input dataframes.
  # ----------------------------------------------------------

  ClassifierRandomForestFit = BuildRandomForestModel( NTrees,
    TrainingSpectralValueDataframe,
    TrainingTreeValueDataframe
  )

  # get a list of all rows ( 0 .. .. nrows-1 ) and 
  # divide it into 20 chunks ... hence we are cutting 
  # each file in our image dataset into strips. This is
  # so that the ReadPixelDataIntoRandomForestMode() 
  # function does not have to return ALL pixel values
  # at once. This would cause a MemoryError.
  # ---------------------------------------------------

  if NROWS>3000: # pretty arbitrary ... 
    NStrips = 20
  else: 
    NStrips = 1

  AllRowIndices = np.arange( 0,NROWS,step=1 )
  RowChunks     = np.array_split( AllRowIndices, NStrips )

  # Create empty list[] that will hold strips containing 
  # our final classification (1s and 0s) of trees/nontrees
  # (that is, vegetation/non-vegetation).
  # -------------------------------------------------------
  ImageStrips = []

  for RowChunk in RowChunks:

    # Getting starting and ending row for strip in which
    # we will proceed with vegetation (forest) classification.
    # --------------------------------------------------------
    StartImageRow = RowChunk[  0  ]
    EndImageRow   = RowChunk[ -1  ]+1
 
    # Get dataframe containing variable names and spectral 
    # pixel values for image area or sub-strip
    # --------------------------------------------------------

    DataFrameForImageStrip = ReadPixelDataIntoRandomForestModel(
      ImgDict,
      StartImageRow,
      EndImageRow
    )

    # Create classified strip of 1s and 0s for 
    # vegetation/non-vegetation.
    # ----------------------------------------
    ClassifiedDataStrip = GetClassification(
      DataFrameForImageStrip,
      ClassifierRandomForestFit,
      (EndImageRow-StartImageRow,NCOLS)
    )

    # Append list[] containing image data strips (2D NumPy arrays)
    # each of which contains 1s and 0s for final classification.
    # ------------------------------------------------------------
    ImageStrips.append(ClassifiedDataStrip)

  # Concatenate all strips of data containing 1s and 0s 
  # to form final image of classifiaction of vegetation 
  # ---------------------------------------------------
  FinalForestVegetationClassification = np.concatenate( ImageStrips,axis=0 )

  locs = np.where( FinalForestVegetationClassification == 1 )
  print( 'number of tree pixels: ' , str(locs[0].shape[0]))

  # Write final classification to Geotiff
  # -------------------------------------
  OutNameGeotiffClassified = os.path.join( 
    OutDir, 'vegetation_forest_classification.tif' )
  WriteGeotiff( gdal.Open(ImgDict['pan']),
    OutNameGeotiffClassified,FinalForestVegetationClassification ) 

  # Write PNG showing "quick look" of forest/woods/vegetation classification.
  # -------------------------------------------------------------------------
  OutNamePNG = os.path.join(
    OutDir,'vegetation_forest_classification.png')
  WritePNG( OutNamePNG, FinalForestVegetationClassification ) 
