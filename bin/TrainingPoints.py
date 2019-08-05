import sys
import re
import os
import numpy as np
import shapefile
from osgeo import osr,gdal,ogr
from pyproj import Proj,transform
from distutils.spawn import find_executable
from Misc import RunProcess

def GetPixelValuesAllImagery(Row,Column,FileNameDict):
  '''function GetPixelValuesAllImagery(Row,Column,FileNameDict):
  This function takes in integer column (X) and row (Y) 
  values, and opens up all imagery (SAVI,NDVI,Pan,RGB,...)
  and retrieves the spectral pixel values at that (row,column)
  point. It returns these pixel values as a comma-delimited or
  comma-separated string. This string will later be written
  to a CSV containing "training data" in which to classify 
  vegetation in the set of satellite imagery.

  Args:
    Row (int): Row of pixel in imagery.
    Column (int): Column of pixel in imagery.
    FileNameDict (dict): Dictionary holding all filenames of imagery set.
  Returns: 
    str: A string holding pixel values from all imagery for input point (Row,Column)
  '''

  # Open up NDVI, Panchromatic, Red, Green, Blue,
  # as well as NIR bands as GDAL raster datasets.
  # ----------------------------------------------
  DatasetNDVI  = gdal.Open( FileNameDict['ndvi'] )
  DatasetPan   = gdal.Open( FileNameDict['pan'] )
  DatasetRed   = gdal.Open( FileNameDict['red'] )
  DatasetGreen = gdal.Open( FileNameDict['green'] )
  DatasetBlue  = gdal.Open( FileNameDict['blue'] )
  DatasetNIR   = gdal.Open( FileNameDict['nir'] )

  # Open up SAVI (Soil-Adjusted NDVI) Geotiffs 
  # as GDAL raster image datasets.
  # ------------------------------------------

  DatasetSAVI01 = gdal.Open( FileNameDict['savi01'] )
  DatasetSAVI02 = gdal.Open( FileNameDict['savi02'] )
  DatasetSAVI03 = gdal.Open( FileNameDict['savi03'] )
  DatasetSAVI04 = gdal.Open( FileNameDict['savi04'] )
  DatasetSAVI05 = gdal.Open( FileNameDict['savi05'] )
  DatasetSAVI06 = gdal.Open( FileNameDict['savi06'] )
  DatasetSAVI07 = gdal.Open( FileNameDict['savi07'] )
  DatasetSAVI08 = gdal.Open( FileNameDict['savi08'] )
  DatasetSAVI09 = gdal.Open( FileNameDict['savi09'] )
  DatasetSAVI10 = gdal.Open( FileNameDict['savi10'] )

  # Open up "Background" imagery (Gaussian-filtered)
  # for Red,Green,Blue,NIR,Panchromatic band, as well
  # as NDVI band.
  # --------------------------------------------------
  DatasetBackgroundRed   = gdal.Open( FileNameDict['bg_red'] )
  DatasetBackgroundGreen = gdal.Open( FileNameDict['bg_green'] )
  DatasetBackgroundBlue  = gdal.Open( FileNameDict['bg_blue'] )
  DatasetBackgroundNIR   = gdal.Open( FileNameDict['bg_nir'] )
  DatasetBackgroundPan   = gdal.Open( FileNameDict['bg_pan'] )
  DatasetBackgroundNDVI  = gdal.Open( FileNameDict['bg_ndvi'] )

  # Get Pixel values at point (Row,Column) for the following:
  #   (1) NDVI Geotiff
  #   (2) Panchromatic Image
  #   (3) Red,Green,Blue,NIR channels
  # ---------------------------------------------------------
  NDVI_Value = DatasetNDVI.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  PanchromaticValue = DatasetPan.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]

  RedValue    = DatasetRed.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  GreenValue  = DatasetGreen.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BlueValue   = DatasetBlue.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  NIRValue    = DatasetNIR.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
 
  # Get pixel values at (Row,Column) location for 
  #  Soil-Adjusted NDVI (SAVI) imagery in imagery set.
  # --------------------------------------------------
  SAVI01Value = DatasetSAVI01.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI02Value = DatasetSAVI02.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI03Value = DatasetSAVI03.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI04Value = DatasetSAVI04.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI05Value = DatasetSAVI05.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI06Value = DatasetSAVI06.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI07Value = DatasetSAVI07.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI08Value = DatasetSAVI08.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI09Value = DatasetSAVI09.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  SAVI10Value = DatasetSAVI10.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]

  # Get Pixel value(s) at Row,Column location for: 
  #   Background Red,Green,Blue,NIR,Panchromatic,and NDVI channels
  #   i.e. "background" means image was "gaussian filtered"
  # --------------------------------------------------------------
  BackgroundValueRed   = DatasetBackgroundRed.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BackgroundValueGreen = DatasetBackgroundGreen.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BackgroundValueBlue  = DatasetBackgroundBlue.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BackgroundValueNIR   = DatasetBackgroundNIR.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BackgroundValuePan   = DatasetBackgroundPan.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]
  BackgroundValueNDVI  = DatasetBackgroundNDVI.GetRasterBand(1).ReadAsArray(Column,Row,1,1)[0,0]

  # Close all Image Raster datasets ... we set them to None 
  # -------------------------------------------------------
  DatasetNDVI,DatasetPan = None, None
  DatasetRed,DatasetGreen,DatasetBlue,DatasetNIR = None,None,None,None
  DatasetSAVI01,DatasetSAVI02,DatasetSAVI03,DatasetSAVI04,DatasetSAVI05 = None,None,None,None,None
  DatasetSAVI06,DatasetSAVI07,DatasetSAVI08,DatasetSAVI09,DatasetSAVI10 = None,None,None,None,None
  DatasetBackgroundRed,DatasetBackgroundGreen,DatasetBackgroundBlue = None,None,None
  DatasetBackgroundNIR,DatasetBackgroundPan,DatasetBackgroundNDVI = None,None,None

  # Store all pixel values in a list[] 
  # ----------------------------------
  PixelValues = [ NDVI_Value,PanchromaticValue,RedValue,GreenValue,BlueValue,NIRValue,\
    SAVI01Value,SAVI02Value,SAVI03Value,SAVI04Value,SAVI05Value,
    SAVI06Value,SAVI07Value,SAVI08Value,SAVI09Value,SAVI10Value,
    BackgroundValueRed,BackgroundValueGreen,BackgroundValueBlue,
    BackgroundValueNIR,BackgroundValuePan,BackgroundValueNDVI ]
 
  # Join all pixel values in list to comma-separated string
  # -------------------------------------------------------
  OutPixelValueString = ','.join(
    [ str(PixelValueString) for PixelValueString in PixelValues ])

  # We only write to CSV those rows where ALL pixel
  # values are some valid valid, and not NaN or NoData (np.nan). 
  # If there is a np.nan in the row, return NONE, else, 
  # return the string.
  # ------------------------------------------------------------
  if 'nan' in str(OutPixelValueString):
    return None
  else:
    return OutPixelValueString

def ReadShapeFilePoints(ShapeFileName,ProjStr):
  '''function ReadShapeFilePoints( ShapeFileName, ProjStr ):
  This function reads the points in a shapefile (should be a POINTS 
  shapefile) and returns the points in two lists as part of a 
  dictionary. 

  Args:
    ShapeFileName (str): Name of shapefile. Should have .shp extension.
    ProjStr (str): Projection string (proj4) of imagery dataset.  
  Returns:
    dict: Dictionary with two lists[] of longitude(s) and latitudes(s).
  '''
  if not os.path.isfile(ShapeFileName):
    print('  \n Cannot open a shapefile. Not a file: "'+ShapeFileName+\
      '" . Try using full or relative path. Exiting ... \n')
    sys.exit(1)
  elif not ShapeFileName.endswith('.shp'):
    print('  \n Not a valid SHAPEFILE: "'+ShapeFileName+\
      '" . It should have a .shp extension. Exiting ... \n')
    sys.exit(1)
  else: pass

  try:
    ShapeFileReader = shapefile.Reader(ShapeFileName)
  except:
    return None

  # Read projection of shapefile
  # ----------------------------
  ShapeFileObject = ogr.Open(ShapeFileName)
  Layer = ShapeFileObject.GetLayer()
  SpatialRef = Layer.GetSpatialRef().ExportToWkt()

  # Make sure geometries in shapfile are 
  # of type POINT
  # -------------------------------------
  PointGeometries = ShapeFileReader.shapes()

  # Make sure input shapefile has at least 
  # one geometry.
  # ------------------------------------------
  if len(list(PointGeometries))<1:
    return None

  # Initialize two lists[] to hold 
  # longitude and latitude values in shapefile
  # ------------------------------------------
  Lons,Lats   = [],[]
 
  # Begin to iterate through geometries in 
  # the shapefile.
  # -------------------------------------------
  for p in PointGeometries:
    
    # If the geometry is not of type POINT, 
    # then continue.
    # -------------------------------------  
    if not p.shapeTypeName == 'POINT': continue
    
    # Get Longitude,Latitude from POINT geometry
    # ------------------------------------------
    Lon,Lat = p.points[0]

    # Below, we project the shapefile point from
    # its projection native to the shapefile 
    # to the same projection as the input
    # imagery set
    # -------------------------------------------
    InProjectionSRS = osr.SpatialReference()
    InProjectionSRS.ImportFromWkt( SpatialRef )

    OutProjSRS = osr.SpatialReference()
    OutProjSRS.ImportFromWkt( ProjStr )
    OutLon,OutLat = transform(Proj(InProjectionSRS.ExportToProj4()),
      Proj(OutProjSRS.ExportToProj4()),Lon,Lat)

    # Append lists of Longitude,Latitudes.
    # ------------------------------------
    Lons.append(OutLon)
    Lats.append(OutLat)

  return { 
    'Latitudes' : Lats,
    'Longitudes': Lons
  }

def WriteTrainingPointsCSV( OutDir,ImgDict,Shpfile,CSVWriter,ProjStr,IsBackground ):
  '''fucntion WriteTRainingPointsToCSV( OutDIr,ImgDict,Shpfile,CSVWriter,ProjStr,IsBackground ):
  This function takes in a shapefile, reads it set of Latitude and Longitude
  points, then converts those points from Latitude/Longitude (projected) 
  coordinate space to Row/Column space of the input imagery (in ImgDict). 
  For each point in the shapefile, a new line is written to the training data
  CSV dataset (CSVWriter object) containing all pixel values for all 
  of the input satellite imagery dataset (NDVI,SAVI,RGB,...).

  Args:
    OutDir (str): Output directory where "training" points CSV will be located.
    ImgDict (dict): Dictionary{} holding names of all raster imagery used for vegetation classification.
    Shpfile (str): Shapefile containing points whose corresponding pixel values will be written to CSV.
    CSVWriter (file): CSV text file object. Open for writing.
    ProjStr (str): Projection string for output projection.
    IsBackground (int): 1 or 0 , for vegetation and non-vegeation. Flag for final "Label" column in CSV.

  '''
  # Get path of gdaltransform GDAL command-line tool.
  # If it does not exist (NONE), then exit. 
  # --------------------------------------------------

  GDAL_Transform_Path = find_executable('gdaltransform')
  if GDAL_Transform_Path is None:
    print('  \n  Unable to find geotransform command tool. Exiting ... ' )
    sys.exit(1)

  # Get a dict{} holding two lists of 
  # latitudes and longitudes from input shapfile
  # --------------------------------------------
  LatLonsDict = ReadShapeFilePoints( Shpfile, ProjStr )
  if LatLonsDict is None:
    print('  \n    Failure to open following shapefile: '+Shpfile+'. Exiting ...')
    sys.exit(1)

  # Getlatitudes,longitudes as two lists[] of floats
  # ------------------------------------------------
  Lons = LatLonsDict['Longitudes']
  Lats = LatLonsDict['Latitudes']
  
  # Write Latitudes,Longitudes to CSV
  # ---------------------------------
  MapCoordinatesCSV = os.path.join( OutDir, 'xy.csv' )
  PointsCSV = open(MapCoordinatesCSV,'w')
  for X,Y in zip(Lons,Lats):
    PointsCSV.write('%s\n'%(str(X)+' '+str(Y)))
  PointsCSV.close()

  # Use Latitudes,Longitudes CSV to convert those 
  # points into a NEW CSV with Column,Row values 
  # from satellite imagery set. Here, we are 
  # essentially transforming a CSV from 
  # Latitude/Longitude into image Column/Row space
  # -----------------------------------------------
  RowsColsCSV = MapCoordinatesCSV.replace('xy','rowscols')
  TransformPointsCommand=' '.join([
    GDAL_Transform_Path,' -i ',ImgDict['pan'],' < ',MapCoordinatesCSV,' > ',RowsColsCSV])
  RunProcess(TransformPointsCommand)

  # Read set of Rows and Columns corresponding to 
  # points in input shapefile as a NROWSx2 array 
  # ---------------------------------------------
  RowsAndColumns = np.rint(np.loadtxt(RowsColsCSV)[:,0:2])
  RowsAndColumns = RowsAndColumns[RowsAndColumns.min(axis=1)>-1,:] 
  if RowsAndColumns.shape[0]<1:
    print('  \n    Unable to find any valid training data within geographic domain of input imagery.')
    sys.exit(1)

  # Clean up text-files containing Latitudes,Longitudes 
  # and Rows and columns ... we don't need them anymore
  # at this point. So delete them from disk.
  # ----------------------------------------------------
  os.remove(MapCoordinatesCSV)
  os.remove(RowsColsCSV)

  # determine flag label in CSV for points in shapefile
  # if "True" was passed-in, it is a value of "1" (i.e. trees)
  # else it is a zero
  # ----------------------------------------------------------
  if IsBackground:
    LabelColumnValue = 0
  else: 
    LabelColumnValue = 1

  # Open up panchromatic image file ... to get 
  # 2D dimensions of all input imagery.
  # ------------------------------------------
  DatasetPanchromatic = gdal.Open(ImgDict['pan'])
  NumRows, NumCols = DatasetPanchromatic.RasterYSize,DatasetPanchromatic.RasterXSize
  DatasetPanchromatic=None
  del DatasetPanchromatic

  # write pixel values representing points in shapefile to CSV
  # ----------------------------------------------------------
  for Point in range(RowsAndColumns.shape[0]):
   
    # get current column,row ~ (X,Y)
    # ------------------------------
    Column,Row = RowsAndColumns[Point][0],RowsAndColumns[Point][1]
    if Column>NumCols-1 or Row>NumRows-1: continue

    try:
      LonValue        = Lons[Point]
      LatValue        = Lats[Point]
      OutString  = GetPixelValuesAllImagery(Row,Column,ImgDict)
      if OutString is not None:
        OutString += ','+str(LabelColumnValue)
        CSVWriter.write('%s\n'%OutString)
    except Exception as e: print('WARNING: ' , str(e))

def CreateTrainingPointsCSV(BackgroundPtsShpfile,TargetPtsShpfile,ImgDict,OutDir,NoDataVal):
  '''function CreateTRainingPointsCSV(BackgroundPtsShpfile,TargetPtsShpfile,ImgDict,OutDir,NoDataVal):
  This is the "main" function for producing a CSV file that will hold 
  our "Training Data" used for classification (woods/forest) in the set of 
  satellite imagery. To this end, it writes a column header to this CSV 
  that contains the variables used for classification (SAVI, NDVI, RGB, ...). 
  It then iterates through all image files stored in the dictionary ImgDict{} 
  and gathers the relevant pixel values for those points stored in the two 
  shapefiles BackgroundPtsShpfile and TargetPtsShpfile. The final column 
  "label" will mark those rows that are a "tree" and 0s that are "not tree".

  Args:
    BackgroundPtsShpfile (str): Shapefile with POINTS for non-tree (non-vegetation).
    TargetPtsShapefile (str): Shapefile with POINTS for tree (vegetation/woods).
    ImgDict (dict): Python dictionary{} with all satellite imagery (NDVI,RGB,Pan,SAVI,...)
    OutDir (str): Output directory.
    NoDataVal (float): No Data value. Usually 0 or -9999.
  Returns:
    str: Name of CSV containing all pixel value training data.
  '''

  # Define out filename of CSV that will hold input "training" data
  # pixel values from all imagery (NDVI, SAVI,...). These pixel 
  # values will correspond to those points for the two input 
  # shapfiles for "Background" (NOT trees/veg.) and "Target" points
  # (Trees/woods/vegetation). 
  # ----------------------------------------------------------------
  OutnameCSV = os.path.join( 
    OutDir, 'TrainingPoints.csv')
  if os.path.isfile(OutnameCSV): 
    os.remove(OutnameCSV)

  # Open up this new CSV for writing. Write column
  # header.
  # ----------------------------------------------
  CSV = open(OutnameCSV,'w')
  HeaderStringCSV = 'NDVI,Pan,R,G,B,NIR,SAVI01,SAVI02,\
    SAVI03,SAVI04,SAVI05,SAVI06,SAVI07,SAVI08,SAVI09,SAVI10,\
    Background_Red,Background_Green,Background_Blue,Background_NIR,Background_Pan,\
    Background_NDVI,Label'
  HeaderStringCSV=HeaderStringCSV.replace(' ','')
  CSV.write('%s\n'%HeaderStringCSV)

  # use panchromatic image file (i.e. JPEG/Geotiff) to read 
  # in projection of input imagery 
  # -------------------------------------------------------
  PanFilename = ImgDict['pan']
  PanchromaticDataset = gdal.Open(PanFilename,gdal.GA_ReadOnly)
  ProjStr = PanchromaticDataset.GetProjectionRef()
  PanchromaticDataset = None
  del PanchromaticDataset

  # make sure gdaltransform (from GDAL) is available ... if not, 
  # we exit and return NONE.
  # --------------------------------------------------------------
  if find_executable('gdaltransform') is None:
    print('  \n    Please pass in filenames for Red,Green,Blue,NIR bands (JPEG/Geotiff).')
    return None

  # Use full satellite imagery set and shapefile for TARGET points
  # (i.e. points marking vegetation/trees/woods/forest)
  # to append/write CSV with corresponding training data with all
  # pixel values (as well as a value of "1" for the "Label" column
  # ---------------------------------------------------------------
  WriteTrainingPointsCSV(
    OutDir,ImgDict, 
    TargetPtsShpfile,CSV,ProjStr,False
  )

  # use satellite imagery and shapefile for background (i.e. not-trees)
  # to append/write CSV with training data for background
  # --------------------------------------------------------------------
  WriteTrainingPointsCSV(
    OutDir,ImgDict, 
    BackgroundPtsShpfile,CSV,ProjStr,True
  )
  CSV.close()
  return OutnameCSV
