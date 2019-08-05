import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr,gdal,gdalconst
from matplotlib.pylab import *

def ResampleImage( SourceImageFilename, SourceDataset, DestinationDataset, OutFileName, Interp ):
  '''function resample( srcImageFilename,sourceDataset,dstDataset,outname,interp):
  This function resamples a low-resolution multispectral Geotiff to larger 
  dimensions by means of the "interp" method (i.e. gdalconst.GRA_Cubic) passed
  into this function. Ultimately, this function resamples or resizes the  
  multispectral geotiff dataset, referred to by "sourceDataset" (and srcImageFilename)
  to the same dimensions as the panchromatic Geotiff image Geotiff file by 
  means of an interpolation method (i.e. bicubic resampling). 

  Args:
    SourceImageFilename (str): source (low-res.) multispectral Geotiff filename. 
    sourceDataset (osgeo.gdal.Dataset): input multispectral GDAL dataset object.
    dstDataset (osgeo.gdal.Dataset): destination (high-res.) panchromatic dataset object. 
    outname (str): name of outputted resampled Geotiff
    interp (int): GDAL interpolation method (i.e. gdalconst.GRA_Cubic) 
  Returns:
    str: Name of resmapled image filename.
  '''

  # If output File dataset already exists, then remove it 
  # from disk 
  # --------------------------------------------------------
  if os.path.isfile( OutFileName ):
    os.remove( OutFileName )

  # Get source (i.e. source or low-resolution) image 
  # projection, geotransfrom, and number of bands
  # --------------------------------------------------------
  SourceProjection        = SourceDataset.GetProjection()
  SourceGeotransform      = SourceDataset.GetGeoTransform()
  SourceNumRasters        = SourceDataset.RasterCount

  # Get Projection and Geotransform for "destination" image
  # --------------------------------------------------------
  DestinationProjection   = DestinationDataset.GetProjection()
  DestinationGeotransform = DestinationDataset.GetGeoTransform()
  DestinationNRows        = DestinationDataset.RasterYSize
  DestinationNCols        = DestinationDataset.RasterXSize

  # remove output file if it already exists 
  # ---------------------------------------
  DestinationDataset = gdal.GetDriverByName('GTiff').Create( 
    OutFileName, DestinationNCols,DestinationNRows,SourceNumRasters,gdalconst.GDT_Float32)
  DestinationDataset.SetGeoTransform(DestinationGeotransform)
  DestinationDataset.SetProjection(DestinationProjection)
  
  gdal.ReprojectImage(SourceDataset, 
    DestinationDataset, SourceProjection, SourceProjection,Interp)
  dst_ds=None
  del dst_ds
  
  Pan = DestinationDataset.GetRasterBand(1).ReadAsArray()
  return ( OutFileName, Pan ) 

def RunProcess( cmd ):
  '''function RunProcess( cmd ):
  This fucntion executes a command in the shell using 
  Python's subprocess module.

  Args:
    cmd (str): Command to be executed.
  '''
  proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
  proc.wait()

def WritePNG( OutFileName, OutDataArray ):
  '''function WritePNG( OutFileName, OutDataArray ):
  This function writes a 2D NumPy array to a PNG file.

  Args:
    OutFileName (str): Name of output PNG.
    OutDataArray (numpy.ndarray): Output 2D NumPy array.
  '''
  ioff()
  plt.title('')
  plt.imshow( OutDataArray, cmap=plt.cm.gray )
  plt.savefig( OutFileName, dpi=200 )
  plt.close()

def WriteGeotiff( ReferenceDataset, OutFileName, OutDataArray ): 
  '''function WriteGeotiff( 
  This function writes a Geotiff. To this end, it uses an input 
  GDAL dataset (as a reference) to get a projection string and
  geotransform. It writes the output data as a float32 array.

  Args:
    ReferenceDataset (osgeo.gdal.Dataset): 
      Reference GDAL dataset to get projection and geostransform.
    OutFileName (str): output filename Geotiff string.
    OutArrayData (numpy.ndarray): 2D NumPy array to be written to Geotiff.
  Returns: 
    None
  '''

  # If output file already exists on-disk, remove it.
  # -------------------------------------------------
  if os.path.isfile(OutFileName): os.remove(OutFileName)

  # Open up output Geotiff driver. Get output 
  # dimensions using reference dataset.
  # -----------------------------------------
  driver = gdal.GetDriverByName('GTiff')
  nrows  = ReferenceDataset.RasterYSize
  ncols  = ReferenceDataset.RasterXSize

  # Create output Geotiff dataset. Set projection and 
  # geotransform. 
  # -------------------------------------------------
  dst_ds = driver.Create( OutFileName,ncols,nrows,1,gdal.GDT_Float32 )

  # Set projection and geotransform.
  # --------------------------------

  dst_ds.SetGeoTransform( ReferenceDataset.GetGeoTransform() )
  dst_ds.SetProjection( ReferenceDataset.GetProjection() )
  dst_ds.GetRasterBand(1).WriteArray( OutDataArray )
  dst_ds=None
  del dst_ds
