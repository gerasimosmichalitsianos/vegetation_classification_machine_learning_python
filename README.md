###### GENERAL DESCRIPTION: VEGETATION AND FOREST CLASSIFICATION WITH MACHINE LEARNING AND PYTHON

    Satellite imagery products often come with several multispectral bands: Red, Green, Blue, NIR, and
    sometimes, a high-resolution panchromatic band. Some of these products may come from satellites 
    (orbiters) including the Landsat (i.e. NASA's Landsat 8, launched in 2013), as well as 
    Sentinel II (ESA's Copernicus program). 
       
    This program is meant to be called on the UNIX-like command-line, in which the user must 
    pass-in 1-band (1 image layer or array) georeferenced image filenames (i.e. Geotiffs or JPEG2000, 
    PNG with "world file," etc.), including those for the Red, Green, Blue, and NIR bands. These are 
    required. The user must also pass-in two filename(s) for two shapefiles. One of these shapefiles 
    will hold points marking trees/forest/vegetation, and the other holidng points for 
    non-trees/non-vegetation/non-forest. Finally, the user must also pass-in the number of "trees" 
    (integer) for the number of "trees" used in Python/Sklearn's ExtraTreesClassifier() method
    (usually between 1 and 15). Optionally, this program allows the user to pass-in a NOData (nan)
    value (usually -9999 or 0) at the command line, as well as a desired output directory.
       
    This program will do the following:
      (1) Computes a panchromatic image Geotiff if necessary (if user did not pass this at command-line)
          using the formula (Red+Green+Blue+NIR)/4.0.
      (2) Resamples the panchromatic band (usually by downsampling using nearest-neighbor), if necessary, 
          (i.e. if it was passed at the command-line) to the same resolution and dimensions as the input 
          multispectral RGB,NIR imagery. 
      (3) Computes a supporting set of georeferenced (GDAL) raster satellite imagery files, including 
          Normalized Difference Vegetation Index (NDVI), Soil-Adjusted NDVI (SAVI) for L = 0.1,0.2,...1.0 
          (see https://wiki.landscapetoolbox.org/doku.php/remote_sensing_methods:soil-adjusted_vegetation_index),
          as well as "background" imagery (Gaussian-filtered) for the Red,Green,Blue,NIR,Panchromatic, and 
          NDVI band and/or band combinations. Also creates an RGB Geotiff mosaic. 
      (4) Reads both vegetation and non-vegetation POINT shapefiles and converts these latitude/longitude
          projected points into Row/Column space for the input NIR/RGB imagery passed-in. Gathers corresponding
          pixel values from all imagery (RGB,NIR,Panchromatic, "Backround" imagery, NDVI, SAVI) and writes all
          pixel values to a CSV file. Values of 1 mark "trees/vegetation" and 0s mark "non-trees/non-vegetation" 
          in the right-most column of this CSV ("Label"). Each row in this CSV represents one POINT from one 
          of the TWO shapefile(s) passed-in at command-line. This CSV hence contains "Training Data" 
          used for vegetation classification.
      (5) Uses CSV from (5), as well as Python machine-learning tools from Sklearn and Pandas, to 
          write a PNG and Geotiff containing a final forest (woods/trees) classification for the 
          imagery set. 1s are forest (vegetation), 0s are non-forest (i.e. not vegetation). 
             
    For increased accuracy, ensure both input shapefile(s) have more points (i.e. training data).

###### PYTHON VERSION

    Supports Python 3.x.
         
###### FURTHER EXPLANATION OF METHODOLOGY
 
    See see a simplifed version of this code, visit this site:
    https://gerasimosmichalitsianos.wordpress.com/2019/05/26/classification-of-forest-vegetation-in-satellite-imagery-using-machine-learning-and-python/
       
    This example blog entry shows how the code works, assuming all "training imagery" (NDVI,SAVI,...)
    has already been created, along with the CSV containing the "training data" of 
    all relevant pixel values and tree/non-tree label (1 or 0 in the final "Label" column).
    
###### INSTALLATION

    First check out the code and change into the directory:
    
    $ git clone https://github.com/gerasimosmichalitsianos/vegetation_classification
    $ cd vegetation_classification/
    
    Then build the image using Dockerfile:
    
    $ docker build -t vegetationclassify .
    
###### GENERAL USAGE

    To use this program, run it at command-line with the 
    following possible arguments:

    $ VegetationClassification
      
    Command-Line Options:
      { --help, -h }
        Display this help usage message
      { --ntrees, --numtrees, --numbertrees }
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
    
###### EXAMPLE USAGE

    $ git clone https://github.com/gerasimosmichalitsianos/vegetation_classification
    $ cd vegetation_classification/
    $ docker build -t vegetationclassify .
    
    $ DIR=/home/gmichali/HONG_KONG_SENTINEL
    $ red=$DIR/T49QHE_20190125T030011_B04_10m_subset.jp2
    $ green=$DIR/T49QHE_20190125T030011_B03_10m_subset.jp2 
    $ blue=$DIR/T49QHE_20190125T030011_B02_10m_subset.jp2 
    $ NIR=$DIR/T49QHE_20190125T030011_B08_10m_subset.jp2
    
    $ treeshapefile=$DIR/hk_trees.shp
    $ nontreeshapefile=$DIR/hk_not_trees.shp

    $ docker run -v $DIR:$DIR vegetationclassify --red $red \
      --green $green \
      --blue $blue --nir $NIR \
      --trees $treeshapefile \
      --nontrees $nontreeshapefile --ntrees 3 --nodata 0

###### FUNCTIONALITY

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

###### SAMPLE OUTPUTS:

![Alt text](https://i.imgur.com/usfzp1y.png)

    Left: RGB (Red-Green-Blue) composite a 10m resolution Sentinel 2A Scene (Hong Kong SAR)
    Right: Python machine-learning forest (woods/vegetation) classification.
       
![Alt text](https://i.imgur.com/corJyDg.png)

    Left: RGB (Red-Green-Blue) composite a 30m resolution Landsat 8 Scene (Paliki Peninsula, island of Kefalonia, Greece)
    Right: Python machine-learning forest (woods/vegetation) classification.

![Alt text](https://i.imgur.com/JTC2v6L.png)

    Left: RGB (Red-Green-Blue) composite a 10m resolution Sentinel 2A Scene (Karavomilos and Sami, island of Kefalonia, Greece)
    Right: Python machine-learning forest (woods/vegetation) classification.

###### @AUTHOR:

    Gerasimos Michalitsianos
    gerasimosmichalitsianos@gmail.com
    January 2021
