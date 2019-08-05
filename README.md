###### GENERAL DESCRIPTION: VEGETATION AND FOREST CLASSIFICATION WITH MACHINE LEARNING AND PYTHON

       
       
       
###### COMMAND-LINE USAGE MESSAGE:
       
        
###### Sample Outputs

    NAME:
      VegetationClassification{.py}
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

      $ python VegetationClassification.py
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
      Last Updated: 2 August 2019

###### sample outputs:


![Alt text](https://i.imgur.com/JTC2v6L.png)

       Left: RGB (Red-Green-Blue) composite a Sentinel 2A Scene (Karavomilos and Sami, island of Kefalonia, Greece)
       Right: Python machine-learning forest (woods/vegetation) classification.

###### usage: 



###### @author: 
       Gerasimos Michalitsianos
       gerasimosmichalitsianos@gmail.com
       August 2019
