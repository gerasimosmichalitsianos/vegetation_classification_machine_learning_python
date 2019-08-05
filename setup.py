from distutils.core import setup
setup(
    name='VegetationClassification',
    version='1.0.0',
    scripts=['bin/VegetationClassification.py','bin/ImageClassification.py','bin/Misc.py','bin/TrainingPoints.py','bin/TrainingImagery.py',], 
    license='MIT',
    include_package_data=True, 
    long_description=open('README.md').read(),
    entry_points = {
        'console_scripts': ['VegetationClassification=VegetationClassification:main'],
    }
)
