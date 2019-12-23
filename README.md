# Program to compare a grid of synthetic models and observations

# Description:
## Author: 
Jean-Loup Baudino
## Files and directories
 - Tools.py: contains functions and class used by the other programs
 - pipeline.py: compares the observation and models and generate a XML file storing the result
 - input_data.json: input file for pipeline.py
 - input: input directory containing the data to compare
    - obs: directory containing observations files and filter transmissions
    - grids: directory containing the grids of models 
       - cloud: Exo-REM models with clouds stored in XML containing spectrum and profiles
       - nocloud: Exo-REM models without cloud stored in XML containing spectrum and profiles
 - output: output directory where the results are saved
