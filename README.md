# Program to compare a grid of synthetic models and observations

# Description:
## Author: 
Jean-Loup Baudino
## Scientific use
Please cite Baudino et al. 2015, 2017 and Charnay et al. 2018 (full reference in the export-bibtex.bib)
## How to
The easy way to use this program is to pass by the Jupyter notebook PlotResults.ipynb
## Files and directories
 - Tools.py: contains functions and class used by the other programs
 - pipeline.py: compares the observations and models and generate a XML file storing the results
 - input_data.json: input file for pipeline.py
 - input: input directory containing the data to compare
    - obs: directory containing observations files and filter transmissions
    - grids: directory containing the grids of models 
       - cloud: Exo-REM models with clouds stored in XML containing spectrum and profiles
       - nocloud: Exo-REM models without clouds stored in XML containing spectrum and profiles
 - output: output directory where the results are saved
 - PlotResults.ipynb: do the comparison between models and observation and plot the result as grids or histograms
 - export-bibtex.bib: contains the full reference list of papers relative to this work