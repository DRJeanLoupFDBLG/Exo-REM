'''
author:
       Jean-Loup Baudino
date:
     11/01/2020
description:
            generate a sqlite database of the grid.
'''
import sqlite3
from numpy import loadtxt


##############################
# Parameters
##############################
dir_input="input/grids/cloud/"
##############################


tabeGrille=loadtxt(dir_input+"ListXmlFiles.info",
dtype=[("logg",float),("teff",float),("z",float),("loc",'S90')])



conn = sqlite3.connect('Exo-REM.db')
print ("Opened database successfully")

conn.execute('''CREATE TABLE GRIDCLOUD
         (ID INT PRIMARY KEY,
         GRAVITY      FLOAT    NOT NULL,
         TEMPERATURE  FLOAT    NOT NULL,
         METALLICITY  FLOAT    NOT NULL,
         LOC          TEXT     NOT NULL,
         CHI2         FLOAT,
         RADIUS       FLOAT,
         MASS         FLOAT
         );''')
print ("Table created successfully")

n=1
for model in tabeGrille:
    conn.execute(
        "INSERT INTO GRIDCLOUD (ID,GRAVITY,TEMPERATURE,METALLICITY,LOC) \
        VALUES (?, ?, ?, ?, ?)", (n, model['logg'], model['teff'], model['z'], model['loc'].decode('ascii'))
        )
    n+=1

conn.commit()
print("Records created successfully")

conn.close()