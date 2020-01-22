"""
Desc: Library of tools to interact with models and observations
Auth: Jean-Loup Baudino
Devl: Jean-Loup Baudino (JLB), Ryan Garland (RG)
Date: 16/12/19
Cont:    
    * gaussian1D(height, x, center_x, width_x)
    * ConstrFilter(loc, flux ,tabWavGeneral)
    * RadiusCalc (logg, M)
    * xi2 (calc,obs,errObs)
    * MassCalc (logg, R)
    * PlusMinus (Tab)
    * RminXsig(Rrange, xi2max, obs, synth)
    * RmaxXsig(Rrange, xi2min, obs, synth)
    * GenMmwamu(NameArray)
    * altitud(NameArray,vmr,g, mass,rad,P,T)
Hist: 
    * 16/12/19 creation From LocTools.py JLB
"""

import numpy as np
import math
from astropy.constants import k_B, R, G, M_jup, R_jup, R_sun
from astropy import units as u
import csv
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from pylab import savefig, contourf, meshgrid, show, figure, colorbar, rc, xlabel, ylabel, savefig, arange, contour, title
from math import pi
from scipy.stats import chi2
from scipy.interpolate import griddata
from astropy.table import Table
import sqlite3


def gaussian1D(height, x, center_x, width_x, logfile=[]):
    """Returns a gaussian function with the given parameters"""
    if width_x <= 0 :
        logfile.add({'Error':'width of gaussian not valid', 
        'Function':'gaussian1D', 
        'Input':[height, x, center_x, width_x,
        ]})
        Result = float('nan')
    else:
        width_x = float(width_x)
        Result = height*np.exp(-((center_x-x)/width_x)**2)
    return Result, logfile

def ConstrFilter(loc, Trans ,tabWavGeneral, logfile=[]):
    """
    author:
        JLB
    Def:
        Extrapolate a filter function at a given resolution
    Input:
        * loc = list of wavelength of the filter
        * Trans = list of transmission of the filter
        * tabWavGeneral = list of wavelength of the spectrum
    Output:
        transmission (non-normalised)
    """
    loc=np.array(loc)
    Trans=np.array(Trans)
    tabWavGeneral=np.array(tabWavGeneral)
    if min(loc)>max(tabWavGeneral) or max(loc)<min(tabWavGeneral) :
        logfile.add({'Error':'filter not included in the spectrum', 
        'Function':'ConstrFilter', 
        'Input':[min(loc), max(tabWavGeneral), max(loc), min(tabWavGeneral),
        ]})
        Result = np.zeros(len(tabWavGeneral))
    else:
        FiltreTrans=np.zeros(len(tabWavGeneral))
        for i in range(0,len(tabWavGeneral)):
            if (tabWavGeneral[i] > min(loc)) and (tabWavGeneral[i] < max(loc)):
                FiltreTrans[i]=sum(
                    gaussian1D(
                        Trans/np.sqrt(2.*np.pi), 
                        loc, 
                        tabWavGeneral[i], 
                        ((10000./((10000./tabWavGeneral[i])-20))-tabWavGeneral[i])/(2.*np.sqrt(2*np.log(2))),
                        logfile=logfile)[0])/sum(Trans)
        Result = FiltreTrans
    return Result, logfile

def RadiusCalc (logg, M, logfile=[]):
    """
    author:
        JLB
    Def:
        compute radius for a given gravity and mass
    Input:
        * logg : log10(g[cgs])
        * M : mass [Mjup]
    Output:
        radius in Rjup 
    """
    if logg < 2 :
        logfile.add({'Error':'gravity inferior to 100 cgs', 
        'Function':'RadiusCalc', 
        'Input':[logg, M,
        ]})
        Result = float('nan')
    else:
        Result = np.sqrt((M*M_jup)/((10**(logg-2))/(G)))/R_jup
    return Result, logfile

def xi2 (calc,obs,errObs, logfile=[]) :
    """
    author:
        JLB
    Def:
        compute the chi2 between model and observation
    Input:
        * calc : synthetic flux (same size as the other)
        * obs : observation flux (same size as the other)
        * errObs : uncertainties on the observations (same size as the other)
    Output:
        chi2
    """
    if len(errObs[errObs <= 0]) > 0:
        logfile.add({'Error':'negative error', 
        'Function':'xi2', 
        'Input':[calc,obs,errObs,
        ]})
        Result = float('nan')
    else:
        Result = sum(((calc-obs)**2)/(errObs**2.))
    return Result, logfile

def MassCalc (logg, R, logfile=[]):
    """
    author:
        JLB
    Def:
        compute mass for a given gravity and radius
    Input:
        * logg : log10(g[cgs])
        * R : radius [Rjup]
    Output:
        mass in Mjup
    """
    if logg < 2 :
        logfile.add({'Error':'gravity inferior to 100 cgs', 
        'Function':'MassCalc', 
        'Input':[logg, R,
        ]})
        Result = float('nan')
    else:
        Result = ((((u.m/(u.s**2))*10.**(logg-2))/(G)) * ( R*R_jup )**2) / M_jup
    return Result, logfile

def PlusMinus (Tab, logfile=[]):
    """
    author:
        JLB
    Def: 
        give the maximal difference value in a array compare to the mean value
    Input:
        Tab: an array
    Output:
        max difference compare to mean value
    """
    if len(Tab) == 0 :
        logfile.add({'Error':'empty array', 
        'Function':'PlusMinus', 
        'Input':[Tab,
        ]})
        Result = float('nan')
    else:
        Result = max(abs(np.mean(Tab)-min(Tab)),abs(max(Tab)-np.mean(Tab)))       
    return Result, logfile

def RminXsig(Rrange, xi2max, obs, synth, logfile=[]):
    """
    author:
        JLB
    Def:
        When a planetary radius is too big, give the radius minimal that keep the chi2 < a given value
    Input:
        * Rrange: array of targeted radius range
        * xi2max: maximal chi2 that stop the decrease of the radius
        * obs: array with wavelength, flux, err flux
        * synth: array with synthetic flux
    Output:
        minimal radius in conditions
    """
    if len(obs[:,2][obs[:,2] <= 0]) > 0:
        logfile.add({'Error':'negative error', 
        'Function':'RminXsig', 
        'Input':[Rrange, xi2max, obs, synth,
        ]})
        Result = float('nan')
    else:    
        R=max(Rrange)
        Err=xi2(synth*R**2,obs[:,1],obs[:,2])
        while Err < xi2max:
            R=R-0.05
            Err=xi2(synth*R**2,obs[:,1],obs[:,2])
        Result = R+0.05
    return Result, logfile

def RmaxXsig(Rrange, xi2min, obs, synth, logfile=[]):
    """
    author:
        JLB
    Def:
        When a planetary radius is too smal, give the maximal radius that keep the chi2 < a given value
    Input:
        * Rrange: array of targeted radius range
        * xi2max: maximal chi2 that stop the decrease of the radius
        * obs: array with wavelength, flux, err flux
        * synth: array with synthetic flux
    Output:
        maximal radius in conditions
    """
    if len(obs[:,2][obs[:,2] <= 0]) > 0:
        logfile.add({'Error':'negative error', 
        'Function':'RmiaxXsig', 
        'Input':[Rrange, xi2max, obs, synth,
        ]})
        Result = float('nan')
    else:    
        R=min(Rrange)
        Err=xi2(synth*R**2,obs[:,1],obs[:,2])
        while Err < xi2min:
            R=R+0.05
            Err=xi2(synth*R**2,obs[:,1],obs[:,2])
        Result = R-0.05
    return Result, logfile

def GenMmwamu(NameArray, logfile = []):
    #JLB Aug 2018
    #######
    #Input: an array of hte name of the molecules retrieved
    #   Read a file with 3 columns: Nemesis Id, name of the molecule, molar mass
    #Output: an array with the molar mass of te given molecules
    ########

    RefCat=np.genfromtxt("input/mmw.amu", dtype={'names': ('Id', 'Name', 'MolarMass'),
        'formats': ('i', 'U10', 'f')}, delimiter=",", skip_header=1)
    mmwamu=np.zeros(len(NameArray))
    errPart=[]
    for n in range(len(NameArray)):
        if np.isin(NameArray[n], RefCat["Name"]):        
            for item in enumerate(RefCat["Name"]):
                if NameArray[n]==item[1]:
                    mmwamu[n]= RefCat["MolarMass"][item[0]]
        else :
            errPart.append(NameArray[n])
    if len(errPart) > 0 :
        logfile.append({'Error':'name of molecules non recognised', 
        'Function':'NameArray', 
        'Input':[errPart,
        ]})
        Result = float('nan')
    else:         
        Result = mmwamu
    return Result, logfile

def altitud(NameArray, vmr, g, mass, rad, P, T, logfile = []):
    #RG 2017 mod JLB Nov 2018
    ######
    #Input: NameArray,vmr,g,rad,P,T
    #  Compute H
    #Output: H[km]
    #######    
    amu = 1.67e-27
    nPressLvl=len(P)            
    
    # calculate height profile
    mmw = np.zeros(nPressLvl)
    mmwamu = GenMmwamu(NameArray, logfile=logfile)
    if mmwamu == float('nan') or g*mass*rad <= 0 or len(P) == 0 or len(T)==0:
        logfile.append({'Error':'problem input', 
        'Function':'altitud', 
        'Input':[logfile.last(),vmr,g, mass,rad,P,T,
        ]})
        Result = float('nan')
    else :
        for i in range(nPressLvl):
            mmw[i] = np.sum(mmwamu * vmr[:,i]) * amu
        H = np.zeros(nPressLvl)
        SH = np.zeros(nPressLvl)
        Hdiff=np.zeros(nPressLvl) + R_jup.to(u.km).value   
        while not all(Hdiff < 0.01):
            #constant gravity
            radh = rad*R_jup.to(u.km).value + H
            SCALE = (k_B.to(u.kg*(u.km**2)/(u.K*u.s**2))*T) /(((mmw*u.g).to(u.kg))*g)
            for i in range(nPressLvl):
                if (i != 0):
                    SH[i]=0.5*(SCALE[i].value+SCALE.value[i-1])
                    H[i] = H[i-1] - SH[i] * np.log(P[i]/P[i-1])
            Hzero = H
            # variable gravity
            radh=rad*R_jup.to(u.km).value+H
            g = G.to((u.km**3)/(u.kg*u.s**2))*(mass*M_jup)/(radh*u.km)**2
            SCALE = (k_B.to(u.kg*(u.km**2)/(u.K*u.s**2))*T) /(((mmw*u.g).to(u.kg))*g)
            for i in range(nPressLvl):
                if (i != 0):
                    SH[i]=0.5*(SCALE[i].value+SCALE[i-1].value)
                    H[i] = H[i-1] - SH[i] * np.log(P[i]/P[i-1])
            Hdiff=abs(H-Hzero)
    Result = H/1000.
    return Result, logfile

#############

def gaus(x,a,x0,sigma, logfile = []):
    """
    gaussian function
    https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fitgaus(nbins, bins, logfile = []):
    """
    return fit if gaussian on historgram
    https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    """
    x_bins=np.zeros(len(nbins))
    for npos in range(0,len(nbins)):
        x_bins[npos]=(bins[npos]+bins[npos])/2
        mean = sum(nbins*x_bins)/sum(nbins)
        sigma = (max(x_bins)-min(x_bins))/2.# sum(nbins*(x_bins-mean)**2.)/sum(nbins)
    return curve_fit(gaus,x_bins,nbins,p0=[1,mean,sigma])

def nofitgaus(nbins, bins, logfile = []):
    """
    return fit if gaussian on historgram
    https://stackoverflow.com/questions/19206332/gaussian-fit-for-python
    """
    x_bins=np.zeros(len(nbins))
    for npos in range(0,len(nbins)):
        x_bins[npos]=(bins[npos]+bins[npos])/2
        mean = sum(nbins*x_bins)/sum(nbins)
        sigma = (max(x_bins)-min(x_bins))/2.
    return [mean,sigma]



def plotGrid(param, fit, dir_output,
             R_cond, M_cond,
             NbDegreeFree,
             FitType,typeCloud,planetName,
             z=["05", "1", "5"], logfile = []
             ):
    """
    author:
        JLB
    date:
        01/07/2019
    description:
        plot a grid gravity vs Teff and sigma color code
    """
    radmin_c=(fit.field('radius')>min(R_cond))
    radmax_c=(fit.field('radius')<max(R_cond))
    massmin_c=(fit.field('mass')>min(M_cond))
    massmax_c=(fit.field('mass')<max(M_cond))
    in_Rad=np.logical_and(radmin_c,radmax_c)
    in_Mas=np.logical_and(massmin_c,massmax_c)
    c=np.logical_and(in_Rad,in_Mas)

    for zi in z:
        figure(figsize=(40,20),dpi=100)

        rc('xtick', labelsize=40)
        rc('ytick', labelsize=40)                
        if zi=="05":
            z_par=(param.field("metalicity")[c]<0.4)
        elif zi=="1":
            z_par=(param.field("metalicity")[c]==1)
        elif zi=="5":
            z_par=(param.field("metalicity")[c]>3)
        else:
            print("unknown metallicity")
        
        loggf=np.log10(np.array(param.field("gravity"))[c])[z_par]
        Tefff=np.array(param.field("Teff"))[c][z_par]
        xi2f=np.array(fit.field("chi2final"))[c][z_par]
        massf=np.array(fit.field("mass"))[c][z_par]/M_jup.value

        xi = np.linspace(min(Tefff),max(Tefff),1+int((max(Tefff)-min(Tefff))/50))
        yi = np.linspace(min(loggf),max(loggf),1+int((max(loggf)-min(loggf))/0.1))
        X,Y=meshgrid(xi,yi)
        
        levels10 = [0.68, 0.75, 0.89, 0.94, 0.96]
        prob=[r"$1\sigma$",r"$2\sigma$",r"$3\sigma$",r"$4\sigma$",r"$5\sigma$"]

        contourf(X,Y,griddata((Tefff,loggf), xi2f, (X,Y), method='linear'), 
                levels=chi2.isf(np.ones(len(levels10))-levels10, NbDegreeFree+1.), cmap=cm.coolwarm, extend="both") 
        cb=colorbar(shrink=0.4)
        cb.ax.set_yticklabels(prob)

        ylabel("$\log(g[cgs])$", size=60, labelpad=20)
        xlabel("$T_\mathrm{eff}$[K]", size=60, labelpad=20)

        savefig(str(dir_output[0])+"ExoREMclassicXi2map"+str(FitType[0])+"_"+str(typeCloud)+"_z"+str(zi)+"_"+str(planetName[0])+".pdf", format="pdf", dpi=600)
    show()
    return()


def plotHisto(datasets, grids, paramJup, param, sig_max, R_cond, M_cond, 
        dir_output, planetname, ER_input, NbDegreeFree, colors_planets, 
        FitType,typeCloud,planetName,
        option=["y","y","y","y","y"], logfile = []
        ):
    """
    author:
        JLB
    date:
        01/07/2019
    desciption:
        show historgram of physical results
    input:
        * datasets: array of observations names
        * grids: array of grids
        * paramJup: result of fit
        * sig_max: max sigma to take into-account
        * R_cond: array with min and max radius in [m]
        * M_cond: array with min anx max mass in [kg]
        * option: array of y or n to compute histogram of : Teff, log10(g[cgs]), Mass[Mjup], Rad(Rjup) andmetallicity (0.3-1-3 x solaire)

    """
    fig, axs = plt.subplots(option.count("y"), 1 , figsize=(15, 20),
                                    tight_layout=True)

    for p in range(len(datasets)):
        titleTeff=""
        titleLogg=""
        titleMass=""
        titleRadius=""

        for m in range(len(grids)):
            n=p+m

            if str(sig_max)=="All":
                xi2_c=(paramJup[n]["chi2final"])>0
            else:
                xi2_c=chi2.cdf((paramJup[n]["chi2final"])/NbDegreeFree, NbDegreeFree)<=sig_max
            radmin_c=(paramJup[n]).field('radius')>min(R_cond)
            radmax_c=(paramJup[n]).field('radius')<max(R_cond)
            massmin_c=(paramJup[n]).field('mass')>min(M_cond)
            massmax_c=(paramJup[n]).field('mass')<max(M_cond)
            in_Rad=np.logical_and(radmin_c,radmax_c)
            in_Mas=np.logical_and(massmin_c,massmax_c)
            in_RandMas=np.logical_and(in_Rad,in_Mas)
            in_cond=np.logical_and(xi2_c,in_RandMas)

            nplot=0
            if option[0]=="y":
                nbins, bins, patches = axs[0].hist(paramJup[n][in_cond]['Teff'], histtype='step', fill=False, bins=15, weights=1./paramJup[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleTeff=titleTeff+str(int(popt[0]))+" +- "+str(int(abs(popt[1])))+" K\n"
                axs[0].set_xlabel("$T_\mathrm{eff}$[K]")
                axs[0].set_ylabel("Normalised count with 1/$\chi^2$ coefficient")
                nplot=nplot+1
            
            if option[1]=="y":
                nbins, bins, patches = axs[nplot].hist(np.log10(paramJup[n][in_cond]['gravity'].quantity.value),histtype='step', fill=False, bins=10, weights=1./paramJup[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleLogg=titleLogg+str('%.1f' %float(popt[0]))+" +- "+str('%.1f' %float(abs(popt[1])))+"\n"
                axs[nplot].set_xlabel("$\log$(g[cgs])")
                nplot=nplot+1
            
            if option[2]=="y":
                nbins, bins, patches = axs[nplot].hist(paramJup[n][in_cond]['mass']/M_jup.value,histtype='step', fill=False, range=[min(M_cond/M_jup.value),max(M_cond)/M_jup.value], bins=15, weights=1./paramJup[n][in_cond]['chi2final'], color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                nplot=nplot+1
            
            if option[3]=="y":
                nbins, bins, patches = axs[nplot].hist(paramJup[n][in_cond]['radius']/R_jup.value,histtype='step', fill=False, range=[0.6,2],  bins=15, weights=1./paramJup[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleRadius=titleRadius+str('%.1f' %float(popt[0]))+" +- "+str('%.1f' %float(abs(popt[1])))+"Rjup\n"
                nplot=nplot+1
            
            if option[4]=="y":
                axs[nplot].hist(paramJup[n][in_cond]['metalicity'],histtype='step', fill=False, log=True, bins=15, weights=1./paramJup[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
        '''
        axs[0].set_title(titleTeff)
        axs[1].set_title(titleLogg)
        axs[2].set_title(titleMass)
        axs[3].set_title(titleRadius)
        '''
        ylabel("Normalised count with 1/$\chi^2$ coefficient")
    fig.savefig(str(dir_output[0])+"HistogramsWthCdt_"+str(sig_max)+str(FitType[p])+"_"+str(typeCloud)+"_"+str(planetName[0])+".pdf", format="pdf", dpi=600)
    #plt.show()
    
    rad_f=paramJup[n][in_cond]['radius']/R_jup.value
    mas_f=paramJup[n][in_cond]['mass']/M_jup.value
    Tef_f=paramJup[n][in_cond]['Teff']
    z_f=paramJup[n][in_cond]['metalicity']
    lgg_f=np.log10(paramJup[n][in_cond]['gravity'])
    chi_f=paramJup[n][in_cond]['chi2final']
    avTef=sum(Tef_f*chi_f)/sum(chi_f)
    print(int(avTef),"+-",int(avTef-min(Tef_f)+max(Tef_f)-avTef)/2)
    avlgg=sum(lgg_f*chi_f)/sum(chi_f)
    print(int(avlgg),"+-",int(avlgg-min(lgg_f)+max(lgg_f)-avlgg)/2)
    avz=sum(z_f*chi_f)/sum(chi_f)
    print(int(avz),"+-",int(avz-min(z_f)+max(z_f)-avz)/2)

    returnTab=paramJup[n][in_cond]
    return(returnTab)

def CompareResults(datasets, grids, sig_max, R_cond, M_cond,
                dir_output, planetname, ER_input, dirNem, 
                ExoREMarray,OEarray,sig,FitType,
                option=["y","y","y","y","y"], logfile = []
                ):
    """
    author:
        JLB
    date:
        09/07/2019
    description:
        use results from Exo-REM and NEMESIS OE to generate comparative histogram
    inputs:
        * datasets
        * grids
        * sig_max
        * R_cond
        * M_cond
        * dir_output
        * planetname
        * ER_input
        * dirNem
        * option
        * ExoREMarray
        * OEarray
    """
    # histogram molecules
    fig, axs = plt.subplots(5, 1 , figsize=(15, 20),
            tight_layout=True)

    for p in range(len(datasets)):
        titleTeff=""
        titleLogg=""
        titleMass=""
        titleRadius=""

        for m in range(len(grids)):
            n=p+m
            if str(sig_max)=="All":
                xi2_c=(ExoREMarray[n]["chi2final"])>0
            else:
                xi2_c=chi2.cdf((ExoREMarray[n]["chi2final"])/NbDegreeFree, NbDegreeFree)<=sig_max
            radmin_c=(ExoREMarray[n]['radius'])>min(R_cond)
            radmax_c=(ExoREMarray[n]['radius'])<max(R_cond)
            massmin_c=(ExoREMarray[n]['mass'])>min(M_cond)
            massmax_c=(ExoREMarray[n]['mass'])<max(M_cond)
            in_Rad=np.logical_and(radmin_c,radmax_c)
            in_Mas=np.logical_and(massmin_c,massmax_c)
            in_RandMas=np.logical_and(in_Rad,in_Mas)
            in_cond=np.logical_and(xi2_c,in_RandMas)

            nplot=0
            if option[0]=="y":
                nbins, bins, patches = axs[0].hist(ExoREMarray[n][in_cond]['Teff'], histtype='step', fill=False, bins=15, weights=1./ExoREMarray[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleTeff=titleTeff+str(int(popt[0]))+" +- "+str(int(abs(popt[1])))+" K\n"
                nplot=nplot+1

            if option[1]=="y":
                nbins, bins, patches = axs[nplot].hist(np.log10(ExoREMarray[n][in_cond]['gravity'].quantity.value),histtype='step', fill=False, bins=10, weights=1./ExoREMarray[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleLogg=titleLogg+str('%.1f' %float(popt[0]))+" +- "+str('%.1f' %float(abs(popt[1])))+"\n"
                nplot=nplot+1

            if option[2]=="y":
                nbins, bins, patches = axs[nplot].hist(ExoREMarray[n][in_cond]['mass']/M_jup.value,histtype='step', fill=False, range=[min(M_cond/M_jup.value),max(M_cond)/M_jup.value], bins=15, weights=1./ExoREMarray[n][in_cond]['chi2final'], color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                nplot=nplot+1

            if option[3]=="y":
                nbins, bins, patches = axs[nplot].hist(ExoREMarray[n][in_cond]['radius']/R_jup.value,histtype='step', fill=False, range=[0.6,2],  bins=15, weights=1./ExoREMarray[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))
                popt=nofitgaus(nbins, bins)
                titleRadius=titleRadius+str('%.1f' %float(popt[0]))+" +- "+str('%.1f' %float(abs(popt[1])))+"Rjup\n"
                nplot=nplot+1

            if option[4]=="y":
                axs[nplot].hist(ExoREMarray[n][in_cond]['metalicity'],histtype='step', fill=False, log=True, bins=15, weights=1./ExoREMarray[n][in_cond]['chi2final'],color=(colors_planets[m][0],colors_planets[m][1],colors_planets[m][2]))

        axs[0].set_title(titleTeff)
        axs[1].set_title(titleLogg)
        axs[2].set_title(titleMass)
        axs[3].set_title(titleRadius)

    show()

    fig.savefig(str(dir_output[0])+"HistogramsWthCdt_"+str(sig)+str(FitType[p])+"_"+str(typeCloud)+"_"+str(planetName[0])+".pdf", format="pdf", dpi=600)
    rad_f=ExoREMarray[n][in_cond]['radius']/R_jup.value
    mas_f=ExoREMarray[n][in_cond]['mass']/M_jup.value
    Tef_f=ExoREMarray[n][in_cond]['Teff']
    z_f=ExoREMarray[n][in_cond]['metalicity']
    lgg_f=np.log10(ExoREMarray[n][in_cond]['gravity'])
    chi_f=ExoREMarray[n][in_cond]['chi2final']
    avTef=sum(Tef_f*chi_f)/sum(chi_f)
    print(int(avTef),"+-",int(avTef-min(Tef_f)+max(Tef_f)-avTef)/2)
    avlgg=sum(lgg_f*chi_f)/sum(chi_f)
    print(int(avlgg),"+-",int(avlgg-min(lgg_f)+max(lgg_f)-avlgg)/2)
    avz=sum(z_f*chi_f)/sum(chi_f)
    print(int(avz),"+-",int(avz-min(z_f)+max(z_f)-avz)/2)


    return()

#############

class InputFile():
    '''
    description :
                 class to read the JSON input file
    author : 
            JL Baudino
    date :
          22/12/2019
    '''
    def __init__(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
        self.R=data['R']
        self.Distance=data['Distance']
        self.FitType=data['FitType']
        self.dir_input=data['dir_input']
        self.dir_obs=data['dir_obs']
        self.dir_output=data['dir_output']
        self.planetName=data['planetName']
        self.typeCloud=data['typeCloud']
        self.database=data['database']
        self.table=data['table']
    
    def get_param(self):
        return self.R, self.Distance, self.FitType, self.dir_input, self.dir_obs, self.dir_output, self.planetName, self.typeCloud

class ErrLog():
    '''
    description :
                 class to manage the log file
    author : 
            JL Baudino
    date :
          22/12/2019
    '''
    def __init__(self, file):
        self.log=[]
        self.path=file
    
    def last(self):
        return self.log[-1]

    def add(self, info):
        self.log.append(info)
    
    def save(self):
        np.savetxt(self.path,self.log,fmt="%s")

