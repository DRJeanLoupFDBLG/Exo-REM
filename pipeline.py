from numpy import loadtxt, array, zeros, e, arange,exp, flipud, savetxt,trapz, concatenate, argsort, mean
from math import sqrt,pi,log
from pylab import plot, show, semilogy, errorbar, figure, xlim, ylim, rc, ylabel, xlabel
from astropy.table import Table
from Tools import *
import astropy.io.votable.tree as vo
import os.path
import csv
import sys
import json


def UseExoREM(
        inputfile="input_data.json",
        test_log="test_log.txt"
        ):
    """
    Author : JLB
    Date: 28/06/2019
    Description:
        - read data, filters, grid of models
        - fit the data 
        - save full result
        - save 5 sigma result to add OE later
    input: 
        JSON file containing:
        - R [m] 
        - Distance [m], 41.29 pc (d e) 10 pc (b c)
        - FitType="SED" #Full, SED, Spec
        - dir_input="../../data/Grilles/Charnay221118/run_cloud_f1.0_nosulfide_timescale_S0.003_newH2O-22-11-18/test/"
        - dir_output="Analyse_Exo-REM/b/"
        - planetName='HR8799b'
        - typeCloud='0003new'#'0003new', 'nocloud
    output:
    """

    print("###########################################")
    print("##Begining of the pipeline")
    print("###########################################")

    err = ErrLog(test_log)
    err.add([{"Title":"Pipeline running"}])

    print("###########################################")
    print("##Reading param file")
    print("###########################################")

    # Read the input file
    param_file = InputFile(inputfile)
    R, Distance, FitType, dir_input, dir_obs, dir_output, planetName, typeCloud = param_file.get_param()

    print("###########################################")
    print("##Reading obs file and list of models")
    print("###########################################")

    # Read list of model in grid
    datafile=dir_obs+planetName+"_SPHERE_"+FitType+".xml"
    tabeGrille=loadtxt(dir_input+"ListXmlFiles.info",dtype=[("logg",float),("teff",float),("z",float),("loc",'S90')])


    #Exo-REM step
    tabWavGeneral=arange(16000,00,-20)

    print("###########################################")
    print("##Reading obs spectrum")
    print("###########################################")

    #Load observations
    SED=Table.read(datafile, table_id="phot")
    Spectrum=Table.read(datafile, table_id="spec")

    err.save()
    print("###########################################")
    print("##Reading filters")
    print("###########################################")

    #Load filter and resolution
    if FitType=="Full":
        loc=array(list(Spectrum.field("lambda"))+list(SED.field("lambda")))[:,0]
        ResSpec=array(Spectrum.field("lambda"))[:,0]/array(Spectrum.field("resolution"))[:,0]
        FiltreId=loadtxt(dir_obs+"Tab_name-file_filters.tsv",dtype={'names':("id","file"),'formats':('S40','S40')},delimiter='	')
    elif FitType=="SED":
        loc=array(list(SED.field("lambda")))[:,0]
        FiltreId=loadtxt(dir_obs+"Tab_name-file_filters.tsv",dtype={'names':("id","file"),'formats':('S40','S40')},delimiter='	')
    elif FitType=="Spec":
        loc=array(list(Spectrum.field("lambda")))[:,0]
        ResSpec=loc/array(Spectrum.field("resolution"))[:,0]
    
    FiltreTrans=[]

    err.save()
    print("###########################################")
    print("##Building transmission filter")
    print("###########################################")

    # Generation of transmission a the resolution of Exo-REM
    ## Transform Ex-REM cm-1 -> mum
    tabWavGeneral=10000./tabWavGeneral
    ## genration it-self
    for i in range(len(loc)):
        if (FitType=="Full") or (FitType=="Spec"):
            if i<len(list(Spectrum.field("lambda"))):
                FilNN, err = gaussian1D(1./sqrt(2.*pi), tabWavGeneral, \
                    float(loc[i]), (float(ResSpec[i])/(2*sqrt(2*log(2)))), logfile=err)
            else:
                Filtre = (loadtxt(str(FiltreId[FiltreId["id"]==array(SED.field("Filter"))[i-len(list(Spectrum.field("lambda")))]]["file"][0])[2:-1]))
                FilNN, err = ConstrFilter(Filtre[:,0],Filtre[:,1],tabWavGeneral, logfile=err)
        elif FitType=="SED":
            Filtre=(loadtxt(str(FiltreId[FiltreId["id"]==array(SED.field("Filter"))[i]]["file"][0])[2:-1]))
            FilNN, err = ConstrFilter(Filtre[:,0],Filtre[:,1],tabWavGeneral, logfile=err)
        if len(FilNN[FilNN != float('nan')]) > 0 :                                                                                                   
            FilNom = FilNN/sum(FilNN*tabWavGeneral)
        else:
            FilNom = zeros(len(FilNom))
        FiltreTrans.append(FilNom)
    
    FiltreTrans=array(FiltreTrans)

    err.save()
    print("###########################################")
    print("##Adding Spectra in one array")
    print("###########################################")

    #Application of the filters on models
    if FitType=="Full":
        ErrSED=[1]*len(SED.field('ErrorF_lambda'))
        for i in range(len(SED.field('ErrorF_lambda'))):
            ErrSED[i]=array(SED.field('ErrorF_lambda'))[i,0]
        Spectre=(array(list(Spectrum.field('F_lambda'))+list(SED.field('F_lambda')))[:,0])
        ErrSpectre=(array(list(array(Spectrum.field('ErrorF_lambda'))[:,0])+ErrSED))
    elif FitType=="SED":
        ErrSED=[1]*len(SED.field('ErrorF_lambda'))
        for i in range(len(SED.field('ErrorF_lambda'))):
            ErrSED[i]=array(SED.field('ErrorF_lambda'))[i,0]
        Spectre=(array(list(SED.field('F_lambda')))[:,0])
        ErrSpectre=(array(ErrSED))
    elif FitType=="Spec":
        Spectre=(array(list(Spectrum.field('F_lambda')))[:,0])
        ErrSpectre=(array(Spectrum.field('ErrorF_lambda'))[:,0])

    err.save()
    print("###########################################")
    print("##Applying filters on models")
    print("###########################################")

    #Genrate grid:
    tabSauver=[]
    for model in tabeGrille:
        file= (str(model["loc"])[2:-1])
        if os.path.isfile(dir_input+file)==True:
            ys=Table.read(dir_input+file, table_id="Spectrum", )["F_lambda"]                            
    #---------calcul grille de flux
            SpectreFinal=[]
            for n in range(len(loc)):
                SpectreFinal.append(FiltreTrans[n].dot(ys)/sum(FiltreTrans[n]))
            SpectreFinal=array(SpectreFinal)
            tabSauver.append([model, SpectreFinal])

    err.save()
    print("###########################################")
    print("##Comparing models and spectrum")
    print("###########################################")

    #########
    #Analysis
    #########
    tab_recap=[]

    FluxRef=array(tabSauver)
    deltaDistance=0.05 #pc
    Name=(planetName+"_"+FitType+"")
    RadRange=[0.6,2.]
    Norm=""

    #Extract observation
    if FitType=="Full":
        ErrSED=[1]*len(SED.field('ErrorF_lambda'))
        for i in range(len(SED.field('ErrorF_lambda'))):
            ErrSED[i]=array(SED.field('ErrorF_lambda'))[i,0]

        Spectre=(array(list(Spectrum.field('F_lambda'))+list(SED.field('F_lambda')))[:,0])
        ErrSpectre=(array(list(array(Spectrum.field('ErrorF_lambda'))[:,0])+ErrSED))
    elif FitType=="SED":
        ErrSED=[1]*len(SED.field('ErrorF_lambda'))
        for i in range(len(SED.field('ErrorF_lambda'))):
            ErrSED[i]=array(SED.field('ErrorF_lambda'))[i,0]
        Spectre=(array(list(SED.field('F_lambda')))[:,0])
        ErrSpectre=(array(ErrSED))
    elif FitType=="Spec":
        Spectre=(array(list(Spectrum.field('F_lambda')))[:,0])
        ErrSpectre=(array(Spectrum.field('ErrorF_lambda'))[:,0])

    if Norm!="":
        ErrSpectre=ErrSpectre/Spectre[Norm]
        Spectre=Spectre/Spectre[Norm]

    #
    specDeg=zeros(len(Spectre))
    Tab1sig=[]
    Tab3sig=[]
    TabLoc1sig=[]
    Tab5sig=[]
    Tab10sig=[]
    for ligne in FluxRef:
        xi2test=0
        flux=ligne[-1][:,0]
        info=ligne[0]
        if Norm != "":
            Fnorm=float(flux[int(Spectre[int(Norm)])])
        for Filtre in range(len(Spectre)):
            if Norm != "":
                specDeg[Filtre]=float(flux[int(Filtre)])/Fnorm
            else:
                specDeg[Filtre]= flux[Filtre]*(3.086e17/Distance)**2

        diff=((specDeg-Spectre)/ErrSpectre)**2.
        ErrAv=xi2(specDeg,Spectre,ErrSpectre)
        Rayon2=sum((1/diff)*Spectre/specDeg)/sum(1/diff)
        Rayon=sqrt(Rayon2)
        corrspecDeg=specDeg*Rayon2
        ErrAv, err = xi2(specDeg,Spectre,ErrSpectre, logfile=err)
        Err, err = xi2(corrspecDeg,Spectre,ErrSpectre, logfile=err)
        
        Masse, err = MassCalc (float(info[0]), Rayon, logfile=err)
        FinalToSave=[Rayon, info[0], info[1], info[2], info[3], Masse, ErrAv, Err, corrspecDeg]
        tab_recap.append(FinalToSave)
        
        if Err/len(Spectre)<=1:
            Tab1sig.append(FinalToSave)
            TabLoc1sig.append(info[3])
        if Err/len(Spectre)<=3:
            Tab3sig.append(FinalToSave)
        if Err/len(Spectre)<=5:
            Tab5sig.append(FinalToSave)
        if Err/len(Spectre)<=10:
            Tab10sig.append(FinalToSave)

    err.add([{"Info":"End Computation"}])
    err.save()
    print("###########################################")
    print("##Saving results")
    print("###########################################")

    #print results
    TabLoc1sig=array(TabLoc1sig)
    savetxt(dir_output+"List1Sigma_"+FitType+"_"+typeCloud+"_"+Name+".info",TabLoc1sig,fmt="%s")

    tab_recap=array(tab_recap,dtype=object)
    Tab1sig=array(Tab1sig,dtype=object)
    Tab3sig=array(Tab3sig,dtype=object)
    Tab5sig=array(Tab5sig,dtype=object)
    Tab10sig=array(Tab10sig,dtype=object)

    with open(dir_output+"Tableaufinal_"+FitType+"_"+typeCloud+"_"+Name+".tsv", "w") as TabFinal:
        Tabrecp = csv.writer(TabFinal, delimiter='	')
        for ligne in tab_recap:
            Tabrecp.writerow(ligne)

    with open(dir_output+"Param_"+FitType+"_"+typeCloud+"_"+Name+".tsv", "w") as Param:
        ParamFinal = csv.writer(Param, delimiter='	')

    ##############
    # Save results
    ##############
    
    # Create a new VOTable file...
    votable = vo.VOTableFile()

    # ...with one resource...
    resource = vo.Resource(name="result_fit")
    votable.resources.append(resource)

    # ... with three tables
    table_param = vo.Table(votable, ID="parameters")
    resource.tables.append(table_param)
    table_fit = vo.Table(votable, ID="fit")
    resource.tables.append(table_fit)
    table_spec = vo.Table(votable, ID="spectre")
    resource.tables.append(table_spec)

    table_param.fields.extend([
        vo.Field(votable, name="gravity", datatype="double", unit="cm.s-2", arraysize="1"),
        vo.Field(votable, name="Teff", datatype="double", unit="K", arraysize="1"),
        vo.Field(votable, name="metalicity", datatype="double", arraysize="1"),
        vo.Field(votable, name="file", datatype="char", arraysize="*")
    ])

    table_fit.fields.extend([
        vo.Field(votable, name="radius", datatype="double", unit="m", arraysize="1"),
        vo.Field(votable, name="mass", datatype="double", unit="kg", arraysize="1"),
        vo.Field(votable, name="chi2avfit", datatype="double", arraysize="1"),
        vo.Field(votable, name="chi2final", datatype="double", arraysize="1")
    ])

    table_spec.fields.extend([
        vo.Field(votable, name="lambda", datatype="double", unit="um", arraysize=str(len(loc))),
        vo.Field(votable, name="F_lambda", datatype="double", unit="W.m-2.um-1", arraysize=str(len(loc))),
    ])
    
    table_param.create_arrays(len(tab_recap))
    table_fit.create_arrays(len(tab_recap))
    table_spec.create_arrays(len(tab_recap))


    # Now table.array can be filled with data
    for i in range(len(tab_recap)):
        table_param.array[i] = (10.**tab_recap[i,1],tab_recap[i,2],tab_recap[i,3],str(tab_recap[i,4])[2:-1])
        table_fit.array[i] = (tab_recap[i,0]*R,tab_recap[i,5]*1.8987e+27,tab_recap[i,6],tab_recap[i,7])
        table_spec.array[i] = (loc,tab_recap[i,8])
    votable.to_xml(dir_output+"FitFinal"+FitType+"_"+typeCloud+"_"+planetName+".xml")


#UseExoREM()