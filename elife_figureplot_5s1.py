# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:45:07 2015

@author: rennocosta
"""

# %%


import sys, argparse
import numpy as np
from numpy import *
#from brian2 import *
from matplotlib import *
import gzip
import pickle
import support_filename as rfn
import copy
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

envir = "BIOME"
filenames = rfn.remappingFileNames(envir)



# %%

seed_input = arange(1)
seed_www = arange(8,12)

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

hpc_pcompl_th = 80



# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

morph_per = 100

morphnum = 21

ct = 0

simulation_num = 7


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

size1 = (len(reference4),len(reference3),len(reference2)*len(reference1),morphnum) 
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1)) 

#cucuH1 = np.zeros(size1)
#cucuH2 = np.zeros(size1)
#cucuM1 = np.zeros(size1)
#cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)

avaaG1 = np.zeros(size2)
avaaG2 = np.zeros(size2)

ddff1 = np.zeros(size2)
ddff2 = np.zeros(size2)
ddnn1 = np.zeros(size2)
ddnn2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       dist_pf1,dist_pf2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    #cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    #cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    #cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    #cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ii4,ii3,ppp] = corrVectHPC1
                    avaaH2[ii4,ii3,ppp] = corrVectHPC2
                    avaaM1[ii4,ii3,ppp] = corrVectMEC1
                    avaaM2[ii4,ii3,ppp] = corrVectMEC2
                    #avaaM2[ppp,ii4] = corrVectMEC2
                    
                    avaaG1[ii4,ii3,ppp] = corrVectMECvsGRID1
                    avaaG2[ii4,ii3,ppp] = corrVectMECvsGRID2
                    
                    ddnn1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:])/np.sum(dist_pf1[0][0:11:])
                    ddnn2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:])/np.sum(dist_pf2[0][0:11:])
                    if ((np.sum(dist_pf1[0][1:11:])) > 0.0):                   
                        ddff1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf1[0][1:11:]);
                    if ((np.sum(dist_pf2[0][1:11:])) > 0.0):
                        ddff2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf2[0][1:11:]);
                    
                                        
                    
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                


# %%

plt.figure()
rrr = reference3;
rrr2 = reference4;

subplot(1,2,1)
aa = np.mean(avaaH1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(avaaM1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1C2.eps', format='eps', dpi=1000)

# %%

rrr = reference3;
rrr2 = reference4;

plt.figure()

# average number of place fields
subplot(1,3,1) 
aa = np.mean((ddff1[:,:,:] + ddff2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 1, vmax=5)

# percentage of cells with place fields (not shown in paper)
subplot(1,3,2) 
aa = np.mean((ddnn1[:,:,:] + ddnn2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=0.06)

# gridness
subplot(1,3,3)
aa = np.mean((avaaG1[:,:,:] + avaaG2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1AB2.eps', format='eps', dpi=1000)














# %%

seed_input = arange(1)
seed_www = arange(8,12)

seed_path = 0

lrate_hpc_mec = 0
lrate_ec_hpc = 0

hpc_pcompl_th = 80



# %% with pre run

theta_cycles = 7
arena_runs = 1
pre_runs = 0  # 50


mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

morph_per = 100

morphnum = 21

ct = 0

simulation_num = 7


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

size1 = (len(reference4),len(reference3),len(reference2)*len(reference1),morphnum) 
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1)) 

#cucuH1 = np.zeros(size1)
#cucuH2 = np.zeros(size1)
#cucuM1 = np.zeros(size1)
#cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)

avaaG1 = np.zeros(size2)
avaaG2 = np.zeros(size2)

ddff1 = np.zeros(size2)
ddff2 = np.zeros(size2)
ddnn1 = np.zeros(size2)
ddnn2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       dist_pf1,dist_pf2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    #cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    #cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    #cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    #cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ii4,ii3,ppp] = corrVectHPC1
                    avaaH2[ii4,ii3,ppp] = corrVectHPC2
                    avaaM1[ii4,ii3,ppp] = corrVectMEC1
                    avaaM2[ii4,ii3,ppp] = corrVectMEC2
                    #avaaM2[ppp,ii4] = corrVectMEC2
                    
                    avaaG1[ii4,ii3,ppp] = corrVectMECvsGRID1
                    avaaG2[ii4,ii3,ppp] = corrVectMECvsGRID2
                    
                    ddnn1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:])/np.sum(dist_pf1[0][0:11:])
                    ddnn2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:])/np.sum(dist_pf2[0][0:11:])
                    if ((np.sum(dist_pf1[0][1:11:])) > 0.0):                   
                        ddff1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf1[0][1:11:]);
                    if ((np.sum(dist_pf2[0][1:11:])) > 0.0):
                        ddff2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf2[0][1:11:]);
                    
                                        
                    
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                


# %%

plt.figure()
rrr = reference3;
rrr2 = reference4;

subplot(1,2,1)
aa = np.mean(avaaH1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(avaaM1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1C1.eps', format='eps', dpi=1000)

# %%

rrr = reference3;
rrr2 = reference4;

plt.figure()

# average number of place fields
subplot(1,3,1) 
aa = np.mean((ddff1[:,:,:] + ddff2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 1, vmax=5)

# percentage of cells with place fields (not shown in paper)
subplot(1,3,2) 
aa = np.mean((ddnn1[:,:,:] + ddnn2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

# gridness
subplot(1,3,3)
aa = np.mean((avaaG1[:,:,:] + avaaG2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1AB1.eps', format='eps', dpi=1000)








# %%

seed_input = arange(1)
seed_www = arange(8,12)

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 0

hpc_pcompl_th = 80



# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

morph_per = 100

morphnum = 21

ct = 0

simulation_num = 7


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

size1 = (len(reference4),len(reference3),len(reference2)*len(reference1),morphnum) 
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1)) 

#cucuH1 = np.zeros(size1)
#cucuH2 = np.zeros(size1)
#cucuM1 = np.zeros(size1)
#cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)

avaaG1 = np.zeros(size2)
avaaG2 = np.zeros(size2)

ddff1 = np.zeros(size2)
ddff2 = np.zeros(size2)
ddnn1 = np.zeros(size2)
ddnn2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       dist_pf1,dist_pf2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    #cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    #cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    #cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    #cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ii4,ii3,ppp] = corrVectHPC1
                    avaaH2[ii4,ii3,ppp] = corrVectHPC2
                    avaaM1[ii4,ii3,ppp] = corrVectMEC1
                    avaaM2[ii4,ii3,ppp] = corrVectMEC2
                    #avaaM2[ppp,ii4] = corrVectMEC2
                    
                    avaaG1[ii4,ii3,ppp] = corrVectMECvsGRID1
                    avaaG2[ii4,ii3,ppp] = corrVectMECvsGRID2
                    
                    ddnn1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:])/np.sum(dist_pf1[0][0:11:])
                    ddnn2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:])/np.sum(dist_pf2[0][0:11:])
                    if ((np.sum(dist_pf1[0][1:11:])) > 0.0):                   
                        ddff1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf1[0][1:11:]);
                    if ((np.sum(dist_pf2[0][1:11:])) > 0.0):
                        ddff2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf2[0][1:11:]);
                    
                                        
                    
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                


# %%

plt.figure()
rrr = reference3;
rrr2 = reference4;

subplot(1,2,1)
aa = np.mean(avaaH1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(avaaM1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1C3.eps', format='eps', dpi=1000)

# %%

rrr = reference3;
rrr2 = reference4;

plt.figure()

# average number of place fields
subplot(1,3,1) 
aa = np.mean((ddff1[:,:,:] + ddff2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 1, vmax=5)

# percentage of cells with place fields (not shown in paper)
subplot(1,3,2) 
aa = np.mean((ddnn1[:,:,:] + ddnn2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=0.06)

# gridness
subplot(1,3,3)
aa = np.mean((avaaG1[:,:,:] + avaaG2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1AB3.eps', format='eps', dpi=1000)





# %%

seed_input = arange(1)
seed_www = arange(8,12)

seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

hpc_pcompl_th = 80



# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

morph_per = 100

morphnum = 21

ct = 0

simulation_num = 7


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

size1 = (len(reference4),len(reference3),len(reference2)*len(reference1),morphnum) 
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1)) 

#cucuH1 = np.zeros(size1)
#cucuH2 = np.zeros(size1)
#cucuM1 = np.zeros(size1)
#cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)

avaaG1 = np.zeros(size2)
avaaG2 = np.zeros(size2)

ddff1 = np.zeros(size2)
ddff2 = np.zeros(size2)
ddnn1 = np.zeros(size2)
ddnn2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       dist_pf1,dist_pf2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    #cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    #cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    #cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    #cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ii4,ii3,ppp] = corrVectHPC1
                    avaaH2[ii4,ii3,ppp] = corrVectHPC2
                    avaaM1[ii4,ii3,ppp] = corrVectMEC1
                    avaaM2[ii4,ii3,ppp] = corrVectMEC2
                    #avaaM2[ppp,ii4] = corrVectMEC2
                    
                    avaaG1[ii4,ii3,ppp] = corrVectMECvsGRID1
                    avaaG2[ii4,ii3,ppp] = corrVectMECvsGRID2
                    
                    ddnn1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:])/np.sum(dist_pf1[0][0:11:])
                    ddnn2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:])/np.sum(dist_pf2[0][0:11:])
                    if ((np.sum(dist_pf1[0][1:11:])) > 0.0):                   
                        ddff1[ii4,ii3,ppp] = np.sum(dist_pf1[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf1[0][1:11:]);
                    if ((np.sum(dist_pf2[0][1:11:])) > 0.0):
                        ddff2[ii4,ii3,ppp] = np.sum(dist_pf2[0][1:11:] * np.array([1,2,3,4,5,6,7,8,9,10])) / np.sum(dist_pf2[0][1:11:]);
                    
                                        
                    
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                


# %%

plt.figure()
rrr = reference3;
rrr2 = reference4;

subplot(1,2,1)
aa = np.mean(avaaH1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(avaaM1[:,:,:],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1C4nu.eps', format='eps', dpi=1000)

# %%

rrr = reference3;
rrr2 = reference4;

plt.figure()

# average number of place fields
subplot(1,3,1) 
aa = np.mean((ddff1[:,:,:] + ddff2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 1, vmax=5)

# percentage of cells with place fields (not shown in paper)
subplot(1,3,2) 
aa = np.mean((ddnn1[:,:,:] + ddnn2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=0.06)

# gridness
subplot(1,3,3)
aa = np.mean((avaaG1[:,:,:] + avaaG2[:,:,:])/2,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

plt.savefig('Figures/rennocostatort_elife_fig5s1AB4nu.eps', format='eps', dpi=1000)




