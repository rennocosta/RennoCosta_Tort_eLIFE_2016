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

#
# FIGURE 5
#

seed_input = arange(8)
seed_www = arange(8)

#seed_input = np.array([2,3,4,5,6,7])
#seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 10
hpc_pcompl_th = 80

simulation_num = 2


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5A 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5A.eps', format='eps', dpi=1000)


# %% FIGURA 5B

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5B.eps', format='eps', dpi=1000)


# %% FIGURA 5C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5C.eps', format='eps', dpi=1000)











# %%

#
# FIGURE 5s2
#

seed_input = arange(8)
seed_www = arange(8)

seed_input = np.array([0,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 0
hpc_pcompl_th = 80

simulation_num = 2


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s2B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s2B.eps', format='eps', dpi=1000)


# %% FIGURA 5s2C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s2C.eps', format='eps', dpi=1000)


# %% FIGURA 5s2D

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s2D.eps', format='eps', dpi=1000)



















# %%

#
# FIGURE 5s3
#

seed_input = arange(8)
seed_www = arange(8)


seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 7


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morph_per = np.array([  0,  5, 10, 15, 20, 25, 30 , 35, 40, 42, 44 , 46, 48, 
                      50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 
                      72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 
                      92, 94, 96, 98, 100], dtype=np.int32)


morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s3B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s3B.eps', format='eps', dpi=1000)


# %% FIGURA 5s3C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s3C.eps', format='eps', dpi=1000)


# %% FIGURA 5s3D

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s3D.eps', format='eps', dpi=1000)













# %%

#
# FIGURE 5s4
#

seed_input = arange(8)
seed_www = arange(8)

seed_input = np.array([1,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 0

mec_ratio = [70]
hpc_ratio = 10
hpc_pcompl_th = 80

simulation_num = 2


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s4B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s4B.eps', format='eps', dpi=1000)


# %% FIGURA 5s4C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s4C.eps', format='eps', dpi=1000)


# %% FIGURA 5s4D

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s4D.eps', format='eps', dpi=1000)












# %%

#
# FIGURE 5s5
#

seed_input = arange(8)
seed_www = arange(8)

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 0

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 7


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morph_per = np.array([  0,  5, 10, 15, 20, 25, 30 , 35, 40, 42, 44 , 46, 48, 
                      50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 
                      72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 
                      92, 94, 96, 98, 100], dtype=np.int32)

morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s5B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s5B.eps', format='eps', dpi=1000)


# %% FIGURA 5s5C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s5C.eps', format='eps', dpi=1000)


# %% FIGURA 5s5D

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s5D.eps', format='eps', dpi=1000)
















# %%

#
# FIGURE 5s6
#

seed_input = arange(8)
seed_www = arange(8)

seed_input = np.array([1,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 2


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)

morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s6B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s6B.eps', format='eps', dpi=1000)


# %% FIGURA 5s6C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s6C.eps', format='eps', dpi=1000)













# %%

#
# FIGURE 5s7
#

seed_input = arange(8)
seed_www = arange(8)

seed_input = np.array([0,1,2,3,4,5,6,7])
seed_www = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 99

simulation_num = 7


# %% with pre run

theta_cycles = 7
arena_runs = 5
pre_runs = 6  

morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)


morphnum = 21

ct = 0

reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)


for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 5s7B 


paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)

plt.figure()

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s7B.eps', format='eps', dpi=1000)


# %% FIGURA 5s7C

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s7C.eps', format='eps', dpi=1000)


# %% FIGURA 5s7D

torange = [0,10,20,30,40,50,60,70,80,90,100]

plt.figure()

for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig5s7D.eps', format='eps', dpi=1000)



















# %%

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([0,1,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 0
hpc_pcompl_th = 80

name_string = 'B';


simulation_num = 2

# %%

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 10
hpc_pcompl_th = 80

name_string = 'B2';


simulation_num = 2

# %%

name_string = 'C';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([0,1,2,3,4,5,6,7])
seed_www = np.array([0,1,2,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 0

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 7

# %%

name_string = 'C2';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([1,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 0

mec_ratio = [70]
hpc_ratio = 10
hpc_pcompl_th = 80

simulation_num = 2


# %%

name_string = 'D';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([0,1,2,3,4,5,6,7])
seed_www = np.array([0,1,2,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 7

# %%

name_string = 'D2';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([1,2,3,4,5,6,7])
seed_www = np.array([0,1,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 10
hpc_pcompl_th = 80

simulation_num = 2


# %%

name_string = 'E';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([0,1,2,3,4,5,6,7])
seed_www = np.array([0,1,2,3,4,5,6])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [0]
hpc_ratio = 35
hpc_pcompl_th = 80

simulation_num = 7


# %%

name_string = 'F';

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)

seed_input = arange(8)
seed_www = arange(8)

#seed_www = np.array([0,2,3,4,6])
#seed_input = np.array([])
#seed_www = np.array([0,1])

seed_input = np.array([0,1,2,3,4])
seed_www = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = [70]
hpc_ratio = 35
hpc_pcompl_th = 99

simulation_num = 7




# %% with pre run

#seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(4)
#seed_path = 0

#lrate_hpc_mec = 100
#lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


#mec_ratio = [70]
#hpc_ratio = 10
#hpc_pcompl_th = 80
#morph_per = np.concatenate([int0(np.linspace(0,100,21)),int0([42,44,46,48,52,54,56,58])])
#morph_per = np.concatenate([int0(np.linspace(0,40,11)),int0(np.linspace(42,100,30)),int0(np.linspace(71,89,10)),int0([30,34,35,37,38,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69])])
#morph_per = int0(np.linspace(0,100,21))

#morph_per = np.array([  0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  42,  44,
#        46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
#        71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#        84,  85,  86,  87,  88,  89,  90,  92,  94,  96,  98, 100], dtype=np.int32)

#morph_per = np.array([  0,   4,   8,  10, 12,  16,  20,  24,  28,  30, 32,  36,  40,  42,  44,
#        46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
#        71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#        84,  85,  86,  87,  88,  89,  90,  92,  94,  96,  98, 100], dtype=np.int32)
        
        
#morph_per = np.array([  0,   4,   8,  10, 12,  16,  20,  24,  28,  30, 32,  36,  40,  42,  44,
#        46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
#        71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#        84,  85,  86,  87,  88,  89,  90,  92,  94,  96,  98, 100], dtype=np.int32)
#        
#        
#morph_per = np.array([  0,   4,     10, 12,  16,  20,  24,  28,  30, 32,  36,  40,  42,  44,
#        46,    50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,
#        72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#        84,  85,  86,  87,  88,  89,  90,  92,  94,  96,  98, 100], dtype=np.int32)


morph_per = np.array([  0,  10, 20, 30 ,40, 50, 60, 70, 80, 90, 100], dtype=np.int32)


#morph_per = [98];

morphnum = 21

ct = 0




reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = morph_per

size1 = (len(reference3)*len(reference2)*len(reference1),len(morph_per),morphnum) 
size2 = (len(reference3)*len(reference2)*len(reference1),len(morph_per)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    ppp = -1
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio,hpc_pcompl_th,morph_per[ii4]]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,0)+'z', 'rb') as ff:
                       corrVectMECGRID1,corrVectHPCGRID1,corrVectMEC1,corrVectHPC1,corrVectMECvsGRID1 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       corrVectMECGRID2,corrVectHPCGRID2,corrVectMEC2,corrVectHPC2,corrVectMECvsGRID2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                    #   corrVectMECGRIDx,corrVectHPCGRIDx,corrVectMECx,corrVectHPCx = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                    #   dist_pf1,dist_pf2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,4)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC,pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2 = pickle.load(ff)
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,5)+'z', 'rb') as ff:
                    #   pvCorrelationCurveHPCLesion,pvCorrelationCurveHPC1Lesion,pvCorrelationCurveHPC2Lesion,pvCorrelationCurveMECLesion,pvCorrelationCurveMEC1Lesion,pvCorrelationCurveMEC2Lesion = pickle.load(ff)
                       
                    cucuH1[ppp,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii4] = corrVectHPC1
                    avaaH2[ppp,ii4] = corrVectHPC2
                    avaaM1[ppp,ii4] = corrVectMEC1
                    avaaM2[ppp,ii4] = corrVectMEC2
                    
                    #print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                except:
                    print("%d %d ii %d ww %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
# %% FIGURA 4A - PV CORR as a Function of morph level

#
#
#

paaa = np.argsort(morph_per)                
            
aaaH1 = np.median(cucuH1[:,paaa,20],axis=0)
aaaH1u = np.percentile(cucuH1[:,paaa,20],10,axis=0)
aaaH1l = np.percentile(cucuH1[:,paaa,20],90,axis=0)

#aaaH2 = np.median(cucuH2[:,paaa,20],axis=0)
aaaM1 = np.median(cucuM1[:,paaa,20],axis=0)
aaaM1u = np.percentile(cucuM1[:,paaa,20],10,axis=0)
aaaM1l = np.percentile(cucuM1[:,paaa,20],90,axis=0)
#aaaM2 = np.median(cucuM2[:,paaa,20],axis=0)

#aaaH1 = cucuH1[3,paaa,20]
#aaaH2 = cucuH2[3,paaa,20]
#aaaM1 = cucuM1[3,paaa,20]
#aaaM2 = cucuM2[3,paaa,20]

plot(morph_per[paaa],aaaH1,'r')
plot(morph_per[paaa],aaaH1l,'r')
plot(morph_per[paaa],aaaH1u,'r')
plot(morph_per[paaa],aaaM1,'k')
plot(morph_per[paaa],aaaM1l,'k')
plot(morph_per[paaa],aaaM1u,'k')

#gaga = 27

#aaaH1 = cucuH1[gaga,paaa,20]
#aaaH2 = cucuH2[gaga,paaa,20]
#aaaM1 = cucuM1[gaga,paaa,20]
#aaaM2 = cucuM2[gaga,paaa,20]

#plot(morph_per[paaa],aaaH1.transpose(),'r')
#plot(morph_per[paaa],aaaM1.transpose(),'k')




plt.ylim((-0.1,1.1)) 

#plt.savefig('Figures/SVG/elife_fig_5A_' + name_string + '.eps', format='eps', dpi=1000)


# %% FIGURA 4B - MORPH HPC

gaga = 3

torange = [0,10,20,30,40,50,60,70,80,90,100]
#for #ii in arange( cucuH1.shape[1]):   
for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    #plot(np.linspace(0,1,21),cucuH1[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH2[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(0,1,21),np.mean(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.mean(cucuH2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
    #plot(np.linspace(1,0,21),cucuH1[0,ii,:],color=[1.0*((morph_per[ii]/100)),0.0,1.0*(1-(morph_per[ii]/100))]) np.median(cucuM1[:,paaa,20],axis=0)
#plot(cucuM1[0,:,:].transpose(),'r')
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/SVG/elife_fig_5B_mn_' + name_string + '.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/remapping_HPC_morphing_median.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median_nolearn.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/Fig4B_' + name_string + '.eps', format='eps', dpi=1000)


# %% FIGURA 4B - MORPH HPC

gaga = 3

torange = [0,10,20,30,40,50,60,70,80,90,100]
#for #ii in arange( cucuH1.shape[1]):   
for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    #plot(np.linspace(0,1,21),cucuH1[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH2[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(0,1,21),np.mean(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),np.mean(cucuH2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
    #plot(np.linspace(1,0,21),cucuH1[0,ii,:],color=[1.0*((morph_per[ii]/100)),0.0,1.0*(1-(morph_per[ii]/100))]) np.median(cucuM1[:,paaa,20],axis=0)
#plot(cucuM1[0,:,:].transpose(),'r')
plt.ylim((-0.1,1.1)) 

#plt.savefig('Figures/SVG/elife_fig_5B_md_' + name_string + '.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/remapping_HPC_morphing_median.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median_nolearn.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/Fig4Bm_' + name_string + '.eps', format='eps', dpi=1000)

# %% FIGURA 4B - MORPH HPC

gaga = 3

torange = [100]
#for #ii in arange( cucuH1.shape[1]):   
for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    #plot(np.linspace(0,1,21),cucuH1[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH2[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))]) 
    for kk in arange(shape(cucuH1)[0]):
        plot(np.linspace(0,1,21),cucuH1[kk,popa,:],color=[0.5,0.5,0.5])
    plot(np.linspace(0,1,21),np.median(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(0,1,21),np.mean(cucuH1[:,popa,:],axis=0),color=[0.0,1.0*((morph_per[popa]/100)),1.0*(1-(morph_per[popa]/100))])
       
    #plot(np.linspace(1,0,21),np.median(cucuH2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(0,1,21),np.mean(cucuH1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),np.mean(cucuH2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
        
    #plot(np.linspace(1,0,21),cucuH1[0,ii,:],color=[1.0*((morph_per[ii]/100)),0.0,1.0*(1-(morph_per[ii]/100))]) np.median(cucuM1[:,paaa,20],axis=0)
#plot(cucuM1[0,:,:].transpose(),'r')
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/SVG/elife_fig_5Bs_mn_' + name_string + '.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/remapping_HPC_morphing_median.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median_nolearn.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/Fig4Bxm_' + name_string + '.eps', format='eps', dpi=1000)


# %% FIGURA 4C - MORPH MEC

gaga = 3

torange = [0,10,20,30,40,50,60,70,80,90,100]
#for #ii in arange( cucuH1.shape[1]):   
for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    #plot(np.linspace(0,1,21),cucuH1[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH2[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(0,1,21),np.mean(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.mean(cucuM2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH1[0,ii,:],color=[1.0*((morph_per[ii]/100)),0.0,1.0*(1-(morph_per[ii]/100))]) np.median(cucuM1[:,paaa,20],axis=0)
#plot(cucuM1[0,:,:].transpose(),'r')
plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/SVG/elife_fig_5C_mn_' + name_string + '.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/remapping_HPC_morphing_median.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median_nolearn.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/Fig4C_' + name_string + '.eps', format='eps', dpi=1000)



# %% FIGURA 4C - MORPH MEC

gaga = 3

torange = [0,10,20,30,40,50,60,70,80,90,100]
#for #ii in arange( cucuH1.shape[1]):   
for ii in arange( len(torange)):
    popa = np.argwhere(morph_per==torange[ii])[0][0]
    #plot(np.linspace(0,1,21),cucuH1[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH2[gaga,popa,:],color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(0,1,21),np.median(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    plot(np.linspace(1,0,21),np.median(cucuM2[:,popa,:],axis=0),'--',color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(0,1,21),np.mean(cucuM1[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),np.mean(cucuM2[:,popa,:],axis=0),color=[1.0*((morph_per[popa]/100)),0.0,1.0*(1-(morph_per[popa]/100))])
    #plot(np.linspace(1,0,21),cucuH1[0,ii,:],color=[1.0*((morph_per[ii]/100)),0.0,1.0*(1-(morph_per[ii]/100))]) np.median(cucuM1[:,paaa,20],axis=0)
#plot(cucuM1[0,:,:].transpose(),'r')
plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/SVG/elife_fig_5C_md_' + name_string + '.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median.eps', format='eps', dpi=1000)
#plt.savefig('Figures/SVG/remapping_HPC_morphing_median_nolearn.eps', format='eps', dpi=1000)

#plt.savefig('Figures/SVG/Fig4Cm_' + name_string + '.eps', format='eps', dpi=1000)







