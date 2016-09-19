# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:45:01 2016

@author: rennocosta
"""

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

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [0,10,15,20,25,30,35,40,50]
hpc_pcompl_th = [80,80,80,80,80,80,80,80,80]

# %%

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50



morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect = [0,1,2,3,4,5,6,7,8,9,10,11,12]

input_noise_vect2 = np.array(input_noise_vect)/16;

#morph_per = [98];

morphnum = 21

ct = 0

simulation_num = 6


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass





# %%
plt.figure()
reffa = hpc_ratio;
#reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig6B.eps', format='eps', dpi=1000)





# %%

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [0,10,15,20,25,30,35,40,50]
hpc_pcompl_th = [80,80,80,80,80,80,80,80,80]


# %%


theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect2 = input_noise_vect;


morphnum = 21

ct = 0

simulation_num = 4


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
                
        # %% FIGURA 4A - PV CORR as a Function of morph level

#
#

plt.figure()
reffa = hpc_ratio;
#reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    
plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/rennocostatort_elife_fig6D.eps', format='eps', dpi=1000)






























# %%

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [40,40,40,40,40,40]
hpc_pcompl_th = [40,50,60,70,90,99]

# %%

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50



morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect = [0,1,2,3,4,5,6,7,8,9,10,11,12]

input_noise_vect2 = np.array(input_noise_vect)/16;

#morph_per = [98];

morphnum = 21

ct = 0

simulation_num = 6


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass





# %%
plt.figure()
#reffa = hpc_ratio;
reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig6s1A.eps', format='eps', dpi=1000)





# %%

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [40,40,40,40,40,40]
hpc_pcompl_th = [40,50,60,70,90,99]


# %%


theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect2 = input_noise_vect;


morphnum = 21

ct = 0

simulation_num = 4


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
                
        # %% FIGURA 4A - PV CORR as a Function of morph level

#
#

plt.figure()
#reffa = hpc_ratio;
reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    
plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/rennocostatort_elife_fig6s1B.eps', format='eps', dpi=1000)
      
















# %%

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [0,0,0,0,0,0]
hpc_pcompl_th = [40,50,60,70,90,99]

# %%

theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50



morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect = [0,1,2,3,4,5,6,7,8,9,10,11,12]

input_noise_vect2 = np.array(input_noise_vect)/16;

#morph_per = [98];

morphnum = 21

ct = 0

simulation_num = 6


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass





# %%
plt.figure()
#reffa = hpc_ratio;
reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    

plt.ylim((-0.1,1.1)) 

plt.savefig('Figures/rennocostatort_elife_fig6s1C.eps', format='eps', dpi=1000)





# %%

seed_input = arange(8)#np.array([0,1,2,3,4,5,6,7])#np.array([0,1,2,4,5,6])#arange(3)
#seed_www = arange(8)

seed_www = np.array([8])
#seed_input = np.array([0,1,2,3,4,5,6,7])

seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

mec_ratio = 70
#hpc_ratio = [0,10,15,20,25,35]
hpc_ratio = [0,0,0,0,0,0]
hpc_pcompl_th = [40,50,60,70,90,99]


# %%


theta_cycles = 7
arena_runs = 5
pre_runs = 6  # 50


morph_per = np.array([100], dtype=np.int32)

input_noise_vect = [0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.7,0.8,0.9,0.99]
input_noise_vect2 = input_noise_vect;


morphnum = 21

ct = 0

simulation_num = 4


reference1 = seed_input
reference2 = seed_www
reference3 = hpc_ratio
reference4 = morph_per

size1 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect),morphnum) 
size2 = (len(reference2)*len(reference1),len(reference3),len(morph_per),len(input_noise_vect)) 

cucuH1 = np.zeros(size1)
cucuH2 = np.zeros(size1)
cucuM1 = np.zeros(size1)
cucuM2 = np.zeros(size1)

avaaH1 = np.zeros(size2)
avaaH2 = np.zeros(size2)
avaaM1 = np.zeros(size2)
avaaM2 = np.zeros(size2)



for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio,hpc_ratio[ii3],hpc_pcompl_th[ii3],morph_per[ii4]]
                
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
                       
                    
                    pvCorrelationCurveHPC1[:,0] = corrVectHPC1;
                    pvCorrelationCurveHPC2[:,0] = corrVectHPC2;
                    pvCorrelationCurveMEC1[:,0] = corrVectMEC1;
                    pvCorrelationCurveMEC2[:,0] = corrVectMEC2;
                    
                    
                    cucuH1[ppp,ii3,ii4,:] = pvCorrelationCurveHPC1
                    cucuH2[ppp,ii3,ii4,:] = pvCorrelationCurveHPC2
                    cucuM1[ppp,ii3,ii4,:] = pvCorrelationCurveMEC1
                    cucuM2[ppp,ii3,ii4,:] = pvCorrelationCurveMEC2
                    #cucuL1[ppp,ii4,:] = pvCorrelationCurveLEC1
                    #cucuL2[ppp,ii4,:] = pvCorrelationCurveLEC2
                    
                    avaaH1[ppp,ii3,ii4,:] = corrVectHPC1
                    avaaH2[ppp,ii3,ii4,:] = corrVectHPC2
                    avaaM1[ppp,ii3,ii4,:] = corrVectMEC1
                    avaaM2[ppp,ii3,ii4,:] = corrVectMEC2

                except:
                    print("%d %d %d %d %d" % (reference4[ii4],reference3[ii3],hpc_pcompl_th[ii3],reference1[ii1],reference2[ii2]))
                    pass
                
                
        # %% FIGURA 4A - PV CORR as a Function of morph level

#
#

plt.figure()
#reffa = hpc_ratio;
reffa = hpc_pcompl_th;

morpha = [100]
popa = np.argwhere(morph_per==morpha)[0][0]
for kk in arange(shape(hpc_ratio)[0]):
    plot(input_noise_vect2,np.median(avaaH1[:,kk,popa,:],axis=0),color=[1.0*((reffa[kk]/np.max(reffa))),0.0,1.0*(1-(reffa[kk]/np.max(reffa)))]);
    
plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/rennocostatort_elife_fig6s1D.eps', format='eps', dpi=1000)
      