# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:27:31 2015

@author: rennocosta
"""

# %%

import sys, argparse
import numpy as np
from brian2 import *
from matplotlib import *
import gzip
import pickle
import support_filename as rfn
import copy


#envir = "mac"
#envir = "windows"
envir = "BIOME"
filenames = rfn.remappingFileNames(envir)


# %% CASE 1  - SLAM 1-2 / 3-4

seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [10]
hpc_pcompl_th = 80
morph_per = 100


name_string = '1111';


# %% CASE 2 - SLAM 5-6 / 7-8


seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [00]
hpc_pcompl_th = 80
morph_per = 100

name_string = '1110';


# %% CASE 2 - SLAM 36 / 34


seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 000
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [00]
hpc_pcompl_th = 80
morph_per = 100

name_string = '1010';


# %% CASE 3  - SLAM 35 / 33

seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 000
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [10]
hpc_pcompl_th = 80
morph_per = 100


name_string = '1011';






# %% CASE 3

seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [00]
hpc_pcompl_th = 99
morph_per = 100

name_string = '1000';

# %% CASE 4

seed_input = arange(20)
seed_www = arange(20)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 00

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [70]
hpc_ratio = [00]
hpc_pcompl_th = 99
morph_per = 100

name_string = '0110';


# %% CASE 4

seed_input = arange(5)
seed_www = arange(1)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [00]
hpc_ratio = [10]
hpc_pcompl_th = 80
morph_per = 100

name_string = '1111B';


# %% with pre run
#
# CARREGA OS ARQUIVOS DE CONVERGENCIA - COM TUDO - TIPO 1
#

morphnum = 41

ct = 0

simulation_num = 1


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

sublento = (ct+1)*pre_runs + morphnum      
lento = morphnum + sublento * true_runs   
    

r_start = concatenate([[0],morphnum+(arange(true_runs)*sublento)+(ct+1)*pre_runs])   
#r_start = concatenate([[0],(morphnum + (ct+1)*pre_runs) + (arange(true_runs)*morphnum + (ct+1))])
r_end =  r_start.copy() +   morphnum    


size1 = (len(reference4)*len(reference3)*len(reference2)*len(reference1),len(r_start),morphnum,2,theta_cycles)
size2 = (len(reference4)*len(reference3)*len(reference2)*len(reference1),len(r_start),morphnum,2)


ccMECconverge = -1*ones(size1)
ccHPCconverge = -1*ones(size1)
ccMECconvergeDist = -1*ones(size1)
ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

ccpvCorrelationCurveHPC1 = -1*ones(size2)
ccpvCorrelationCurveHPC2 = -1*ones(size2)
ccpvCorrelationCurveMEC1 = -1*ones(size2)
ccpvCorrelationCurveMEC2 = -1*ones(size2) 
ccpvCorrelationCurveLEC1 = -1*ones(size2)
ccpvCorrelationCurveLEC2 = -1*ones(size2)      

ppp = -1

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (ii4,ii3,ii2,ii1))
                    pass





# %% FIGURA 3A - PV CORR

#
#  MEC
#

aaa = np.median(ccpvCorrelationCurveHPC1[:,:,:,0],axis = 0)
aaa1 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],75,axis = 0)
aaa2 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],25,axis = 0)

aaa = np.median(ccpvCorrelationCurveMEC1[:,:,:,0],axis = 0)
aaa1 = np.percentile(ccpvCorrelationCurveMEC1[:,:,:,0],75,axis = 0)
aaa2 = np.percentile(ccpvCorrelationCurveMEC1[:,:,:,0],25,axis = 0)


plot(np.linspace(1.0,0.0,morphnum),aaa[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[0,:],color='k')

plot(np.linspace(1.0,0.0,morphnum),aaa[1,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[2,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[5,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[10,:],color='r')
#plot(np.linspace(1.0,0.0,morphnum),aaa[15,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[20,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[30,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[40,:],color='r')

plot(np.linspace(1.0,0.0,morphnum),aaa[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[50,:],color='k')

plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/SVG/Fig3A_MEC' + name_string + '.eps', format='eps', dpi=1000)


# %% FIGURA 3A - PV CORR

#
#  HPC
#

aaa = np.median(ccpvCorrelationCurveHPC1[:,:,:,0],axis = 0)
aaa1 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],75,axis = 0)
aaa2 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],25,axis = 0)


plot(np.linspace(1.0,0.0,morphnum),aaa[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[0,:],color='k')

plot(np.linspace(1.0,0.0,morphnum),aaa[1,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[2,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[5,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[10,:],color='r')
#plot(np.linspace(1.0,0.0,morphnum),aaa[15,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[20,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[30,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[40,:],color='r')

plot(np.linspace(1.0,0.0,morphnum),aaa[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[50,:],color='k')

plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/SVG/Fig3A_HPC' + name_string + '.eps', format='eps', dpi=1000)









# %% with pre run
#
# CARREGA OS ARQUIVOS DE CONVERGENCIA - COM TUDO - TIPO 2
#


morphnum = 41

ct = 1

simulation_num = 1


reference1 = seed_input
reference2 = seed_www
reference3 = mec_ratio
reference4 = hpc_ratio

sublento = (ct+1)*pre_runs + morphnum      
lento = morphnum + sublento * true_runs   
    

r_start = concatenate([[0],morphnum+(arange(true_runs)*sublento)+(ct+1)*pre_runs])   
#r_start = concatenate([[0],(morphnum + (ct+1)*pre_runs) + (arange(true_runs)*morphnum + (ct+1))])
r_end =  r_start.copy() +   morphnum    


size1 = (len(reference4)*len(reference3)*len(reference2)*len(reference1),len(r_start),morphnum,2,theta_cycles)
size2 = (len(reference4)*len(reference3)*len(reference2)*len(reference1),len(r_start),morphnum,2)


ccMECconverge = -1*ones(size1)
ccHPCconverge = -1*ones(size1)
ccMECconvergeDist = -1*ones(size1)
ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

ccpvCorrelationCurveHPC1 = -1*ones(size2)
ccpvCorrelationCurveHPC2 = -1*ones(size2)
ccpvCorrelationCurveMEC1 = -1*ones(size2)
ccpvCorrelationCurveMEC2 = -1*ones(size2) 
ccpvCorrelationCurveLEC1 = -1*ones(size2)
ccpvCorrelationCurveLEC2 = -1*ones(size2)      

ppp = -1

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                       MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (ii4,ii3,ii2,ii1))
                    pass





# %% FIGURA 3A - PV CORR

#
#  MEC
#

aaa = np.median(ccpvCorrelationCurveMEC1[:,:,:,0],axis = 0)
aaa1 = np.percentile(ccpvCorrelationCurveMEC1[:,:,:,0],75,axis = 0)
aaa2 = np.percentile(ccpvCorrelationCurveMEC1[:,:,:,0],25,axis = 0)


plot(np.linspace(1.0,0.0,morphnum),aaa[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[0,:],color='k')

plot(np.linspace(1.0,0.0,morphnum),aaa[1,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[2,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[5,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[10,:],color='r')
#plot(np.linspace(1.0,0.0,morphnum),aaa[15,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[20,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[30,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[40,:],color='r')

plot(np.linspace(1.0,0.0,morphnum),aaa[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[50,:],color='k')

plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/SVG/Fig3B_MEC' + name_string + '.eps', format='eps', dpi=1000)


# %% FIGURA 3A - PV CORR

#
#  HPC
#

aaa = np.median(ccpvCorrelationCurveHPC1[:,:,:,0],axis = 0)
aaa1 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],75,axis = 0)
aaa2 = np.percentile(ccpvCorrelationCurveHPC1[:,:,:,0],25,axis = 0)


plot(np.linspace(1.0,0.0,morphnum),aaa[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[0,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[0,:],color='k')

plot(np.linspace(1.0,0.0,morphnum),aaa[1,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[2,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[5,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[10,:],color='r')
#plot(np.linspace(1.0,0.0,morphnum),aaa[15,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[20,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[30,:],color='r')
plot(np.linspace(1.0,0.0,morphnum),aaa[40,:],color='r')

plot(np.linspace(1.0,0.0,morphnum),aaa[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa1[50,:],color='k')
plot(np.linspace(1.0,0.0,morphnum),aaa2[50,:],color='k')

plt.ylim((-0.1,1.1)) 


plt.savefig('Figures/SVG/Fig3B_HPC' + name_string + '.eps', format='eps', dpi=1000)


# %%

















