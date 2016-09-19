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

seed_input = arange(11)
seed_www = arange(1)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 10

hpc_pcompl_th = 80


# %% will the two curves converge??


seed_path = 0

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

#mec_ratio = [99]
#hpc_ratio = [99]

hpc_pcompl_th = 80
morph_per = 100

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
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1),len(r_start),morphnum,2)


#ccMECconverge = -1*ones(size1)
#ccHPCconverge = -1*ones(size1)
#ccMECconvergeDist = -1*ones(size1)
#ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

#ccpvCorrelationCurveHPC1 = -1*ones(size2)
#ccpvCorrelationCurveHPC2 = -1*ones(size2)
#ccpvCorrelationCurveMEC1 = -1*ones(size2)
#ccpvCorrelationCurveMEC2 = -1*ones(size2) 
#ccpvCorrelationCurveLEC1 = -1*ones(size2)
#ccpvCorrelationCurveLEC2 = -1*ones(size2)      


nices = [0,sublento * true_runs + 1]

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                    #   MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        #ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        #ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ii4,ii3,ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ii4,ii3,ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ii4,ii3,ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ii4,ii3,ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
#                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                    pass








# %%  Average number of cycles < 4 - Naive

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,0,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,0,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)


plt.savefig('Figures/rennocostatort_elife_fig3s1A.eps', format='eps', dpi=1000)

# %%  Average number of cycles < 4 - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)


plt.savefig('Figures/rennocostatort_elife_fig3s1B1.eps', format='eps', dpi=1000)





# %% Average number of cycles to converge - Naive

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,0,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,0,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)


plt.savefig('Figures/rennocostatort_elife_fig3s1Anu.eps', format='eps', dpi=1000)


# %% Average number of cycles to converge - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)


plt.savefig('Figures/rennocostatort_elife_fig3s1B1nu.eps', format='eps', dpi=1000)









# %%

seed_input = arange(11)
seed_www = arange(1)
seed_path = 0

lrate_hpc_mec = 100
lrate_ec_hpc = 00

hpc_pcompl_th = 80


# %% will the two curves converge??


seed_path = 0

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

#mec_ratio = [99]
#hpc_ratio = [99]

hpc_pcompl_th = 80
morph_per = 100

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
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1),len(r_start),morphnum,2)


#ccMECconverge = -1*ones(size1)
#ccHPCconverge = -1*ones(size1)
#ccMECconvergeDist = -1*ones(size1)
#ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

#ccpvCorrelationCurveHPC1 = -1*ones(size2)
#ccpvCorrelationCurveHPC2 = -1*ones(size2)
#ccpvCorrelationCurveMEC1 = -1*ones(size2)
#ccpvCorrelationCurveMEC2 = -1*ones(size2) 
#ccpvCorrelationCurveLEC1 = -1*ones(size2)
#ccpvCorrelationCurveLEC2 = -1*ones(size2)      


nices = [0,sublento * true_runs + 1]

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                    #   MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        #ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        #ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ii4,ii3,ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ii4,ii3,ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ii4,ii3,ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ii4,ii3,ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
#                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                    pass






# %%  Average number of cycles < 7 - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)


plt.savefig('Figures/rennocostatort_elife_fig3s1B2.eps', format='eps', dpi=1000)



# %% Average number of cycles to converge - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)


plt.savefig('Figures/rennocostatort_elife_fig3s1B2nu.eps', format='eps', dpi=1000)









# %%

seed_input = arange(11)
seed_www = arange(1)
seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 10

hpc_pcompl_th = 80


# %% will the two curves converge??


seed_path = 0

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

#mec_ratio = [99]
#hpc_ratio = [99]

hpc_pcompl_th = 80
morph_per = 100

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
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1),len(r_start),morphnum,2)


#ccMECconverge = -1*ones(size1)
#ccHPCconverge = -1*ones(size1)
#ccMECconvergeDist = -1*ones(size1)
#ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

#ccpvCorrelationCurveHPC1 = -1*ones(size2)
#ccpvCorrelationCurveHPC2 = -1*ones(size2)
#ccpvCorrelationCurveMEC1 = -1*ones(size2)
#ccpvCorrelationCurveMEC2 = -1*ones(size2) 
#ccpvCorrelationCurveLEC1 = -1*ones(size2)
#ccpvCorrelationCurveLEC2 = -1*ones(size2)      


nices = [0,sublento * true_runs + 1]

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                    #   MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        #ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        #ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ii4,ii3,ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ii4,ii3,ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ii4,ii3,ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ii4,ii3,ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
#                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                    pass






# %%  Average number of cycles < 7 - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)


plt.savefig('Figures/rennocostatort_elife_fig3s1B3.eps', format='eps', dpi=1000)



# %% Average number of cycles to converge - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)


plt.savefig('Figures/rennocostatort_elife_fig3s1B3nu.eps', format='eps', dpi=1000)












# %%

seed_input = arange(11)
seed_www = arange(1)
seed_path = 0

lrate_hpc_mec = 00
lrate_ec_hpc = 00

hpc_pcompl_th = 80


# %% will the two curves converge??


seed_path = 0

theta_cycles = 7
arena_runs = 1
pre_runs = 5  # 50
true_runs = 50  #1

mec_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]
hpc_ratio = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,99]

#mec_ratio = [99]
#hpc_ratio = [99]

hpc_pcompl_th = 80
morph_per = 100

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
size2 = (len(reference4),len(reference3),len(reference2)*len(reference1),len(r_start),morphnum,2)


#ccMECconverge = -1*ones(size1)
#ccHPCconverge = -1*ones(size1)
#ccMECconvergeDist = -1*ones(size1)
#ccHPCconvergeDist = -1*ones(size1)

ccMECconvergetime = -1*ones(size2)
ccHPCconvergetime = -1*ones(size2)
ccMECconvergeDisttime = -1*ones(size2)
ccHPCconvergeDisttime = -1*ones(size2)

#ccpvCorrelationCurveHPC1 = -1*ones(size2)
#ccpvCorrelationCurveHPC2 = -1*ones(size2)
#ccpvCorrelationCurveMEC1 = -1*ones(size2)
#ccpvCorrelationCurveMEC2 = -1*ones(size2) 
#ccpvCorrelationCurveLEC1 = -1*ones(size2)
#ccpvCorrelationCurveLEC2 = -1*ones(size2)      


nices = [0,sublento * true_runs + 1]

for ii4 in arange(len(reference4)):
    for ii3 in arange(len(reference3)):
        ppp = -1
        for ii2 in arange(len(reference2)):
            for ii1 in arange(len(reference1)):
    
                listofvalues = [ct,seed_input[ii1],seed_www[ii2],seed_path,theta_cycles,arena_runs,pre_runs,true_runs,lrate_hpc_mec,lrate_ec_hpc,lrate_ec_hpc,mec_ratio[ii3],hpc_ratio[ii4],hpc_pcompl_th,morph_per]
                
                ppp = ppp + 1
                
                try:
                    
                    #with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,1)+'z', 'rb') as ff:
                    #   MECconverge,HPCconverge,MECconvergeDist,HPCconvergeDist = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,2)+'z', 'rb') as ff:
                       pvCorrelationCurveHPC1,pvCorrelationCurveHPC2,pvCorrelationCurveMEC1,pvCorrelationCurveMEC2,pvCorrelationCurveLEC1,pvCorrelationCurveLEC2 = pickle.load(ff)
                    with gzip.open(filenames.fileRunPickle(listofvalues,simulation_num,3)+'z', 'rb') as ff:
                       MECconvergetime,HPCconvergetime,MECconvergeDisttime,HPCconvergeDisttime,convergethetatime = pickle.load(ff)

                    for zz in arange(len(r_start)):

                        #ccMECconverge[ppp,zz,:] = MECconverge[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconverge[ppp,zz,:] = HPCconverge[r_start[zz]:r_end[zz],:,:]
                        #ccMECconvergeDist[ppp,zz,:] = MECconvergeDist[r_start[zz]:r_end[zz],:,:]
                        #ccHPCconvergeDist[ppp,zz,:] = HPCconvergeDist[r_start[zz]:r_end[zz],:,:]
                        
                              
                        ccMECconvergetime[ii4,ii3,ppp,zz,:] = MECconvergetime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergetime[ii4,ii3,ppp,zz,:] = HPCconvergetime[r_start[zz]:r_end[zz],:]
                        ccMECconvergeDisttime[ii4,ii3,ppp,zz,:] = MECconvergeDisttime[r_start[zz]:r_end[zz],:]
                        ccHPCconvergeDisttime[ii4,ii3,ppp,zz,:] = HPCconvergeDisttime[r_start[zz]:r_end[zz],:]
                        
#                        ccpvCorrelationCurveHPC1[ppp,zz,:] = pvCorrelationCurveHPC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveHPC2[ppp,zz,:] = pvCorrelationCurveHPC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC1[ppp,zz,:] = pvCorrelationCurveMEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveMEC2[ppp,zz,:] = pvCorrelationCurveMEC2[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC1[ppp,zz,:] = pvCorrelationCurveLEC1[zz,:,:].transpose()
#                        ccpvCorrelationCurveLEC2[ppp,zz,:] = pvCorrelationCurveLEC2[zz,:,:].transpose()

                except:
                    print("%d %d %d %d" % (reference4[ii4],reference3[ii3],reference1[ii1],reference2[ii2]))
                    
                    pass






# %%  Average number of cycles < 7 - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0]<=4,axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=1)


plt.savefig('Figures/rennocostatort_elife_fig3s1B4.eps', format='eps', dpi=1000)



# %% Average number of cycles to converge - After Learning

rrr = reference3;
rrr2 = reference4;

plt.figure()

ccMECconvergetime[ccMECconvergetime<0] = 30
ccHPCconvergetime[ccHPCconvergetime<0] = 30

subplot(1,2,1)
aa = np.mean(ccMECconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)

subplot(1,2,2) 
aa = np.mean(ccHPCconvergetime[:,:,:,len(r_start)-1,0,0],axis=2);
plt.pcolormesh(np.array(rrr),np.array(rrr2),aa,vmin = 0, vmax=30)


plt.savefig('Figures/rennocostatort_elife_fig3s1B4nu.eps', format='eps', dpi=1000)
































