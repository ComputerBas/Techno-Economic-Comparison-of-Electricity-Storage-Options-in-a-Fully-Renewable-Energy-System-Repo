# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:40:46 2022

@author: basmu
"""
import numpy as np
import pandas as pd
import sys
import math
from time import time
import matplotlib.pyplot as plt
from itertools import compress

def EPEX_import():
    yearh = 365*24
    EPEX_def1 = pd.read_excel('EPEX.xlsx')
    EPEX_lastyear = np.array(EPEX_def1["€/MWh"])[-yearh:]
    return EPEX_lastyear

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"Check if YCc_max is allowed"
def YCyc_max(EP1,Chargefac1):
    yhours = 24*365
    return yhours/(EP1*(1+Chargefac1))
       

def AVG_PP_adv_EP(EP,YCycles_arr,CharFac,EPEX_def = EPEX_import()):    
    EPEX_del = EPEX_def.copy()   
    
    #resolution of discretisation
    if EP*(1+CharFac) < 1:
        PP_cycle = np.sort(EPEX_del)
        PP_avg_year = np.cumsum(PP_cycle)/range(1,len(PP_cycle)+1)    
        PP_avg = np.interp(YCycles_arr,range(1,len(PP_cycle)+1),PP_avg_year) 
        return PP_avg
    if EP<1:
        res1 = int(math.ceil(1/EP))
        EPEX_interp = np.interp(np.array(range(len(EPEX_del)*res1))/res1,np.array(range(len(EPEX_del))),EPEX_del)
    else:
        res1 = 1
        EPEX_interp = EPEX_del
    #Discrete charge duration
    h_charge = int(math.ceil(EP*CharFac*res1))
    
    #Discrete duration of a year
    #yearh = 365*24*res1
    
    #Discrete duration of cycle
    h_cycle = int(math.ceil((CharFac+1)*EP*res1))
    
    sumseries = list(moving_average(EPEX_interp,h_charge)) 
    
       
    # plt.plot(sumseries, "lightgray")
    # plt.plot(EPEX_interp, "darkgray")
    
    PP_cycle = []
    MaxY = int(math.ceil(np.max(YCycles_arr)))
    idxfree = [True]*len(sumseries)
    for j in range(1,MaxY):  
        
        if all(item is False for item in idxfree) == True:
            PP_cycle.extend([np.average(EPEX_del)]*(MaxY-j))
            break
        
        else:
            #Check when charging is started
            epexmin = min(list(compress(sumseries.copy(), idxfree)))  
            delidx = sumseries.index(epexmin)
            
            #Prevent double indices
            counter = 1
            
            while idxfree[delidx] == False:
                delidx = [i for i, x in enumerate(sumseries) if x == epexmin][counter]
                counter += 1       
                
            #Delete instances of sumseries that are not possible anymore
            idxsumdelmin = delidx-h_cycle+1
            idxsumdelmax = delidx+h_cycle
            
            if idxsumdelmin < 0:
                idxsumdelmin = 0
            
            if idxsumdelmax > len(sumseries):
                idxsumdelmax = len(sumseries)
                     
            idxfree[idxsumdelmin:idxsumdelmax] = [False]*(idxsumdelmax-idxsumdelmin)

#            plt.plot(np.linspace(idxsumdelmin,idxsumdelmax,len(sumseries2[idxsumdelmin:idxsumdelmax])),sumseries2[idxsumdelmin:idxsumdelmax])
    
            PP_cycle.append(epexmin)
            if j%100 == 0:
                print(".", end = "")
    PP_avg_year = np.cumsum(PP_cycle)/range(1,len(PP_cycle)+1) 
    
    PP_avg = np.interp(YCycles_arr,range(1,len(PP_cycle)+1),PP_avg_year)
    print(">")
    return PP_avg

def PP_AVG_adv(EP_arr,YCycles_arr,CharFac):
    PP_avg_2d = np.zeros((len(EP_arr),len(YCycles_arr)))    

    for i in range(len(EP_arr)):
        PP_avg_2d[i,:] =  AVG_PP_adv_EP(EP_arr[i],YCycles_arr,CharFac)
    
    return PP_avg_2d


# t1 = time()
# PP_AVG_adv(EP_arr,YCycles_arr,1.56)
# print(time()-t1)       
