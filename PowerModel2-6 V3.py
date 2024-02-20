# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:24:58 2022

@author: Sebastiaan Mulder
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import colormaps as cm
import pandas as pd
from EPEX_import3 import PP_AVG_adv
import warnings
from time import time
from itertools import compress
import math
from scipy.interpolate import RegularGridInterpolator

#Here LCOE prices (energy buy-in for RLPCP) can be imported as precalculated with the energy storage data
#Changing the energy storage data requires new calculation of the LCOE data, this can take several hours
LCOEimport = True


LCOEimportarr = np.load("LCOE_DE23.npy")
# LCOEimportarr[14,:,:] = LCOEimportarr[12,:,:]

warnings.simplefilter('error')

Storage_data = pd.read_excel('Data_Schmidt4.xlsx')
Tech_names = np.array(Storage_data["Tech"])

"Resolution"
res = 200

"Constants"
#Energy storage
#Irrelevant
PS_in = 100  #MW

EP_arr = np.logspace(-2,10,res,base = 2)
EPlogsteps = math.log(EP_arr[-1],2)-math.log(EP_arr[0],2)
EPlabels = []
for i in np.logspace(math.log(EP_arr[0],2),math.log(EP_arr[-1],2),int(EPlogsteps+1),base=2):
    EPlabels.append(i)

"YCycles is limited to 4000 if RLPCP is used and LCOE is imported"
YCycles_arr = np.logspace(0,3,res)
    
#Financial
ROI_in = 0.08 #Return on investment/interest on loan

PP_advanced = True
PP_fixprice = 0

t1 = time()

minarray = np.zeros(len(Tech_names))
maxarray = np.zeros(len(Tech_names))
minarray_geo = np.zeros(len(Tech_names))
maxarray_geo = np.zeros(len(Tech_names))
minarray2 = np.zeros(len(Tech_names))
maxarray2 = np.zeros(len(Tech_names))
geoidx = [True]*8 + [False]*7

"restrictions"
allidx = [True]*15
greenidx = [True]*8 + [False]*7
caveidx = [True]*8 + [False]*2 + [True]*4 + [False]
mountidx = [True]*8 + [False] + [True] + [False]*5
retrocidx = [True]*9 + [False]*6
retrogidx = [True]*8 + [False]*2 + [True]*5

Case = ["All options", "Greenfield", "Cavern", "Mountains", "Coal retrofit", "Gas retrofit + Cavern"]

# pointsYC = [101,84,74,10,2]
# pointsED = [2, 9.833333, 27.28378, 82, 149.5]

def LCOS_def(E,P,ROI,Storage_data,YCycles,idx,PP_buy):
    
    "Check if EP is allowed"
    def EP_max(YCycles1,Chargefac):
        yhours = 24*365
        return yhours/(YCycles1*(1+Chargefac))
           
    EP_max_value = EP_max(YCycles,Storage_data["Charging Factor"][idx])
    
    if (E/P) > EP_max_value:
        return None
    
    else:
        "Array setup"
        

        P_buy = Storage_data["Charging Factor"][idx]*E*PP_buy
        #Degradation and cycle life
        CyCdeg = 1-np.exp(np.log(Storage_data["End-of-life"][idx])/Storage_data["Cycle life"][idx])
        YLife = min(Storage_data["Cycle life"][idx]/YCycles,Storage_data["Shelf Life"][idx])
        if YLife < 2:
            return None
        #Repairs
        RepSum = 0
        
        if Storage_data["Replacement nterval (c)"][idx]>0:
            Rep = Storage_data["Cycle life"][idx]/Storage_data["Replacement nterval (c)"][idx]
            TRep = Storage_data["Replacement nterval (c)"][idx]/YCycles
            
            if TRep<YLife:
                RepSum = sum((Storage_data["Replacement ($/kWh)"][idx]*E+Storage_data["Replacement ($/kW)"][idx]*P)/((1+ROI)**(Storage_data["Construction time"][idx]+np.array(range(1,int(Rep+1)))*TRep)))
            
        CAPEX = Storage_data["CAPEX Factor"][idx]*(Storage_data["CAPEX ($/kWh)"][idx]*E*1000 + Storage_data["CAPEX ($/kW)"][idx]*P*1000) + RepSum
        
        # if idx == 0:
        #     CAPEX = CAPEX*0.75

        # if idx == 6:
        #     CAPEX = CAPEX*2
            
        OPEX_last = (Storage_data["OPEX ($/MWh)"][idx]*E*YCycles*Storage_data["DoD"][idx]*((1-CyCdeg)**((YLife-1)*YCycles)*(1-Storage_data["Degradation"][idx])**(YLife-1))+Storage_data["OPEX ($/kW)"][idx]*P*1000)/((1+ROI)**(YLife + Storage_data["Construction time"][idx]))
        OPEX = sum((Storage_data["OPEX ($/MWh)"][idx]*E*YCycles*Storage_data["DoD"][idx]*((1-CyCdeg)**(np.array(range(0,int(YLife)))*YCycles)*(1-Storage_data["Degradation"][idx])**np.array(range(0,int(YLife))))+Storage_data["OPEX ($/kW)"][idx]*P*1000)/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+OPEX_last
        
        C_Charge_last = (P_buy*YCycles)/((1+ROI)**YLife + Storage_data["Construction time"][idx])
        C_Charge = sum((P_buy*YCycles)/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+C_Charge_last
        
        EOL_Discounted = (Storage_data["EoL ($/kWh)"][idx]*E*1000 + Storage_data["EoL ($/kW)"][idx]*P*1000)/((1+ROI)**(1+YLife))
        
        
        #Total energy discharged
        DisRemainder = ((1-CyCdeg)**((YLife-1)*YCycles) * (1-Storage_data["Degradation"][idx])**(YLife-1))/((1+ROI)**(YLife + Storage_data["Construction time"][idx])) 
        SumDis = sum(((1-CyCdeg)**(np.array(range(0,int(YLife)))*YCycles) * (1-Storage_data["Degradation"][idx])**np.array(range(0,int(YLife))))/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+DisRemainder  
        SumElecDischarged = YCycles*Storage_data["DoD"][idx]*E*Storage_data["RT"][idx]*(1-Storage_data["Self-Discharge"][idx]) * SumDis
        # LCOS_inner = (CAPEX+OPEX+C_Charge+EOL_Discounted)/(SumElecDischarged) #$/kWh
        LCOS_inner = (CAPEX+OPEX+EOL_Discounted)/(SumElecDischarged) #$/kWh

        ACC = (CAPEX+OPEX+C_Charge+EOL_Discounted)/(YLife*P*1000)
        return LCOS_inner,ACC,CAPEX,OPEX,C_Charge,EOL_Discounted,SumElecDischarged,RepSum,YLife,PP_buy
        

def Tech_LCOS_Matrix(EP_arr1,YCycles_arr1,PS_in1,ROI_in1,Storage_data1,idx1,PPmode,LCOE):

    LCOS_mat = np.zeros((res,res,10))
    for i in range(res):
        for j in range(res):  
            if PPmode == True:
                PP_LCOE = LCOE[i,j]
            else:
                PP_LCOE = PP_fixprice
            LCOS_mat[i,j,:] = LCOS_def(EP_arr1[i]*PS_in1,PS_in1,ROI_in1,Storage_data1,YCycles_arr[j],idx1,PP_LCOE)       
            
    return LCOS_mat


LCOE_savefile = []


"Calling of functions"
for i in range(len(Tech_names)):
    print("Calculating",Tech_names[i])
    if LCOEimport == True and PP_advanced == True:
        LCOE = LCOEimportarr[i,:,:]
    elif PP_advanced == True:
        LCOE = PP_AVG_adv(EP_arr,YCycles_arr,Storage_data["Charging Factor"][i])
        LCOE_savefile.append(LCOE)
    else:
        LCOE = PP_fixprice
    temp = Tech_LCOS_Matrix(EP_arr,YCycles_arr,PS_in,ROI_in,Storage_data,i,PP_advanced,LCOE)
    globals()["LCOS_" + str(Tech_names[i])] = temp[:,:,0] 
    globals()["ACC_" + str(Tech_names[i])] = temp[:,:,1] 
    globals()["PP_" + str(Tech_names[i])] = LCOE
    globals()["Full_" + str(Tech_names[i])] = temp
        
levels = np.array([0.05,0.1,0.2,0.5,1,2,5,10])
t2 = time()

print("Comptime: ", t2-t1, "s")  
    
"Minimum cost calculation"
def MinCost(mode):
    minval = np.zeros((6,res,res))
    mintech = np.zeros((6,res,res))
    
    secminval = np.zeros((6,res,res))
    secmintech = np.zeros((6, res,res))
    
    idxcollection = [allidx,greenidx,caveidx,mountidx,retrocidx,retrogidx]
    #topchoices = []
    
    for k in range(6):
        for i in range(res):
            for j in range(res):
                
                if mode == "LCOS":
                    LCOStemp = [globals()["LCOS_" + z][i,j] for z in Tech_names[idxcollection[k]]]
                
                elif mode == "ACC":
                    LCOStemp = [globals()["ACC_" + z][i,j] for z in Tech_names[idxcollection[k]]]

    
                if np.isnan(LCOStemp).all() == False:
                    minval[k,i,j] = np.nanmin(LCOStemp)
                    
                if minval[k,i,j] == 0:
                    minval[k,i,j] = None
                    mintech[k,i,j] =  None
                
                else:
                    mintech[k,i,j] =  LCOStemp.index(minval[k,i,j]) 
                    
                    valarr = [True]*len(Tech_names)
                    valarr[int(mintech[k,i,j])] = False 
                    
                    if k == 3 and mintech[k,i,j] == 8: 
                        mintech[k,i,j] += 1
                        
                    if (k == 2 or k == 5) and mintech[k,i,j] > 7: 
                        mintech[k,i,j] += 2
                
                if np.isnan(list(compress(LCOStemp, valarr))).all() == False:
                    secminval[k,i,j] = np.nanmin(list(compress(LCOStemp, valarr)))
                
                if secminval[k,i,j] == 0: 
                    secminval[k,i,j] = None
                    secmintech[k,i,j] =  None
                    
                else:
                    secmintech[k,i,j] =  LCOStemp.index(secminval[k,i,j])
                    
                    if k == 3 and secmintech[k,i,j] == 8: 
                        secmintech[k,i,j] += 1
                        
                    if (k == 2 or k == 5) and secmintech[k,i,j] > 7: 
                        secmintech[k,i,j] += 2
        #topchoices.append(np.unique(mintech[k,::]))
    diff = 1 - minval/secminval           
    
    diff[np.isnan(diff)] = 0
    
    ndiff = (diff*20).round()/20
    
    nndiff = np.where(ndiff < 0.15, ndiff, (diff*5).round()/5)
    return mintech,minval,nndiff, secmintech, secminval

mintechLCOS, minvalLCOS, diffLCOS,secmintechLCOS, secminvalLCOS = MinCost("LCOS")
mintechACC, minvalACC, diffACC, secmintechACC, secminvalACC = MinCost("ACC")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

"--------------Market Points------------"
# plotpoints = ((3000,1/60,"FCR"),(1500,5/60,"aFFR"),(300,15/60,"mFFR"),(1500,1,"Intraday"),(700,3,"Day-Ahead"))

pointsYC = [81,111,90,5,1]
pointsED = [2, 10, 24, 93, 133]


def PlotGraphs(mode):
    #Set mode
    if mode == "LCOS":
        mintech = mintechLCOS
        minval = minvalLCOS
        nndiff = diffLCOS
    elif mode == "ACC":
        mintech = mintechACC
        minval = minvalACC
        nndiff = diffACC
    
    "--------------Min Tech-----------------"
    Colfac = 0.8
    
    plt.figure(len(Tech_names)+2)
    fig, ((ax0, ax1) ,(ax2, ax3) , (ax4, ax5)) = plt.subplots(3, 2, figsize = (13,13), sharex=True, sharey=True)
    cmap2 = cm['tab10']
    plt.subplots_adjust(left=0.1, bottom=None, right=0.6, top=None, wspace=None, hspace=None)
    
    
    fig.text(0.5, 0.08, "Yearly cycles [-]", ha='center')
    fig.text(0.04, 0.5, "E/P ratio or discharge time [h]", va='center', rotation='vertical')
    
    if mode == "LCOS":
        if PP_advanced == True:
            plt.suptitle("Cheapest Tech (LCOS) \n ROI = " + str(str(ROI_in)))
        else:
            plt.suptitle("Cheapest Tech (LCOS)\n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
    elif mode == "ACC":
        if PP_advanced == True:
            plt.suptitle("Cheapest Tech (ACC) \n RLPCP Algorithm (Germany 2023)\n ROI = " + str(str(ROI_in)))
        else:
            plt.suptitle("Cheapest Tech (ACC)\n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
          
    
    #Replacements
    replacements = [[4,1],[9,2],[6,3],[14,4],[8,5],[12,6]]
    
    for k, ax in enumerate([ax0,ax1,ax2,ax3,ax4,ax5]):
        mintecha = mintech[k,:,:]
        for r in replacements:
            mintecha = np.where(mintecha == r[0],r[1],mintecha)        
        mintechab = np.where(mintecha < 0, np.nan,np.where(mintecha>6, np.nan, mintecha))
        
        ax.set_title(Case[k])
        
        im = ax.imshow(mintechab, cmap = truncate_colormap(cmap2,0,0.699), origin='lower', vmin=(-0.5), vmax=(7-0.5), interpolation=None, alpha = nndiff[k]**Colfac)
    
        ax.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
        ax.set_xticks(np.linspace(0,res-1,4), [1,10,100,1000])
        
        # for y,d,n in plotpoints:
        #     ax.plot(np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        #     ax.annotate(n,xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(5, -4.5), textcoords='offset points')        
        
        # if mode == "LCOS":           
         
        #     for y,d,n in plotpoints[-2:]:
        #         ax.plot(np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        #         if n != "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(-15, 6), textcoords='offset points', fontsize = 8)      
        #         elif n == "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(0, 6), textcoords='offset points', fontsize = 8)                  
            
        # elif mode == "ACC":

        #     for y,d,n in plotpoints[:3]:
        #         ax.plot(np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        #         if n != "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(-15, 6), textcoords='offset points', fontsize = 8)      
        #         elif n == "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(0, 6), textcoords='offset points', fontsize = 8)                      
   
        
        for i in range(5):
            ax.plot(np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
            ax.annotate(str(i+1),xy = (np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200))),color = "black", xytext=(5, -4.5), textcoords='offset points')        
    
    
    cb_ax = fig.add_axes([0.66, 0.375, 0.25, 0.25])
    colorbarplot = np.array([[0,0,0,0,0,0],[1.1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6.1,6,6,6,6,6]])
    alphaplot = np.array([[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05]])
    cb_ax.imshow(colorbarplot, cmap = truncate_colormap(cmap2,0,0.699), alpha = alphaplot**Colfac)
    if mode == "LCOS":
        cb_ax.set_title("LCOS difference \n with second cheapest solution")
    elif mode == "ACC":
        cb_ax.set_title("ACC difference \n with second cheapest solution")
    cb_ax.set_xticks(ticks = np.linspace(0,5,6),labels = ["80%","60%","40%","20%","10%","5%"])
    cb_ax.set_yticks(ticks = np.linspace(0,6,7),labels = [Tech_names[0],Tech_names[4],Tech_names[9],Tech_names[6],Tech_names[14],Tech_names[8],Tech_names[12]])
    
    
    plt.show()
    plt.figure()
    
    "---------------- Min LCOS ----------------------"
    
    plt.figure(len(Tech_names)+3)
    fig, ((ax0, ax1) ,(ax2, ax3) , (ax4, ax5)) = plt.subplots(3, 2, figsize = (8,12), sharex=True, sharey=True)
    
    plt.subplots_adjust(left=0.1, bottom=None, right=0.85, top=None, wspace=None, hspace=None)
    fig.text(0.5, 0.08, "Yearly cycles [-]", ha='center')
    fig.text(0.04, 0.5, "E/P ratio or discharge time [h]", va='center', rotation='vertical')
    
    for k, ax in enumerate([ax0,ax1,ax2,ax3,ax4,ax5]):
        ax.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
        ax.set_xticks(np.linspace(0,res-1,4), [1,10,100,1000])
        ax.set_title(Case[k])
        if mode == "LCOS":           
            im = ax.contourf(minval[k],levels=1000*levels, cmap = "plasma_r",vmin = 100, vmax=10000, norm=colors.LogNorm())
            
        #     for y,d,n in plotpoints[-2:]:
        #         ax.plot(np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        #         if n != "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(-15, 6), textcoords='offset points', fontsize = 8)      
        #         elif n == "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(0, 6), textcoords='offset points', fontsize = 8)                  
            
        # elif mode == "ACC":
        #     im = ax.contourf(minval[k],levels=100*levels, cmap = "plasma_r",vmin = 5, vmax = 1000, norm=colors.LogNorm())
            
        #     for y,d,n in plotpoints[:3]:
        #         ax.plot(np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        #         if n != "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(-15, 6), textcoords='offset points', fontsize = 8)      
        #         elif n == "FCR":
        #             ax.annotate(n + "\n" + str(round(minval[k,int(np.interp(d,EP_arr,np.linspace(0,199,200))),int(np.interp(y,YCycles_arr,np.linspace(0,199,200)))],2)),xy = (np.interp(y,YCycles_arr,np.linspace(0,199,200)), np.interp(d,EP_arr,np.linspace(0,199,200))),color = "black", xytext=(0, 6), textcoords='offset points', fontsize = 8)                      

        for i in range(5):
            ax.plot(np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
            ax.annotate(str(i+1),xy = (np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200))),color = "black", xytext=(5, -4.5), textcoords='offset points')        
        
    
    cb_ax = fig.add_axes([0.9, 0.1, 0.04, 0.85])     
 

    if mode == "LCOS":
        cbar = fig.colorbar(im, cax=cb_ax,label = "LCOS [$/MWh]")  
        cbar.set_ticks([100,200,500,1000,2000,5000,10000])
        cbar.set_ticklabels(["100","200","500","1000","2000","5000","10000"])
        
        if PP_advanced == True:
            fig.suptitle("LCOS of cheapest tech \n RLPCP Algorithm (Germany 2023)\n ROI = " + str(str(ROI_in)))
        else:
            fig.suptitle("LCOS of cheapest tech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
    elif mode == "ACC":
        cbar = fig.colorbar(im, cax=cb_ax,label = "ACC [$/kW*y]")
        cbar.set_ticks([10,20,50,100,200,500,1000])
        cbar.set_ticklabels(["10","20","50","100","200","500","1000"])
        if PP_advanced == True:
            fig.suptitle("ACC of cheapest tech \n RLPCP Algorithm (Germany 2023)\n ROI = " + str(str(ROI_in)))
        else:
            fig.suptitle("ACC of cheapest tech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
                      
    return None


# Interpolate points
pointsarr = np.zeros((5,len(Tech_names)))

for i in range(len(Tech_names)):
    interp = RegularGridInterpolator((EP_arr,YCycles_arr), globals()["LCOS_" + str(Tech_names[i])])
    
    for j in range(5):
        pointsarr[j,i] = interp([pointsED[j],pointsYC[j]])

PlotGraphs("LCOS")
PlotGraphs("ACC")


#Second Best Plotting
plt.figure(len(Tech_names)+2)
fig, ((ax0, ax1) ,(ax2, ax3) , (ax4, ax5)) = plt.subplots(3, 2, figsize = (8,12), sharex=True, sharey=True)
plt.subplots_adjust(left=0.1, bottom=None, right=0.85, top=None, wspace=None, hspace=None)


fig.text(0.5, 0.08, "Yearly cycles [-]", ha='center')
fig.text(0.04, 0.5, "E/P ratio or discharge time [h]", va='center', rotation='vertical')

if PP_advanced == True:
    plt.suptitle("Second Cheapest Tech (LCOS) \n RLPCP Algorithm (Germany 2023)\n ROI = " + str(str(ROI_in)))
else:
    plt.suptitle("Second Cheaperst Tech (LCOS)\n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))


for k, ax in enumerate([ax0,ax1,ax2,ax3,ax4,ax5]):
    mintecha = secmintechLCOS[k,:,:]
    
    ax.set_title(Case[k])
    
    im = ax.imshow(mintecha, cmap = truncate_colormap(cm['tab20'],0,(15/20)-0.001), origin='lower', interpolation=None, vmin = 0, vmax = 14)

    ax.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
    ax.set_xticks(np.linspace(0,res-1,4), [1,10,100,1000])

cb_ax = fig.add_axes([0.9, 0.1, 0.04, 0.85])   
cbar = fig.colorbar(im, cax=cb_ax,label = "Second Best Tech")  
cbar.set_ticks([i for i in range(15)])
cbar.set_ticklabels([str(i) for i in Tech_names])

plt.show()
plt.figure()