# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:24:58 2022

@author: Sebastiaan Mulder
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
import pandas as pd
from EPEX_import3 import PP_AVG_adv
import warnings
from time import time
from itertools import compress

#Here LCOE prices (energy buy-in for RLPCP) can be imported as precalculated with the energy storage data
#Changing the energy storage data requires new calculation of the LCOE data, this can take several hours
LCOEimport = True


LCOEimportarr = np.load("LCOE4.npy")
LCOEimportarr[14,:,:] = LCOEimportarr[12,:,:]

warnings.simplefilter('error')

Storage_data = pd.read_excel('Data_Schmidt3.xlsx')
Tech_names = np.array(Storage_data["Tech"])

"Resolution"
res = 200

"Constants"
#Energy storage
#Irrelevant
PS_in = 10  #MW

EP_arr = np.logspace(-1,10,res,base = 2)

"YCycles is limited to 4000 if RLPCP is used and LCOE is imported"
YCycles_arr = np.logspace(0,4,res)
    
#Financial
ROI_in = 0.08 #Return on investment/interest on loan

PP_advanced = True
PP_fixprice = 167.5

t1 = time()

minarray = np.zeros(len(Tech_names))
maxarray = np.zeros(len(Tech_names))
minarray_geo = np.zeros(len(Tech_names))
maxarray_geo = np.zeros(len(Tech_names))
minarray2 = np.zeros(len(Tech_names))
maxarray2 = np.zeros(len(Tech_names))
figsize_uni = (14,5)
figsize_uni2 = (16,5)
geoidx = [True]*8 + [False]*7

"restrictions"
allidx = [True]*15
greenidx = [True]*8 + [False]*7
caveidx = [True]*8 + [False]*2 + [True]*4 + [False]
mountidx = [True]*8 + [False] + [True] + [False]*5
retrocidx = [True]*9 + [False]*6
retrogidx = [True]*8 + [False]*2 + [True]*5

Case = ["All options", "Greenfield", "Cavern", "Mountains", "Coal retrofit", "Gas retrofit + Cavern"]

pointsYC = [101,84,74,10,2]
pointsED = [2, 9.833333, 27.28378, 82, 149.5]


def Interp2d2(x_val,y_val,x,y,z):

    # Find the nearest grid points to the target point
    x_idx = np.abs(x - x_val).argmin()
    y_idx = np.abs(y - y_val).argmin()
    print(j, x_idx,y_idx)
    return z[x_idx,y_idx]

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
        
        OPEX_last = (Storage_data["OPEX ($/MWh)"][idx]*E*YCycles*Storage_data["DoD"][idx]*((1-CyCdeg)**((YLife-1)*YCycles)*(1-Storage_data["Degradation"][idx])**(YLife-1))+Storage_data["OPEX ($/kW)"][idx]*P*1000)/((1+ROI)**(YLife + Storage_data["Construction time"][idx]))
        OPEX = sum((Storage_data["OPEX ($/MWh)"][idx]*E*YCycles*Storage_data["DoD"][idx]*((1-CyCdeg)**(np.array(range(0,int(YLife)))*YCycles)*(1-Storage_data["Degradation"][idx])**np.array(range(0,int(YLife))))+Storage_data["OPEX ($/kW)"][idx]*P*1000)/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+OPEX_last
        
        C_Charge_last = (P_buy*YCycles)/((1+ROI)**YLife + Storage_data["Construction time"][idx])
        C_Charge = sum((P_buy*YCycles)/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+C_Charge_last
        
        EOL_Discounted = (Storage_data["EoL ($/kWh)"][idx]*E*1000 + Storage_data["EoL ($/kW)"][idx]*P*1000)/((1+ROI)**(1+YLife))
        
        
        #Total energy discharged
        DisRemainder = ((1-CyCdeg)**((YLife-1)*YCycles) * (1-Storage_data["Degradation"][idx])**(YLife-1))/((1+ROI)**(YLife + Storage_data["Construction time"][idx])) 
        SumDis = sum(((1-CyCdeg)**(np.array(range(0,int(YLife)))*YCycles) * (1-Storage_data["Degradation"][idx])**np.array(range(0,int(YLife))))/((1+ROI)**(np.array(range(1,int(YLife+1))) + Storage_data["Construction time"][idx])))+DisRemainder  
        SumElecDischarged = YCycles*Storage_data["DoD"][idx]*E*Storage_data["RT"][idx]*(1-Storage_data["Self-Discharge"][idx]) * SumDis
        
        LCOS_inner = (CAPEX+OPEX+C_Charge+EOL_Discounted)/(SumElecDischarged) #$/kWh
        
        return LCOS_inner,CAPEX,OPEX,C_Charge,EOL_Discounted,SumElecDischarged,RepSum,YLife,PP_buy
        

def Tech_LCOS_Matrix(EP_arr1,YCycles_arr1,PS_in1,ROI_in1,Storage_data1,idx1,PPmode,LCOE):

    LCOS_mat = np.zeros((res,res,9))
    for i in range(res):
        for j in range(res):  
            if PPmode == True:
                PP_LCOE = LCOE[i,j]
            else:
                PP_LCOE = PP_fixprice
            LCOS_mat[i,j,:] = LCOS_def(EP_arr1[i]*PS_in1,PS_in1,ROI_in1,Storage_data1,YCycles_arr[j],idx1,PP_LCOE)       
            
    return LCOS_mat

"Calling of functions"
for i in range(len(Tech_names)):
    print("Calculating",Tech_names[i])
    if LCOEimport == True and PP_advanced == True:
        LCOE = LCOEimportarr[i,:,:]
    elif PP_advanced == True:
        LCOE = PP_AVG_adv(EP_arr,YCycles_arr,Storage_data["Charging Factor"][i])
    else:
        LCOE = PP_fixprice
    temp = Tech_LCOS_Matrix(EP_arr,YCycles_arr,PS_in,ROI_in,Storage_data,i,PP_advanced,LCOE)
    globals()["LCOS_" + str(Tech_names[i])] = temp[:,:,0] 
    globals()["PP_" + str(Tech_names[i])] = LCOE
    globals()["Full_" + str(Tech_names[i])] = temp
    minarray[i] = np.nanmin(temp[:,:,0])
    maxarray[i] = np.nanmax(temp[:,:,0])
    
    minarray2[i] = np.nanmin(LCOE)
    maxarray2[i] = np.nanmax(LCOE)
    
maxlcos = np.max(maxarray)
maxlcos = 10
minlcos = np.min(minarray)
minlcos = 0.1
levels = np.array([0.05,0.1,0.2,0.5,1,2,5,10])

maxlcos_geo = np.max(maxarray[geoidx])
maxlcos_geo = 10
minlcos_geo = np.min(minarray[geoidx])


maxpp = np.max(maxarray2)
minpp = np.min(minarray2)

t2 = time()

print("Comptime: ", t2-t1, "s")  
    
"Minimum cost calculation"
minval = np.zeros((6,res,res))
mintech = np.zeros((6,res,res))

secminval = np.zeros((6,res,res))
secmintech = np.zeros((6, res,res))

idxcollection = [allidx,greenidx,caveidx,mountidx,retrocidx,retrogidx]
topchoices = []

for k in range(6):
    for i in range(res):
        for j in range(res):
    
            LCOStemp = [globals()["LCOS_" + z][i,j] for z in Tech_names[idxcollection[k]]]

            if np.isnan(LCOStemp).all() == False:
                minval[k,i,j] = np.nanmin(LCOStemp)
                
            if minval[k,i,j] == 0:
                minval[k,i,j] = None
                mintech[k,i,j] =  None
            
            else:
                mintech[k,i,j] =  LCOStemp.index(minval[k,i,j])  
                
                valarr = [True]*len(Tech_names)
                valarr[int(mintech[k,i,j])] = False
            
            if np.isnan(list(compress(LCOStemp, valarr))).all() == False:
                secminval[k,i,j] = np.nanmin(list(compress(LCOStemp, valarr)))
            
            if secminval[k,i,j] == 0: 
                secminval[k,i,j] = None
                secmintech[k,i,j] =  None
                
            else:
                secmintech[k,i,j] =  LCOStemp.index(secminval[k,i,j])             
    topchoices.append(np.unique(mintech[k,::]))
diff = 1 - minval/secminval           

diff[np.isnan(diff)] = 0

ndiff = (diff*20).round()/20

nndiff = np.where(ndiff < 0.15, ndiff, (diff*5).round()/5)

# #Min LCOS

# plt.figure(len(Tech_names)+1)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize_uni)
# ima = ax1.contourf(minval, levels= levels, vmin = minlcos, vmax = maxlcos, norm=colors.LogNorm())
# if PP_advanced == True:
#     ax1.set_title("Minimum LCOS for tech considered - RLPCP Algorithm\n ROI = " + str(str(ROI_in)))
# else:
#     ax1.set_title("Minimum LCOS for tech considered - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n ROI = " + str(str(ROI_in)))
#     ax1.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "red") #Li
#     ax1.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "red") #Hy
#     ax1.legend()
# ax1.set_ylabel("E/P ratio or discharge time [h]")
# ax1.set_xlabel("Yearly cycles [#]")
# ax1.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax1.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# # ax1.clim(0.1,10)
# plt.colorbar(ima,label = "LCOS [$/kWh]", ax = ax1)



# imb= plt.contourf(minval[1,:,:],levels = levels, vmin = minlcos, vmax = maxlcos, norm=colors.LogNorm())

# if PP_advanced == True:
#     ax2.set_title("Minimum LCOS for tech considered - RLPCP Algorithm \n Geographically independant\n ROI = " + str(str(ROI_in)))
# else:
#     ax2.set_title("Minimum LCOS for tech considered - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n Geographically independant\n ROI = " + str(str(ROI_in)))
#     ax2.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "red") #Li
#     ax2.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "red") #Hy
#     ax2.legend()
# ax2.set_ylabel("E/P ratio or discharge time [h]")
# ax2.set_xlabel("Yearly cycles [#]")
# ax2.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax2.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# plt.colorbar(imb,label = "LCOS [$/kWh]", ax=ax2)
# # ax2.set_clim(0.1,10)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# #Best Tech
# plt.figure(len(Tech_names)+2)
# fig, (ax3, ax4) = plt.subplots(1, 2, figsize = figsize_uni)
# cmap = cm.get_cmap('tab20')

# ima = ax3.imshow(mintech, cmap = truncate_colormap(cmap,0,(len(Tech_names)/20)), origin='lower', vmin=(-0.5), vmax=(len(Tech_names)-0.5), interpolation=None)
# if PP_advanced == True:
#     ax3.set_title("Cheapest Tech \n RLPCP Algorithm\n ROI = " + str(str(ROI_in)))
# else:
#     ax3.set_title("Cheapest Tech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
# ax3.set_ylabel("E/P ratio or discharge time [h]")
# ax3.set_xlabel("Yearly cycles [#]")
# ax3.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
# ax3.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# cbar = fig.colorbar(ima, ax = ax3, ticks= [n for n in range(len(Tech_names))])
# cbar.ax.set_yticklabels(Tech_names)


# imb = ax4.imshow(mintech2, cmap = truncate_colormap(cmap,0,(len(Tech_names[geoidx])/20)), origin='lower', vmin=(-0.5), vmax=(len(Tech_names[geoidx])-0.5), interpolation=None)
# if PP_advanced == True:
#     ax4.set_title("Cheapest Tech \n RLPCP Algorithm \n Geographically independant\n ROI = " + str(str(ROI_in)))
# else:
#     ax4.set_title("Cheapest Tech \n Buy-in fixed at: "+ str(PP_fixprice) + "$/MWh \n Geographically independant\n ROI = " + str(str(ROI_in)))
# ax4.set_ylabel("E/P ratio or discharge time [h]")
# ax4.set_xlabel("Yearly cycles [#]")
# ax4.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
# ax4.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# cbar = fig.colorbar(imb, ax = ax4, ticks= [n for n in range(len(Tech_names[geoidx]))])
# cbar.ax.set_yticklabels(Tech_names[geoidx])

# #Second best tech
# plt.figure(len(Tech_names)+3)
# fig, (ax3, ax4) = plt.subplots(1, 2, figsize = figsize_uni)
# cmap = cm.get_cmap('tab20')

# ima = ax3.imshow(secmintech, cmap = truncate_colormap(cmap,0,(len(Tech_names)/20)), origin='lower', vmin=(-0.5), vmax=(len(Tech_names)-0.5), interpolation=None)
# if PP_advanced == True:
#     ax3.set_title("2nd cheapest Tech \n RLPCP Algorithm \n ROI = " + str(str(ROI_in)))
# else:
#     ax3.set_title("Cheapest Tech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))
# ax3.set_ylabel("E/P ratio or discharge time [h]")
# ax3.set_xlabel("Yearly cycles [#]")
# ax3.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
# ax3.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# cbar = fig.colorbar(ima, ax = ax3, ticks= [n for n in range(len(Tech_names))])
# cbar.ax.set_yticklabels(Tech_names)


# imb = ax4.imshow(secmintech2, cmap = truncate_colormap(cmap,0,(len(Tech_names[geoidx])/20)), origin='lower', vmin=(-0.5), vmax=(len(Tech_names[geoidx])-0.5), interpolation=None)
# if PP_advanced == True:
#     ax4.set_title("2nd cheapest Tech \n RLPCP Algorithm \n Geographically independant \n ROI = " + str(str(ROI_in)))
# else:
#     ax4.set_title("Cheapest Tech \n Buy-in fixed at: "+ str(PP_fixprice) + "$/MWh \n Geographically independant\n ROI = " + str(str(ROI_in)))
# ax4.set_ylabel("E/P ratio or discharge time [h]")
# ax4.set_xlabel("Yearly cycles [#]")
# ax4.set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
# ax4.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# cbar = fig.colorbar(imb, ax = ax4, ticks= [n for n in range(len(Tech_names[geoidx]))])
# cbar.ax.set_yticklabels(Tech_names[geoidx])

# #second LCOS


# plt.figure(len(Tech_names)+4)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize_uni)
# ima = ax1.contourf(secminval, levels= levels, vmin = minlcos, vmax = maxlcos, norm=colors.LogNorm())
# if PP_advanced == True:
#     ax1.set_title("Minimum LCOS for tech considered - RLPCP Algorithm\n ROI = " + str(str(ROI_in)))
# else:
#     ax1.set_title("Minimum LCOS for tech considered - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n ROI = " + str(str(ROI_in)))
#     # ax1.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "orange") #Li
#     # ax1.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "orange") #Hy
#     ax1.legend()
# ax1.set_ylabel("E/P ratio or discharge time [h]")
# ax1.set_xlabel("Yearly cycles [#]")
# ax1.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax1.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# # ax1.clim(0.1,10)
# plt.colorbar(ima,label = "LCOS [$/kWh]", ax = ax1)



# imb= plt.contourf(secminval[1,:,:],levels = levels, vmin = minlcos, vmax = maxlcos, norm=colors.LogNorm())

# if PP_advanced == True:
#     ax2.set_title("Minimum LCOS for tech considered - RLPCP Algorithm \n Geographically independant\n ROI = " + str(str(ROI_in)))
# else:
#     ax2.set_title("Minimum LCOS for tech considered - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n Geographically independant")
#     # ax2.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "orange") #Li
#     # ax2.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "orange") #Hy
#     ax2.legend()
# ax2.set_ylabel("E/P ratio or discharge time [h]")
# ax2.set_xlabel("Yearly cycles [#]")
# ax2.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax2.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# plt.colorbar(imb,label = "LCOS [$/kWh]", ax=ax2)
# # ax2.set_clim(0.1,10)

# #difference LCOS
# levels2 = np.linspace(0,1,11)
# plt.figure(len(Tech_names)+4)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize_uni)
# ima = ax1.contourf(np.where(diff[0,:,:]==0,np.nan,diff[0,:,:]), levels = levels2)
# if PP_advanced == True:
#     ax1.set_title("Difference between best and second-best LCOS \n ROI = " + str(str(ROI_in)))
# else:
#     ax1.set_title("Difference between best and second-best LCOS - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n ROI = " + str(str(ROI_in)))
#     ax1.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "orange") #Li
#     ax1.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "orange") #Hy
#     ax1.legend()
# ax1.set_ylabel("E/P ratio or discharge time [h]")
# ax1.set_xlabel("Yearly cycles [#]")
# ax1.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax1.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# # ax1.clim(0.1,10)
# plt.colorbar(ima,label = "Difference [-]", ax = ax1)



# imb= plt.contourf(np.where(diff[1,:,:]==0,np.nan,diff[1,:,:]), levels = levels2)

# if PP_advanced == True:
#     ax2.set_title("Difference between best and second-best LCOSl \n Geographically independant \n ROI = " + str(str(ROI_in)))
# else:
#     ax2.set_title("Difference between best and second-best LCOS - Buy-in price fixed at: " + str(PP_fixprice) + "$/MWh \n Geographically independant")
#     ax2.contour(LCOEimportarr[0,:,:],[PP_fixprice],colors = "orange") #Li
#     ax2.contour(LCOEimportarr[7,:,:],[PP_fixprice],colors = "orange") #Hy
#     ax2.legend()
# ax2.set_ylabel("E/P ratio or discharge time [h]")
# ax2.set_xlabel("Yearly cycles [#]")
# ax2.set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
# ax2.set_yticks(np.linspace(0,res-1,5), [0.1,1,10,100,1000])
# plt.colorbar(imb,label = "Difference [-]", ax=ax2)


# #New Colors

# PHS = np.where(mintech == 9, 2, mintech)
# PTES = np.where(mintech == 6, 3, np.where(mintech == 9, 2, mintech))

Colfac = 0.8

plt.figure(len(Tech_names)+2)
fig, ((ax3, ax4) ,(ax5, ax6) , (ax7,ax8)) = plt.subplots(3, 2, figsize = (13,13), sharex=True, sharey=True)
cmap2 = cm.get_cmap('tab10')

plotnr = ["ax3","ax4","ax5","ax6","ax7", "ax8", "ax9", "ax10", "ax11"]

fig.text(0.5, 0.04, "Yearly cycles [#]", ha='center')
fig.text(0.04, 0.5, "E/P ratio or discharge time [h]", va='center', rotation='vertical')

if PP_advanced == True:
    plt.suptitle("Cheapest Tech \n RLPCP Algorithm \n ROI = " + str(str(ROI_in)))
else:
    plt.suptitle("Cheapest Tech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))

topchoices2 = []


for k in range(6):
    mintecha = np.where(mintech[k,:,:] == 12,6,np.where(mintech[k,:,:] == 8,5,np.where(mintech[k,:,:] == 14,4,np.where(mintech[k,:,:] == 6, 3, np.where(mintech[k,:,:] == 9, 2, mintech[k,:,:])))))
    mintechab = np.where(mintecha < 0, np.nan,np.where(mintecha > 6, np.nan, mintecha).astype(int))
    
    topchoices2.append(np.unique(mintechab))
    globals()[plotnr[k]].set_title(Case[k])
    
    if k == 0:
        ima = globals()[plotnr[k]].imshow(mintechab, cmap = truncate_colormap(cmap2,0,0.699), origin='lower', vmin=(-0.5), vmax=(7-0.5), interpolation=None, alpha = nndiff[k]**Colfac)
    im = globals()[plotnr[k]].imshow(mintechab, cmap = truncate_colormap(cmap2,0,0.699), origin='lower', vmin=(-0.5), vmax=(7-0.5), interpolation=None, alpha = nndiff[k]**Colfac)

    globals()[plotnr[k]].set_yticks(np.linspace(0,res-1,7), [0.25,1,4,16,64,256,1024])
    globals()[plotnr[k]].set_xticks(np.linspace(0,res-1,5), [1,10,100,1000,10000])
    for i in range(5):
        globals()[plotnr[k]].plot(np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        globals()[plotnr[k]].annotate(str(i+1),xy = (np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200))),color = "black", xytext=(5, -4.5), textcoords='offset points')        
plt.show()
plt.figure(20)
cbar = fig.colorbar(ima, ax = ax3, ticks= [a for a in range(7)] )
cbar.ax.set_yticklabels([Tech_names[0],Tech_names[1],Tech_names[9],Tech_names[6],Tech_names[14],Tech_names[8],Tech_names[12]])

colorbarplot = np.array([[0,0,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]])
alphaplot = np.array([[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05],[0.8,0.6,0.4,0.2,0.1,0.05]])
ax8.imshow(colorbarplot, cmap = truncate_colormap(cmap2,0,0.499), alpha = alphaplot**Colfac)
ax8.set_title("LCOS difference \n with second cheapest solution")
ax8.set_xticks(ticks = np.linspace(0,5,6),labels = ["80%","60%","40%","20%","10%","5%"])
ax8.set_yticks(ticks = np.linspace(0,6,7),labels = [Tech_names[0],Tech_names[1],Tech_names[9],Tech_names[6],Tech_names[14],Tech_names[8],Tech_names[12]])

"Min LCOS"

plt.figure(len(Tech_names)+3)
fig, ((ax3, ax4) ,(ax5, ax6) , (ax7,ax8)) = plt.subplots(3, 2, figsize = (12,10), sharex=True, sharey=True)

plotnr = ["ax3","ax4","ax5","ax6","ax7", "ax8", "ax9", "ax10", "ax11"]

fig.text(0.5, 0.04, "Yearly cycles [#]", ha='center')
fig.text(0.04, 0.5, "E/P ratio or discharge time [h]", va='center', rotation='vertical')

if PP_advanced == True:
    plt.suptitle("LCOS of cheapest tech \n RLPCP Algorithm \n ROI = " + str(str(ROI_in)))
else:
    plt.suptitle("LCOS of cheapest ech \n Buy-in fixed at: "+ str(PP_fixprice)+ "$/MWh \n ROI = " + str(str(ROI_in)))



for k in range(6):
    globals()[plotnr[k]].set_title(Case[k])
    
    if k == 0:
        ima = globals()[plotnr[k]].contourf(minval[k],levels=1000*levels, cmap = "plasma_r", vmin=1000*minlcos, vmax=1000*maxlcos, norm=colors.LogNorm())
    im = globals()[plotnr[k]].contourf(minval[k],levels=1000*levels, cmap = "plasma_r", origin='lower', vmin=1000*minlcos, vmax=1000*maxlcos, norm=colors.LogNorm())
    for i in range(5):
        globals()[plotnr[k]].plot(np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200)),color = "black",marker = "o")
        globals()[plotnr[k]].annotate(str(i+1),xy = (np.interp(pointsYC[i],YCycles_arr,np.linspace(0,199,200)), np.interp(pointsED[i],EP_arr,np.linspace(0,199,200))),color = "black", xytext=(5, -4.5), textcoords='offset points')        
plt.show()

cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(ima, cax=cb_ax)

plt.colorbar(ima,label = "LCOS [$/MWh]", ax = ax8)
