# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:50:50 2021

@author: csoport33g
"""

import time
start_time = time.time()

import numpy as np
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
from os import listdir
from os.path import isfile, join
import scipy.signal as sig
import traceback

plt.close('all')
plt.ioff()

path=r"H:\DENDRITE"
files = [f for f in listdir(path) if f.endswith('.asc') if isfile(join(path, f))]

f=0
summary_df=pd.DataFrame()

for file_path in files:
  try:  
    print(' ')  
    print(str(np.shape(files)[0]-1)+ '/'+ str(f))
    print(file_path)
    cutted_aps=np.empty((0,11))
        
    
    ms_to_cut=6                    
    rs_compensation_us=130         
    
    automatic_channel_determination=False
    somatic_channel=2               
    dendritic_channel=5            
    ap_trheshold=-0.25          
    folder_for_analysis_output=(path+'\\APs')
    path_to_save=(os.path.join(path,folder_for_analysis_output))
    path_for_sweep_images=(path+'\\'+file_path.replace('.asc','_sweep_images'))
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(path_for_sweep_images):
        os.makedirs(path_for_sweep_images)
    
    file_path_to_save=(str(path_to_save)+ '\\' +(str(file_path).replace('.asc','_APs.asc')))
    
    data=np.loadtxt(path+ '\\' + file_path) #,delimiter=',')
    
    start_of_sweeps=np.where(data[:,0]==0)
    sweep_length=start_of_sweeps[0][1]-start_of_sweeps[0][0]
    sampling_interval=data[1,1]-data[0,1]
    
    if automatic_channel_determination==True:
     try:
        if np.max(data[:sweep_length,2])-np.percentile(data[:sweep_length,2],10) > np.max(data[:sweep_length,5])-np.percentile(data[:sweep_length,5],10):
             somatic_channel=2           
             dendritic_channel=5
        else:
            somatic_channel=5           
            dendritic_channel=2
     except:
         if np.max(data[:sweep_length,2])-np.percentile(data[:sweep_length,2],10) > np.max(data[:sweep_length,4])-np.percentile(data[:sweep_length,4],10):
              somatic_channel=2           
              dendritic_channel=4
         else:
             somatic_channel=4           
             dendritic_channel=2  
    # break
    df=pd.DataFrame(columns=['File_name','Sweep_num','V_drop','Rs','Rin','Resting_Vm'])
    for sweep_num in range(start_of_sweeps[0].shape[0]):
        # print('sweep_num: ', sweep_num)
        plt.close('all')
        df.at[sweep_num ,'File_name']=file_path
        df.at[sweep_num ,'Sweep_num']=sweep_num
        
        
        fig1=plt.figure(figsize=(16,9))
        gs=fig1.add_gridspec(5,4)
        f1_ax1=fig1.add_subplot(gs[0,0:3])
        f1_ax2=fig1.add_subplot(gs[1,0:3])
        f1_ax3=fig1.add_subplot(gs[2,0:3])
        f1_ax4=fig1.add_subplot(gs[3,0:3])
        f1_ax9=fig1.add_subplot(gs[4,0:3])
        
        f1_ax5=fig1.add_subplot(gs[0,3])
        f1_ax6=fig1.add_subplot(gs[1,3])
        f1_ax7=fig1.add_subplot(gs[2,3])
        f1_ax8=fig1.add_subplot(gs[3,3])
        f1_ax10=fig1.add_subplot(gs[4,3])
        
        sweep_data=data[start_of_sweeps[0][sweep_num]:start_of_sweeps[0][sweep_num]+sweep_length]
        soma_signal=sweep_data[:,somatic_channel]
        dendrite_signal=sweep_data[:,dendritic_channel]
        
        f1_ax1.plot(soma_signal,color='black',alpha=0.5)
        f1_ax2.plot(dendrite_signal,color='red',alpha=0.5)
        f1_ax4.plot(sweep_data[:,3])
        
        try:
            current_start=np.where(sweep_data[:,3]==np.max(sweep_data[:,3]))[0][0]
            current_end=np.where(sweep_data[:,3]==np.max(sweep_data[:,3]))[0][-1]
            current_step=np.max(sweep_data[:,3])-np.min(sweep_data[:,3])
            
            resting_vm=np.mean(soma_signal[0:current_start])
            
            f1_ax7.plot(soma_signal[current_start-int(0.001/sampling_interval):current_start+int(0.001/sampling_interval)])
            voltage_drop=np.mean(soma_signal[current_start+int(rs_compensation_us/1000000/sampling_interval)])-np.mean(soma_signal[current_start-int(rs_compensation_us/1000000/sampling_interval)])
            rs_value=voltage_drop/current_step
           
            compensated_soma_signal=np.concatenate((soma_signal[0:current_start], soma_signal[current_start:current_end]-voltage_drop, soma_signal[current_end:-1]))
            f1_ax3.plot(compensated_soma_signal,color='black',alpha=0.5)
            f1_ax3.hlines(np.percentile(compensated_soma_signal,50),0,sweep_length ,color='black',alpha=0.1)
            rin_value=(np.percentile(compensated_soma_signal,50) -np.mean(compensated_soma_signal[0:current_start]))/current_step
            
            APs=sig.find_peaks(compensated_soma_signal,height=ap_trheshold, prominence=0.05 ,width=int(0.1/sampling_interval/1000),distance=int(1/sampling_interval/1000))
            f1_ax3.scatter(APs[0],APs[1]['peak_heights'],color='red',alpha=0.5)
            f1_ax8.plot(compensated_soma_signal[current_start-int(0.001/sampling_interval):current_start+int(0.001/sampling_interval)])
            
            f1_ax9.plot(compensated_soma_signal,color='black',alpha=0.5)
            f1_ax9.plot(soma_signal,color='black',alpha=0.1)
            f1_ax9.plot(np.interp(dendrite_signal,(np.min(dendrite_signal),np.max(dendrite_signal)),(np.min(compensated_soma_signal),np.max(compensated_soma_signal))),color='red',alpha=0.5)
            
            label='Rs: {}MOhm\nRin: {}MOhm\nVdrop: {}mV\nResting Vm: {}'.format(rs_value/1000000,rin_value/1000000,voltage_drop*1000,resting_vm)
            f1_ax10.annotate(label,(0.1, 0.5),xycoords='axes fraction', va='center')
            
            
            df.at[sweep_num ,'V_drop']=voltage_drop
            df.at[sweep_num ,'Rs']=rs_value
            df.at[sweep_num ,'Rin']=rin_value
            df.at[sweep_num ,'Resting_Vm']=resting_vm
            
            for AP_num in range(0,APs[0].shape[0]):
                cutted_data=sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),0:7]
                cutted_compensated_soma_signal=compensated_soma_signal[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2)]
                
                # print('AP_num: ', AP_num)
                cutted_data=np.hstack((cutted_data,np.asarray(np.linspace(0,[(np.shape(cutted_data)[0]*sampling_interval)],np.shape(cutted_data)[0]),)))
                cutted_data=np.hstack((cutted_data,np.reshape(cutted_compensated_soma_signal, (-1, 1))))
                cutted_data=np.hstack((cutted_data,np.full((np.shape(cutted_data)[0],1),sweep_num)))
                cutted_data=np.hstack((cutted_data,np.full((np.shape(cutted_data)[0],1),AP_num)))
                cutted_aps=np.concatenate((cutted_aps,cutted_data),axis=0)
                
                f1_ax5.plot(cutted_data[:,somatic_channel])
                f1_ax6.plot(cutted_data[:,dendritic_channel])
            
            f1_ax10.get_xaxis().set_visible(False)
            f1_ax10.get_yaxis().set_visible(False)
            fig1.tight_layout()
            fig1.savefig((str(path_for_sweep_images) + '\\' + str(sweep_num) + '.png'),dpi=150)
        
        except :
            APs=sig.find_peaks(soma_signal,height=ap_trheshold,prominence=0.05 ,width=int(0.1/sampling_interval/1000),distance=int(1/sampling_interval/1000))
            f1_ax1.scatter(APs[0],APs[1]['peak_heights'],color='red',alpha=0.5)
            
            f1_ax9.plot(soma_signal,color='black',alpha=0.5)
            f1_ax9.plot(np.interp(dendrite_signal,(np.min(dendrite_signal),np.max(dendrite_signal)),(np.min(soma_signal),np.max(soma_signal))),color='red',alpha=0.5)
            
            df.at[sweep_num ,'V_drop']='nan'
            df.at[sweep_num ,'Rs']='nan'
            df.at[sweep_num ,'Rin']='nan'
            df.at[sweep_num ,'Resting_Vm']='nan'
            
            for AP_num in range(0,APs[0].shape[0]):
                # cutted_data=sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),0:7]
                cutted_data=sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),0]
                # cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),0]))
                cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),1]))
                cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),dendritic_channel]))
                cutted_data=np.vstack((cutted_data,np.zeros(cutted_data.shape[1])))
                cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),3]))
                cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),somatic_channel]))
                cutted_data=np.vstack((cutted_data,np.zeros(cutted_data.shape[1])))
                cutted_data=np.vstack((cutted_data,sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),1]-sweep_data[APs[0][AP_num]-int((ms_to_cut/sampling_interval/1000)/3):APs[0][AP_num]+(int((ms_to_cut/sampling_interval/1000)/3)*2),1][0]))
                cutted_data=np.vstack((cutted_data,np.zeros(cutted_data.shape[1])))

                print('AP_num: ', AP_num, '        Rs, R_in, V_drop could not been calculated!')
                # cutted_data=np.hstack((cutted_data,np.asarray(np.linspace(0,[(np.shape(cutted_data)[0]*sampling_interval)],np.shape(cutted_data)[0]))))
                # cutted_data=np.hstack((cutted_data,np.full((np.shape(cutted_data)[0],1),0)))
                cutted_data=np.vstack((cutted_data,np.full((np.shape(cutted_data)[1]),sweep_num)))
                cutted_data=np.vstack((cutted_data,np.full((np.shape(cutted_data)[1]),AP_num)))
                cutted_aps=np.concatenate((cutted_aps,cutted_data.T),axis=0)
                cutted_data=cutted_data.T
                
                f1_ax5.plot(cutted_data[:,somatic_channel])
                f1_ax6.plot(cutted_data[:,dendritic_channel])
                
            f1_ax10.get_xaxis().set_visible(False)
            f1_ax10.get_yaxis().set_visible(False)
            
            fig1.tight_layout()   
            fig1.savefig((str(path_for_sweep_images) + '\\' + str(sweep_num) + '.png'),dpi=150)
            
            
        
    np.savetxt(file_path_to_save,cutted_aps,delimiter='\t')
    df.to_excel((path_for_sweep_images+'\\'+file_path.replace('.asc','.xlsx')))
    summary_df.at[f ,'file_name']=file_path
    summary_df.at[f,'note']='epic'
    f=f+1
  except:
    summary_df.at[f ,'file_name']=file_path
    summary_df.at[f,'note']=traceback.format_exc()
    f=f+1
    
summary_df.to_excel(path+'\summary_df.xlsx')
end_time=time.time()
print(' ')
print('Total run cost: ')
print("--- %s seconds ---" % (end_time - start_time))
    