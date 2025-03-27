
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
from scipy import signal as sig
import sys

plt.close('all')
# plt.ioff()

path=r"H:\DENDRITE\Electrophysiology\_Control_recordings\APs"
use_automatic_channel_definition=False
somatic_channel=2           
axonal_channel=1            

use_filter=True
low_pass_filter=4000


files = [f for f in listdir(path) if f.endswith('selected_aps.xlsx') if isfile(join(path, f))]

f=0

df=pd.DataFrame(columns=['File',
                        'AP_num_in_sweep',
                        'Onset_time_dendrite',
                        'Onset_Vm_dendrite',
                        'Peak_time_dendrite',
                        'Peak_Vm_dendrite',
                        'HalfAmp_time_dendrite',
                        'HalfAmp_Vm_dendrite',
                        'dMax_dendrite',
                        'Amplitude_dendrite',
                        'Halfwidth_dendrite',
                        'HalfIntergral_time_dendrite',
                        
                        'Onset_time_soma',
                        'Onset_Vm_soma',
                        'Peak_time_soma',
                        'Peak_Vm_soma',
                        'HalfAmp_time_soma',
                        'HalfAmp_Vm_soma',
                        'dMax_soma',
                        'Amplitude_soma',
                        'Halfwidth_soma',
                        'HalfIntergral_time_soma',
                       
                        'AP_attenuation',
                        'Latency_at_halfAmp',
                        'Latency_at_halfIntegral',
                        'Latency_at_peak',
                        'Latency_at_onset',
                        'Flag'])
for file_path in files:
    try:      
        print(' ')  
        print(str(np.shape(files)[0]-1)+ '/'+ str(f))
        print(file_path)
        
        
        
        mean_ap=pd.read_excel(os.path.join(path,file_path))
        data=np.array(mean_ap)
        
        spike_to_read=0
        latencies=[]
        length_of_AP=np.argmax(data[:,0])
        sampling_interval=data[1,0]
       
        
        if use_automatic_channel_definition==True:
            if np.amax(data[0:length_of_AP,5])-np.mean(data[0:100,5]) > np.amax(data[0:length_of_AP,2])-np.mean(data[0:100,2]):
                      dendritic_index=2
                      somatic_index=5
            else:
                      dendritic_index=5
                      somatic_index=2
        else:
            if somatic_channel==1:
                somatic_index=6
            if somatic_channel==2:
                somatic_index=3
                
            if axonal_channel==2:
                dendritic_index=3
            if axonal_channel==1:
                dendritic_index=6
        
        
        
        
        default_somatic_index=somatic_index
        for  spike_to_read in range(int(np.shape(data)[0]/(length_of_AP+1))):
          try:
            # print(spike_to_read)
            somatic_index=default_somatic_index
            if any(data[spike_to_read*length_of_AP+spike_to_read :spike_to_read*length_of_AP+length_of_AP+spike_to_read ,8])!=0.0:
                somatic_index=8
            
            # plt.close('all')
            intercept_point=0
            
            
            filter_settings=(low_pass_filter/(((1/sampling_interval)/2)))     
            b,a=sig.bessel(2,filter_settings,analog=False)
            points_to_ignore=int(filter_settings*((1/sampling_interval)/(low_pass_filter/10)))
            
            
            common_x=data[spike_to_read*length_of_AP+spike_to_read :spike_to_read*length_of_AP+length_of_AP+spike_to_read , 0]
            
            ap_num_in_sweep=data[(spike_to_read*length_of_AP+spike_to_read)+1, 10]+1
            sweep_num=data[(spike_to_read*length_of_AP+spike_to_read)+1, 9]+1
            
            
            soma_signal=data[spike_to_read*length_of_AP+spike_to_read :spike_to_read*length_of_AP+length_of_AP+spike_to_read ,somatic_index]
            soma_signal=soma_signal-np.percentile(soma_signal,10)
            
            if use_filter==True:
                soma_signal=sig.lfilter(b,a,soma_signal)
                soma_signal=soma_signal[points_to_ignore:-1]
                common_x=common_x[points_to_ignore:-1]
            
            soma_amplitude=np.amax(soma_signal)-np.percentile(soma_signal,10)
            soma_dMax=np.max(np.gradient(soma_signal, sampling_interval))
            
            peak_time_soma=common_x[np.argmax(soma_signal)]
            peak_vm_soma=np.amax(soma_signal)
            halfamp_time_soma=common_x[(np.where(soma_signal>=soma_amplitude/2)[0][0])]
            halfamp_vm_soma=soma_signal[(np.where(soma_signal>=soma_amplitude/2)[0][0])]
            
            soma_width_at_halfamp= common_x[np.where(soma_signal>=soma_amplitude/2)[0][-1]] -halfamp_time_soma
            
            baseline_fit_start=np.argmax(soma_signal)-(np.argmax(soma_signal)-int(0.001/sampling_interval))
            baseline_fit_end=baseline_fit_start+int((0.001/4)/sampling_interval)
            base_m,base_b=np.polyfit(common_x[baseline_fit_start:baseline_fit_end], 
                                soma_signal[baseline_fit_start:baseline_fit_end],1)
            rising_fit_start=baseline_fit_start+(np.where(soma_signal[baseline_fit_start:np.argmax(soma_signal)] >= (np.mean(soma_signal[baseline_fit_start:baseline_fit_end])+soma_amplitude*0.1) )[0][0])
            rising_fit_end=baseline_fit_start+(np.where(soma_signal[baseline_fit_start:np.argmax(soma_signal)] >= (np.mean(soma_signal[baseline_fit_start:baseline_fit_end])+soma_amplitude*0.3) )[0][0])
            rising_m,rising_b=np.polyfit(common_x[rising_fit_start:rising_fit_end], 
                                soma_signal[rising_fit_start:rising_fit_end],1)
            soma_baseline_fit=base_m*common_x+base_b
            soma_rising_fit=rising_m*common_x+rising_b
        
            try:
                intercept_point=np.where(soma_rising_fit>=soma_baseline_fit)[0][0]
                onset_time_soma=common_x[intercept_point]
                onset_vm_soma=soma_signal[intercept_point]
            except:
                pass
            intercept_point=0
            
            dendrite_signal=data[spike_to_read*length_of_AP+spike_to_read :spike_to_read*length_of_AP+length_of_AP+spike_to_read ,dendritic_index]
            dendrite_signal=dendrite_signal-np.percentile(dendrite_signal,10)
            
            if use_filter==True:
                dendrite_signal=sig.lfilter(b,a,dendrite_signal)
                dendrite_signal=dendrite_signal[points_to_ignore:-1]
                
            
            dendrite_amplitude=np.amax(dendrite_signal)-np.percentile(dendrite_signal,10)
            dendrite_dMax=np.max(np.gradient(dendrite_signal, sampling_interval))
            
            peak_time_dendrite=common_x[np.argmax(dendrite_signal)]
            peak_vm_dendrite=np.amax(dendrite_signal)
            halfamp_time_dendrite=common_x[(np.where(dendrite_signal>=dendrite_amplitude/2)[0][0])]
            halfamp_vm_dendrite=dendrite_signal[(np.where(dendrite_signal>=dendrite_amplitude/2)[0][0])]
            
            dendrite_width_at_halfamp= common_x[np.where(dendrite_signal>=dendrite_amplitude/2)[0][-1]] -halfamp_time_dendrite
            
            baseline_fit_start=np.argmax(dendrite_signal)-(np.argmax(dendrite_signal)-int(0.001/sampling_interval))
            baseline_fit_end=baseline_fit_start+int((0.001/4)/sampling_interval)
            base_m,base_b=np.polyfit(common_x[baseline_fit_start:baseline_fit_end], 
                                dendrite_signal[baseline_fit_start:baseline_fit_end],1)
            rising_fit_start=baseline_fit_start+(np.where(dendrite_signal[baseline_fit_start:np.argmax(dendrite_signal)] >= (np.mean(dendrite_signal[baseline_fit_start:baseline_fit_end])+dendrite_amplitude*0.1) )[0][0])
            rising_fit_end=baseline_fit_start+(np.where(dendrite_signal[baseline_fit_start:np.argmax(dendrite_signal)] >= (np.mean(dendrite_signal[baseline_fit_start:baseline_fit_end])+dendrite_amplitude*0.3) )[0][0])
            rising_m,rising_b=np.polyfit(common_x[rising_fit_start:rising_fit_end], 
                                dendrite_signal[rising_fit_start:rising_fit_end],1)
            dendrite_baseline_fit=base_m*common_x+base_b
            dendrite_rising_fit=rising_m*common_x+rising_b
            
            try:
                intercept_point=np.where(dendrite_rising_fit>=dendrite_baseline_fit)[0][0]
                onset_time_dendrite=common_x[intercept_point]
                onset_vm_dendrite=dendrite_signal[intercept_point]
            except:
                    pass
            
            ap_attenuation=dendrite_amplitude/soma_amplitude
        
            integral_limit1_som=int(np.argmax(soma_signal)-0.001/sampling_interval)
            try:
             integral_limit2_som=int(np.where(soma_signal[np.argmax(soma_signal):]<=soma_signal[int(np.argmax(soma_signal)-0.001/sampling_interval)])[0][0]+np.argmax(soma_signal))
            except:
             integral_limit2_som=soma_signal.size
             
            soma_ix = np.linspace(integral_limit1_som , integral_limit2_som, num=np.shape(soma_signal[integral_limit1_som:integral_limit2_som])[0])
            soma_iy = soma_signal[integral_limit1_som:integral_limit2_som]
            
            integral_limit1_dend=int(np.argmax(dendrite_signal)-0.001/sampling_interval)
            try:
             integral_limit2_dend=int(np.where(dendrite_signal[np.argmax(dendrite_signal):]<=dendrite_signal[int(np.argmax(dendrite_signal)-0.001/sampling_interval)])[0][0]+np.argmax(dendrite_signal))
            except:
             integral_limit2_dend=dendrite_signal.size
             
            dendrite_ix = np.linspace(integral_limit1_dend , integral_limit2_dend, num=np.shape(dendrite_signal[integral_limit1_dend:integral_limit2_dend])[0])
            dendrite_iy = dendrite_signal[integral_limit1_dend:integral_limit2_dend]
            
            soma_integral=integrate.simps(soma_iy,soma_ix)
            dendrite_integral=integrate.simps(dendrite_iy,dendrite_ix)
            
            
            i=1
            soma_fraction_integral=0
            while soma_fraction_integral <= soma_integral/2:
                soma_fraction_ix = np.linspace(integral_limit1_som , integral_limit1_som+i, num=np.shape(soma_signal[integral_limit1_som:integral_limit1_som+i])[0])
                soma_fraction_iy = soma_signal[integral_limit1_som:integral_limit1_som+i]
                soma_fraction_integral=integrate.simps(soma_fraction_iy,soma_fraction_ix)
                i=i+1
            
            j=1
            dendrite_fraction_integral=0
            while dendrite_fraction_integral <= dendrite_integral/2:
                dendrite_fraction_ix = np.linspace(integral_limit1_dend , integral_limit1_dend+j, num=np.shape(dendrite_signal[integral_limit1_dend:integral_limit1_dend+j])[0])
                dendrite_fraction_iy = dendrite_signal[integral_limit1_dend:integral_limit1_dend+j]
                dendrite_fraction_integral=integrate.simps(dendrite_fraction_iy,dendrite_fraction_ix)
                j=j+1
            
            latency_at_halfintegral =(common_x[j+integral_limit1_dend])-(common_x[i+integral_limit1_som])
            latency_at_halfamp=halfamp_time_dendrite-halfamp_time_soma
            latency_at_onset=onset_time_dendrite-onset_time_soma
            latency_at_peak=peak_time_dendrite-peak_time_soma
            
            df.at[f ,'File']=os.path.join(path,file_path)
            df.at[f ,'AP_num_in_sweep']=ap_num_in_sweep
            df.at[f ,'Onset_time_dendrite']=onset_time_dendrite
            df.at[f ,'Onset_Vm_dendrite']=onset_vm_dendrite
            df.at[f ,'Peak_time_dendrite']=peak_time_dendrite
            df.at[f ,'Peak_Vm_dendrite']=peak_vm_dendrite
            df.at[f ,'HalfAmp_time_dendrite']=halfamp_time_dendrite
            df.at[f ,'HalfAmp_Vm_dendrite']=halfamp_vm_dendrite
            df.at[f ,'dMax_dendrite']=dendrite_dMax
            df.at[f ,'Amplitude_dendrite']=dendrite_amplitude
            df.at[f ,'Halfwidth_dendrite']=dendrite_width_at_halfamp
            df.at[f ,'HalfIntergral_time_dendrite']=common_x[j]
            df.at[f ,'Onset_time_soma']=onset_time_soma
            df.at[f ,'Onset_Vm_soma']=onset_vm_soma
            df.at[f ,'Peak_time_soma']=peak_time_soma
            df.at[f ,'Peak_Vm_soma']=peak_vm_soma
            df.at[f ,'HalfAmp_time_soma']=halfamp_time_soma
            df.at[f ,'HalfAmp_Vm_soma']=halfamp_vm_soma
            df.at[f ,'dMax_soma']=soma_dMax
            df.at[f ,'Amplitude_soma']=soma_amplitude
            df.at[f ,'Halfwidth_soma']=soma_width_at_halfamp
            df.at[f ,'HalfIntergral_time_soma']=common_x[i]
            df.at[f ,'AP_attenuation']=ap_attenuation
            df.at[f ,'Latency_at_halfAmp']=latency_at_halfamp
            df.at[f ,'Latency_at_halfIntegral']=latency_at_halfintegral
            df.at[f ,'Latency_at_peak']=latency_at_peak
            df.at[f ,'Latency_at_onset']=latency_at_onset
            
            
            
            fig1, axs =plt.subplots(3,1, sharex=True)
            fig1.suptitle('sweep: Mean of first APs'+' AP:'+str(int(ap_num_in_sweep))+
                              '\n'+ 'latency at half max='+str( "%.1f" % (latency_at_halfamp *1000000))+'us'+
                              '\n'+'latency at half integral='+str( "%.1f" % (latency_at_halfintegral *1000000))+ 'us'+
                              '\n'+'latency at onset='+str( "%.1f" % (latency_at_onset *1000000))+ 'us'+
                              '\n'+'latency at peak='+str( "%.1f" % (latency_at_peak *1000000))+ 'us')
            if somatic_index!=8:
                fig1.suptitle('sweep:'+ str(int(sweep_num))+' AP:'+str(int(ap_num_in_sweep))+
                              '\n'+ 'latency at half max='+str( "%.1f" % (latency_at_halfamp *1000000))+'us'+
                              '\n'+'latency at half integral='+str( "%.1f" % (latency_at_halfintegral *1000000))+ 'us'+
                              '\n'+'latency at onset='+str( "%.1f" % (latency_at_onset *1000000))+ 'us'+
                              '\n'+'latency at peak='+str( "%.1f" % (latency_at_peak *1000000))+ 'us'+
                              '\n'+'NO Rs COMPENSATION!')
            axs[0].plot(common_x,soma_signal, color='black',alpha=0.7, label='Somatic signal')
            axs[0].legend(fontsize=5,loc=2)
            axs[0].scatter(halfamp_time_soma,halfamp_vm_soma, color='red',alpha=0.5)
            axs[0].scatter(peak_time_soma,peak_vm_soma, color='green',alpha=0.5)
            axs[0].scatter(onset_time_soma,onset_vm_soma, color='blue',alpha=0.5)
            
            if use_filter==True:
                axs[0].fill((soma_ix+points_to_ignore)*sampling_interval ,soma_iy, "gray", alpha=0.5)
                axs[0].vlines(((integral_limit1_som+i+points_to_ignore)*sampling_interval ),0,soma_signal[integral_limit1_som+i],'black')
            else:
                 axs[0].fill(soma_ix*sampling_interval ,soma_iy, "gray", alpha=0.5)
                 axs[0].vlines(((integral_limit1_som+i)*sampling_interval ),0,soma_signal[integral_limit1_som+i],'black')
           
            axs[0].plot(common_x,soma_baseline_fit, '-' ,color='blue',alpha=0.1)
            axs[0].plot(common_x,soma_rising_fit, '-', color='gray', alpha=0.1)
            axs[0].set_ylim([np.min(soma_signal)-0.001,np.max(soma_signal)+0.001])
            
            
            axs[1].plot(common_x,dendrite_signal,color='red', alpha=0.7, label='Neurite signal')
            axs[1].legend(fontsize=5,loc=2)
            axs[1].scatter(halfamp_time_dendrite,halfamp_vm_dendrite, color='red',alpha=0.5)
            axs[1].scatter(peak_time_dendrite,peak_vm_dendrite, color='green',alpha=0.5)
            axs[1].scatter(onset_time_dendrite,onset_vm_dendrite, color='blue',alpha=0.5)
            
            if use_filter==True:
                axs[1].fill((dendrite_ix+points_to_ignore)*sampling_interval ,dendrite_iy, "red", alpha=0.3)
                axs[1].vlines(((integral_limit1_dend+j+points_to_ignore)*sampling_interval),0,dendrite_signal[integral_limit1_dend+j],'black')
            else:
                axs[1].fill(dendrite_ix*sampling_interval ,dendrite_iy, "red", alpha=0.3)
                axs[1].vlines(((integral_limit1_dend+j)*sampling_interval),0,dendrite_signal[integral_limit1_dend+j],'black')
            
            axs[1].plot(common_x,dendrite_baseline_fit, '-' ,color='blue',alpha=0.1)
            axs[1].plot(common_x,dendrite_rising_fit, '-', color='gray', alpha=0.1)
            axs[1].set_ylim([np.min(dendrite_signal)-0.001,np.max(dendrite_signal)+0.001])
            
            axs[2].plot(common_x,soma_signal, color='black',alpha=0.5, label='Somatic signal')
            axs[2].plot(common_x,dendrite_signal,color='red', alpha=0.2, label='Neurite signal')
            axs[2].plot(common_x,np.interp(dendrite_signal,(np.min(dendrite_signal),np.max(dendrite_signal)),(np.min(soma_signal),np.max(soma_signal))),color='red', alpha=0.5, label='Normalized neurite signal')
            axs[2].legend(fontsize=5,loc=2)
            fig1.tight_layout()
            
            df.at[f ,'Flag']='epic'
            f=f+1
            fig1.savefig((r'H:\DENDRITE\Electrophysiology\_Control_recordings\APs\mean_APs' + '\\' + file_path + '.png'),dpi=150)
            
            #plt.close('all')
            # if spike_to_read in list_of_selected_aps:
            #     fig1.savefig((str(path_to_save+ '\\' + 'selected_aps'+ '\\')  + str(spike_to_read) + '.png'),dpi=150)
            
          except:
            df.at[f ,'File']=os.path.join(path,file_path)
            df.at[f ,'Flag']=sys.exc_info()
            print('Shit happened here!')
            print(sys.exc_info()[0])
            f=f+1
    
        
    except :
        df.at[f ,'File']=os.path.join(path,file_path)
        df.at[f ,'Flag']=sys.exc_info()
        print(sys.exc_info()[0])
        f=f+1
df.to_excel(r'H:\DENDRITE\Electrophysiology\_Control_recordings\APs\mean_APs\summary_of_mean_aps.xlsx')
     
end_time=time.time()
print(' ')
print('Total run cost: ')
print("--- %s seconds ---" % (end_time - start_time))
    
   