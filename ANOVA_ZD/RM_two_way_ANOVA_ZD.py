# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:12:32 2024

@author: csoport33g
"""


import matplotlib.pyplot as plt
import pandas as pd
import dabest
from scipy import stats
import os
import pingouin as pg
import statsmodels.stats.anova as sm

   
   
plt.close('all')
plt.ion()

path=r"H:\DENDRITE\_Revision\Revision_dataset_for_figures\zd\all_data_df_color_column_manually_added_ZD.xlsx"

df=pd.read_excel(path)
out_path=r"H:\DENDRITE\_Revision\Revision_dataset_for_figures\zd"


df.Amplitude_dendrite=df.Amplitude_dendrite*1000

df_control=df[df.experiment=='control']
df_ttx=df[df.experiment=='zd']

merged_df = df_control.merge(df_ttx, how = 'inner', on = ['cell_id'], suffixes=("_control","_zd"))


my_color_palette = {
                    1 : "cornflowerblue",
                    2 : "salmon",
                    3 : "blue",
                    4 : "red"
                    }


y_lim=(merged_df[["Speed_peak_control","Speed_peak_zd"]].min().min()*0.9 , merged_df[["Speed_peak_control","Speed_peak_zd"]].max().max()*1.1)
merged_df['speed_change']=merged_df.Speed_onset_zd-merged_df.Speed_onset_control
merged_df['speed_change_halfAmp']=merged_df.Speed_half_amp_zd-merged_df.Speed_half_amp_control
merged_df['speed_change_peak']=merged_df.Speed_peak_zd-merged_df.Speed_peak_control


small_df=pd.DataFrame()

small_df['cell_id'] = df.cell_id
small_df['species'] = df['Hum/Rat']
small_df['experiment'] = df.experiment
small_df['Speed_peak'] = df.Speed_peak


rm_anova_results = pg.rm_anova(data=small_df, dv='Speed_peak',within=['species','experiment'], subject='cell_id', detailed=True)
print(rm_anova_results)


# %%
############################

import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Speed_peak ~ C(species) + C(experiment) + C(species):C(experiment)', data=small_df).fit()
anova_table = sm.stats.anova_lm(model, typ=3)
print(anova_table)


anova_table.to_excel(path.replace('.xlsx','_tw_rm_ANOVA.xlsx'))


from bioinfokit.analys import stat

res = stat()
res.tukey_hsd(df=small_df, res_var='Speed_peak', xfac_var=['experiment','species'], anova_model='Speed_peak ~ C(species) + C(experiment) + C(species):C(experiment)')
print('post hoc Tukey HSD:  ',res.tukey_summary)

fos=res.tukey_summary
fos.to_excel(path.replace('.xlsx','_posthoc_Tukey_HSD_results.xlsx'))