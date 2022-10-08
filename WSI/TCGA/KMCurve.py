# encoding: utf-8
from cmath import nan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from PIL import Image
import kaplanmeier as km

def merge_tnm(x):
    if x=='I' or x =='IA' or x=='IB':
        return 'I'
    elif x=='IIA' or x=='IIB' or x=='II':
        return 'II'
    elif x=='IIIA' or x=='IIIC' or x=='IIIB' or x=='III':
        return 'III'
    elif x=='X' or x=='IV':
        return 'IV and V'
    else:
        return x
def main():
    file_path = '/data/pycode/MedIR/WSI/data/tcga_brca_clinical.csv'
    tnm_data = pd.read_csv(file_path, sep=',')
    tnm_data = tnm_data[['vital_status','days_to_death','days_to_last_follow_up','tnm']]
    tnm_data["vital_status"] = tnm_data["vital_status"].apply(lambda x: 0 if x == 'Alive' else 1)
    tnm_data['days'] = tnm_data.apply(lambda x: x['days_to_death'] if x['vital_status']==1 else x['days_to_last_follow_up'], axis=1)
    tnm_data = tnm_data.drop(['days_to_death','days_to_last_follow_up'], axis=1)
    tnm_data = tnm_data.dropna()
    tnm_data = tnm_data.drop(tnm_data[tnm_data['tnm']=='no'].index)
    tnm_data['days'] = tnm_data['days']/30 #turn day to month
    tnm_data['tnm'] = tnm_data['tnm'].apply(lambda x: merge_tnm(x))
    tnm_data = tnm_data.reset_index().drop('index', axis=1)
    #print(tnm_data['tnm'].value_counts())

    # Compute Survival
    results = km.fit(tnm_data['days'], tnm_data['vital_status'], tnm_data['tnm'])
    # Plot
    km.plot(results,savepath='/data/pycode/MedIR/WSI/imgs/brca_km.png', cii_lines='dense',cii_alpha=0.1, title='Kaplan-Meier Curve', width=10, height=6 )

if __name__ == '__main__':
    main()