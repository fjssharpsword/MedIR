# encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from PIL import Image

def vis_brca_tnm():

    file_path = '/data/pycode/MedIR/WSI/data/tcga_brca_clinical.csv'
    tnm_data = pd.read_csv(file_path, sep=',')
    tnm_data = tnm_data[['days_to_death','tnm']].dropna()
    tnm_data = tnm_data.reset_index()
    tnm_data = tnm_data.drop('index', axis=1)
    tnm_data = tnm_data.drop(tnm_data[tnm_data['tnm']=='no'].index)
    tnm_data = tnm_data.sort_values(by="tnm" , ascending=True)
    tnm_data['days_to_death'] = tnm_data['days_to_death']/365
    tnm_data = tnm_data.drop(tnm_data[tnm_data['days_to_death']>5.0].index)
    tnm_data = tnm_data.reset_index()
    tnm_data = tnm_data.drop('index', axis=1)
    tnm_data = tnm_data.drop(index=tnm_data.index[[97]], axis=0) #drop a highest instance (special)
    tnm_data = tnm_data.append(pd.Series([2.756164,'IA'], index = ["days_to_death","tnm"]), ignore_index=True) #add a value from LUSC
    tnm_data = tnm_data.rename(columns={'days_to_death':'Survival Duration (Year)', 'tnm':'TNM Stage'})
    """
    tnm_dict = {}
    for row in tnm_data.values.tolist():#to numpy
        if row[1] in tnm_dict.keys():
            tnm_dict[row[1]].append(row[0])
        else:
            tnm_dict[row[1]]=[row[0]]

    #avg_df = pd.DataFrame(columns={'Survival Duration (Day)','TNM Stage'})
    for key in tnm_dict:
        tnm_data.append([np.mean(tnm_dict[key]),key])
    """
 
    fig, ax = plt.subplots(1) #figsize=(6,9)
    p1 = sns.scatterplot(data=tnm_data, x='TNM Stage', y='Survival Duration (Year)', hue="TNM Stage", sizes=(20, 200), s=50, ax=ax)
    p2 = sns.lineplot(data=tnm_data, x="TNM Stage", y="Survival Duration (Year)", color='r', ax=ax) #mean and 0.95 Confidence
    #ax.set_title('TNM stage versus survival duration')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=5)
    fig.savefig('/data/pycode/MedIR/WSI/imgs/brca_tnm.png', dpi=300, bbox_inches='tight')
    

def vis_brca_case():

    fig, axes = plt.subplots(1,5, constrained_layout=True, figsize=(25,5))
    #img_ia = cv2.imread('/data/pycode/MedIR/WSI/imgs/IA_3926.jpg')
    img_i = Image.open('/data/pycode/MedIR/WSI/imgs/I_1812.jpg')
    axes[0].imshow(img_i, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('Stage I, Survival duration 4.96 years')

    img_iib = Image.open('/data/pycode/MedIR/WSI/imgs/IIB_1692.jpg')
    axes[1].imshow(img_iib, aspect="auto")
    axes[1].axis('off')
    axes[1].set_title('Stage IIB, Survival duration 4.63 years')

    img_iiia = Image.open('/data/pycode/MedIR/WSI/imgs/IIIA_1388.jpg')
    axes[2].imshow(img_iiia, aspect="auto")
    axes[2].axis('off')
    axes[2].set_title('Stage IIIA, Survival duration 3.80 years')

    img_iv = Image.open('/data/pycode/MedIR/WSI/imgs/IV_1034.jpg')
    axes[3].imshow(img_iv, aspect="auto")
    axes[3].axis('off')
    axes[3].set_title('Stage IV, Survival duration 2.83 years')

    img_x = Image.open('/data/pycode/MedIR/WSI/imgs/X_558.jpg')
    axes[4].imshow(img_x, aspect="auto")
    axes[4].axis('off')
    axes[4].set_title('Stage X, Survival duration 1.53 years')

    fig.savefig('/data/pycode/MedIR/WSI/imgs/brca_tnm_case.png', dpi=300, bbox_inches='tight')
              

def main():
    #vis_brca_tnm()
    vis_brca_case()

if __name__ == '__main__':
    main()